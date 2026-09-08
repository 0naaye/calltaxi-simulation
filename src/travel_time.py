"""
travel_time.py — 행정동 간 이동시간 추정

시뮬레이터의 공차 이동과 승차 운행에 사용할
행정동 간 이동시간(분)을 반환한다.

조회 순서
1. 실측 OD 중앙값
2. 직선거리·우회계수·거리대별 속도 기반 추정
3. 시간대 평균속도 기반 근사

동일 동 이동이나 좌표가 없는 경우에는 별도 fallback 값을 사용한다.

이동시간 프로필은 5개 시간대 × 평일/주말로 구성하며,
수요 분석에 사용하는 시간대 구분과는 별도로 관리한다.

사용
get_travel_time(origin, dest, hour, is_weekend)

실행
python src/travel_time.py
"""

from __future__ import annotations

import bisect
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from load import CACHE_DIR, CALLS_FILE, DATA_DIR, MANUAL_DONG, SEOUL_GU, canon_dong
from load import _base_dong as base_dong   # 좌표 매칭 규칙은 load.py 와 하나로 유지한다

__all__ = ["TravelTime", "get_travel_time", "load_rides", "build_od_table",
           "build_speed_table", "build_profile_table", "period_of", "PERIODS"]

# ─────────────────────────────────────────────────────────────
# 상수
# ─────────────────────────────────────────────────────────────

# 시간대 5구간. 경계는 "이 시각부터" — 0~6 심야, 7~9 오전피크, 10~15 낮,
# 16~18 오후피크, 19~23 저녁.
PERIOD_CUTS = (7, 10, 16, 19)
PERIODS = ("late_night", "morning_peak", "daytime", "afternoon_peak", "evening")
PERIOD_KO = {
    "late_night": "심야(0~6시)", "morning_peak": "오전피크(7~9시)",
    "daytime": "낮(10~15시)", "afternoon_peak": "오후피크(16~18시)",
    "evening": "저녁(19~23시)",
}

# 거리대 5구간(도로 주행거리 기준, km).
# 20km 컷은 실측으로 넣은 것이다 — 10km+ 한 칸에 10km 와 30km 가 같이 들어가는데
# 실측 속도가 10~15km 19.3km/h, 25km+ 29.2km/h 로 벌어져 장거리가 중앙 +12.5분
# 과대추정됐다(방화1동→길동 실측 73분 vs 추정 103분). 쪼개니 편의가 −0.7분으로 사라진다.
DIST_CUTS = (2.0, 5.0, 10.0, 20.0)
DIST_BANDS = ("~2km", "2-5km", "5-10km", "10-20km", "20km+")

MIN_TRIPS = 10          # OD 셀 신뢰 최소 건수(명세)
MIN_SPEED_TRIPS = 30    # 속도 셀 신뢰 최소 건수 — 셀이 50개뿐이라 넉넉히 잡는다
# 직선거리 → 도로 주행거리 우회계수.
#
# **1.36 → 1.33 (2026.08.09).** 설정값이 검산값과 어긋나 있었다. 1.36 은 초기
# 산출(평균 1.43 / 중앙 1.36)에서 정한 값인데, 데이터가 두 번 바뀌는 동안
# 검산값이 1.334 → 1.333 → **1.330** 으로 내려왔고 "왜 1.36 이냐"에 답할 근거가
# 남지 않았다. 검산은 30건 이상 OD쌍의 `중앙 승차거리 ÷ 중심점 직선거리` 중앙값이며
# `python src/travel_time.py` 의 [우회계수 검산] 줄에 찍힌다.
#
# **평균(1.386)이 아니라 중앙(1.330)을 쓴다** — 우회비는 오른쪽 꼬리가 길어
# (하천·산지를 도는 소수 OD쌍) 평균이 그쪽에 끌린다.
#
# 이 상수는 캐시 키(`_cache_paths`)에 넣지 않는다. od·speed·profile 테이블은 전부
# **실측 승차거리**로 만들어져 DETOUR 와 무관하고, DETOUR 는 조회 시점에만 곱해진다.
DETOUR = 1.33

# 이상치 컷 — 이 밖은 기록 오류로 보고 테이블 산출에서 뺀다
RIDE_MIN_RANGE = (1.0, 120.0)     # 운행시간(분)
SPEED_RANGE = (2.0, 60.0)         # 평균속도(km/h)

# 반환값 가드
MIN_MINUTES, MAX_MINUTES = 1.0, 120.0
MIN_INTRA_MINUTES = 5.0           # 동 내부 이동 최소값
FALLBACK_KMH = 16.0               # 프로필까지 비었을 때(서울 전체 실측 중앙)
FALLBACK_MINUTES = 20.0


# ─────────────────────────────────────────────────────────────
# 키 매핑
# ─────────────────────────────────────────────────────────────

def period_of(hour: int) -> str:
    """시각(0~23) → 시간대 키."""
    h = int(hour)
    if not 0 <= h <= 23:
        raise ValueError(f"hour 는 0~23 이어야 한다: {hour}")
    return PERIODS[bisect.bisect_right(PERIOD_CUTS, h)]


def period_of_array(hours) -> np.ndarray:
    """벡터판 period_of. 대량 조회(커버율 검증·시뮬 배치)용."""
    h = np.asarray(hours, dtype=np.int16)
    if h.size and (h.min() < 0 or h.max() > 23):
        raise ValueError("hour 는 0~23 이어야 한다")
    return np.asarray(PERIODS, dtype=object)[np.searchsorted(PERIOD_CUTS, h, side="right")]


def band_of(km: float) -> str:
    """거리(km) → 거리대 키."""
    return DIST_BANDS[bisect.bisect_right(DIST_CUTS, float(km))]


def band_of_array(km) -> np.ndarray:
    k = np.asarray(km, dtype=float)
    return np.asarray(DIST_BANDS, dtype=object)[np.searchsorted(DIST_CUTS, k, side="right")]


def haversine_km(lat1, lon1, lat2, lon2):
    """중심점 간 대권거리(km). 스칼라·배열 모두 받는다."""
    r = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(np.asarray(lon2, dtype=float) - np.asarray(lon1, dtype=float))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ─────────────────────────────────────────────────────────────
# 원본 → 실측 운행 표
# ─────────────────────────────────────────────────────────────

_RIDE_COLS = ["승차일시", "하차일시", "출발구", "출발동",
              "목적구", "목적동", "승차거리"]


def load_rides(path: Path = None) -> pd.DataFrame:
    """콜 원본(특장차 한정본)에서 이동시간 산출에 쓸 운행만 남긴다.

    필터(단계별 잔존 건수는 attrs['funnel'] 에 담는다):
      승차·하차 시각이 모두 기록 — 취소·미탑승 건을 뺀 완료 운행
      출발·목적 모두 서울 25구  — 경기·인천 편도는 뺀다
      운행시간 1~120분          — 하차 − 승차
      승차거리 > 0              — 0 은 결측 대체값이다(calibration.md 흡수 사실 ⑴). 완료 건 4,636건이
                                  여기서 빠진다 — 운행은 했으나 거리가 안 남은 건이라
                                  km·kmh 를 만들 수 없다
      평균속도 2~60km/h         — 기록 오류(지연 하차 처리 등) 제거

    반환 컬럼: origin, dest(표준형 동명), minutes, km, kmh, hour, period, is_weekend
    """
    path = Path(path) if path else DATA_DIR / CALLS_FILE
    df = pd.read_csv(path, encoding="utf-8-sig", usecols=_RIDE_COLS, low_memory=False)
    funnel = {"원본": len(df)}

    board = pd.to_datetime(df["승차일시"], format="ISO8601", errors="coerce")
    alight = pd.to_datetime(df["하차일시"], format="ISO8601", errors="coerce")
    ok = board.notna() & alight.notna()
    df, board, alight = df[ok], board[ok], alight[ok]
    funnel["승차·하차 기록"] = len(df)

    ok = df["출발구"].isin(SEOUL_GU) & df["목적구"].isin(SEOUL_GU)
    df, board, alight = df[ok], board[ok], alight[ok]
    funnel["서울 내"] = len(df)

    minutes = (alight - board).dt.total_seconds() / 60
    km = pd.to_numeric(df["승차거리"], errors="coerce") / 1000

    ok = minutes.between(*RIDE_MIN_RANGE) & (km > 0)
    df, board, minutes, km = df[ok], board[ok], minutes[ok], km[ok]
    funnel["시간·거리 정상"] = len(df)

    kmh = km / (minutes / 60)
    ok = kmh.between(*SPEED_RANGE)
    df, board, minutes, km, kmh = df[ok], board[ok], minutes[ok], km[ok], kmh[ok]
    funnel["속도 정상"] = len(df)

    out = pd.DataFrame({
        "origin": df["출발동"].map(canon_dong),
        "dest": df["목적동"].map(canon_dong),
        "minutes": minutes.astype("float32").to_numpy(),
        "km": km.astype("float32").to_numpy(),
        "kmh": kmh.astype("float32").to_numpy(),
        "hour": board.dt.hour.astype("int8").to_numpy(),
        "is_weekend": (board.dt.weekday >= 5).to_numpy(),
    })
    out["period"] = period_of_array(out["hour"].to_numpy())
    out.attrs["funnel"] = funnel
    return out.reset_index(drop=True)


def load_centroids(path: Path = None) -> pd.DataFrame:
    """동명(표준형) → 중심좌표 조회표. 구 없이 동명만으로 찾는다.

    콜 원본과 중심점 파일의 동명이 어긋나는 걸 3종 키로 흡수한다(load.py 와 동일 규칙):

      canon  표준형 그대로 — '이태원제1동' → '이태원1동'
      base   숫자·꼬리 '동' 을 뗀 기본명 그룹의 중심점 평균. 분동/합동 대응 —
             콜의 '상일동' 이 경계상 '상일1동+상일2동' 인 경우, '공릉1·3동' 처럼
             가운뎃점이 빠져 '공릉13동' 이 되는 경우
      alias  행정구역 개편으로 개명된 동(신설동·용두동 → 용신동 등). load.MANUAL_DONG

    조회는 canon 먼저, 실패하면 base. base 키는 꼬리 '동' 이 없어 canon 키와
    문자열이 겹치지 않으므로 한 표에 같이 담는다.

    `신사동`(강남·관악)처럼 구가 다른 동명이 겹치면 중심점을 평균한다.
    get_travel_time 시그니처가 동명 하나뿐이라 구를 구분할 수 없고, 겹치는 이름은
    소수라 2·3순위 근사에서만 오차로 남는다(attrs['n_ambiguous']).

    반환: dong, lat, lon, key_kind, n_src
    """
    path = Path(path) if path else DATA_DIR / "행정동_중심점.csv"
    cen = pd.read_csv(path, encoding="utf-8-sig")
    parts = cen["adm_nm"].str.split(" ", expand=True)
    cen["dong"] = parts.iloc[:, 2:].fillna("").agg("".join, axis=1).map(canon_dong)

    agg = dict(lat=("lat", "mean"), lon=("lon", "mean"), n_src=("adm_nm", "size"))
    canon = cen.groupby("dong", as_index=False).agg(**agg).assign(key_kind="canon")
    base = (cen.assign(dong=cen["dong"].map(base_dong))
               .groupby("dong", as_index=False).agg(**agg).assign(key_kind="base"))

    # 개명 동 — 원래 이름으로 들어와도 새 동 좌표를 쓰게 한다
    coord = canon.set_index("dong")
    alias = []
    for (_, src), target in MANUAL_DONG.items():
        s, t = canon_dong(src), canon_dong(target)
        if t in coord.index and s not in coord.index:
            alias.append({"dong": s, "lat": coord.at[t, "lat"], "lon": coord.at[t, "lon"],
                          "n_src": 1, "key_kind": "alias"})
    alias = pd.DataFrame(alias, columns=canon.columns)

    out = (pd.concat([canon, alias, base], ignore_index=True)
             .drop_duplicates(subset="dong", keep="first")   # canon > alias > base
             .reset_index(drop=True))
    out.attrs["n_ambiguous"] = int((canon["n_src"] > 1).sum())
    out.attrs["n_canon"] = len(canon)
    return out


# ─────────────────────────────────────────────────────────────
# 3단 테이블 산출
# ─────────────────────────────────────────────────────────────

def build_od_table(rides: pd.DataFrame, min_trips: int = MIN_TRIPS) -> pd.DataFrame:
    """1순위 — (출발동, 목적동, 시간대, 주말) 별 운행시간 중앙값.

    min_trips 미만 셀은 표에 넣지 않는다(2순위로 흘려보낸다).
    반환: origin, dest, period, is_weekend, minutes, n
    """
    g = (rides.groupby(["origin", "dest", "period", "is_weekend"], observed=True)
              .agg(minutes=("minutes", "median"), n=("minutes", "size"))
              .reset_index())
    out = g[g["n"] >= min_trips].reset_index(drop=True)
    out.attrs["n_cells_all"] = len(g)
    return out


def build_speed_table(rides: pd.DataFrame, min_trips: int = MIN_SPEED_TRIPS) -> pd.DataFrame:
    """2순위 — (거리대, 시간대, 주말) 별 중앙 속도.

    거리대는 실측 `승차거리`(도로거리) 기준으로 나눈다. 조회할 때도
    직선거리 × DETOUR(도로거리 추정) 로 거리대를 정해 기준을 맞춘다.
    반환: band, period, is_weekend, kmh, n
    """
    df = rides.assign(band=band_of_array(rides["km"].to_numpy()))
    g = (df.groupby(["band", "period", "is_weekend"], observed=True)
           .agg(kmh=("kmh", "median"), n=("kmh", "size"))
           .reset_index())
    return g[g["n"] >= min_trips].reset_index(drop=True)


def build_profile_table(rides: pd.DataFrame) -> pd.DataFrame:
    """3순위·특수 케이스 — (시간대, 주말) 10개 프로필.

    kmh       시간대 평균속도(3순위 직선거리 근사용)
    intra_min 출발동 == 목적동 실측 중앙값(동 내부 이동). 표본 부족하면 5분.
    minutes   전체 운행시간 중앙값(좌표를 못 찾는 동의 최후 폴백)
    """
    g = (rides.groupby(["period", "is_weekend"], observed=True)
              .agg(kmh=("kmh", "median"), minutes=("minutes", "median"), n=("kmh", "size"))
              .reset_index())

    same = rides[rides["origin"] == rides["dest"]]
    intra = (same.groupby(["period", "is_weekend"], observed=True)
                 .agg(intra_min=("minutes", "median"), n_intra=("minutes", "size"))
                 .reset_index())

    out = g.merge(intra, on=["period", "is_weekend"], how="left")
    thin = out["n_intra"].isna() | (out["n_intra"] < MIN_TRIPS)
    out.loc[thin, "intra_min"] = MIN_INTRA_MINUTES
    out["n_intra"] = out["n_intra"].fillna(0).astype(int)
    out["intra_min"] = out["intra_min"].clip(lower=MIN_INTRA_MINUTES)
    return out


# ─────────────────────────────────────────────────────────────
# 조회기
# ─────────────────────────────────────────────────────────────

TIERS = ("od", "speed", "line", "intra", "default")
TIER_KO = {
    "od": "1순위 실측 OD", "speed": "2순위 속도테이블", "line": "3순위 직선근사",
    "intra": "동 내부 이동", "default": "좌표없음 폴백",
}


class TravelTime:
    """이동시간 조회기. `TravelTime.build()` 로 만들고 `.lookup()` 으로 쓴다."""

    def __init__(self, od: pd.DataFrame, speed: pd.DataFrame,
                 profile: pd.DataFrame, centroids: pd.DataFrame):
        self.od, self.speed, self.profile, self.centroids = od, speed, profile, centroids

        self._od = {(r.origin, r.dest, r.period, bool(r.is_weekend)): float(r.minutes)
                    for r in od.itertuples(index=False)}
        self._speed = {(r.band, r.period, bool(r.is_weekend)): float(r.kmh)
                       for r in speed.itertuples(index=False)}
        self._profile = {(r.period, bool(r.is_weekend)):
                         (float(r.kmh), float(r.intra_min), float(r.minutes))
                         for r in profile.itertuples(index=False)}
        self._coord = {r.dong: (float(r.lat), float(r.lon))
                       for r in centroids.itertuples(index=False)}

    # ── 조회 ──────────────────────────────────────────────

    def coord_of(self, dong: str):
        """표준형 동명 → (lat, lon). 없으면 기본명으로 한 번 더, 그래도 없으면 None."""
        return self._coord.get(dong) or self._coord.get(base_dong(dong))

    def lookup(self, origin: str, dest: str, hour: int,
               is_weekend: bool = False) -> tuple[float, str]:
        """(이동시간 분, 사용한 순위 키). 순위 키는 TIERS 참조."""
        o, d = canon_dong(origin), canon_dong(dest)
        period, wk = period_of(hour), bool(is_weekend)

        minutes = self._od.get((o, d, period, wk))
        if minutes is not None:
            return _clip(minutes), "od"

        kmh_p, intra_min, med_min = self._profile.get(
            (period, wk), (FALLBACK_KMH, MIN_INTRA_MINUTES, FALLBACK_MINUTES))

        if o == d:
            return _clip(intra_min), "intra"

        co, cd = self.coord_of(o), self.coord_of(d)
        if co is None or cd is None:
            return _clip(med_min), "default"

        km = float(haversine_km(co[0], co[1], cd[0], cd[1])) * DETOUR
        if km <= 0:
            return _clip(intra_min), "intra"

        kmh, tier = self._speed.get((band_of(km), period, wk)), "speed"
        if kmh is None:
            kmh, tier = kmh_p, "line"
        return _clip(km / kmh * 60), tier

    def minutes(self, origin: str, dest: str, hour: int, is_weekend: bool = False) -> float:
        """분만 반환. 시뮬 엔진이 부르는 진입점."""
        return self.lookup(origin, dest, hour, is_weekend)[0]

    def lookup_many(self, origin, dest, hour, is_weekend) -> pd.DataFrame:
        """벡터 조회. 반환: minutes, tier 컬럼 DataFrame(입력 순서 유지).

        커버율 검증처럼 수십만 건을 한꺼번에 볼 때 쓴다. 스칼라 lookup 과
        같은 폴백 사슬을 merge 로 옮긴 것이라 결과는 동일하다.
        """
        o = pd.Series(origin, dtype="object").map(canon_dong).to_numpy()
        n = len(o)
        df = pd.DataFrame({
            "origin": o,
            "dest": pd.Series(dest, dtype="object").map(canon_dong).to_numpy(),
            "period": np.broadcast_to(period_of_array(hour), (n,)),
            "is_weekend": np.broadcast_to(np.asarray(is_weekend, dtype=bool), (n,)),
        })
        minutes = np.full(n, np.nan)
        tier = np.empty(n, dtype=object)

        # 1순위
        hit = df.merge(self.od, on=["origin", "dest", "period", "is_weekend"],
                       how="left")["minutes"].to_numpy()
        got = ~np.isnan(hit)
        minutes[got], tier[got] = hit[got], "od"

        # 프로필(시간대·요일) — 나머지 순위가 공통으로 쓴다
        prof = df.merge(self.profile, on=["period", "is_weekend"], how="left")
        kmh_p = prof["kmh"].fillna(FALLBACK_KMH).to_numpy()
        intra = prof["intra_min"].fillna(MIN_INTRA_MINUTES).to_numpy()
        med = prof["minutes"].fillna(FALLBACK_MINUTES).to_numpy()

        # 동 내부
        same = ~got & (df["origin"].to_numpy() == df["dest"].to_numpy())
        minutes[same], tier[same] = intra[same], "intra"

        # 좌표 — 표준형으로 먼저, 빈 곳만 기본명으로 다시(coord_of 와 같은 순서)
        coord = self.centroids.set_index("dong")[["lat", "lon"]]

        def _xy(col):
            names = df[col]
            lat = names.map(coord["lat"])
            lon = names.map(coord["lon"])
            miss = lat.isna()
            if miss.any():
                fb = names[miss].map(base_dong)
                lat[miss] = fb.map(coord["lat"])
                lon[miss] = fb.map(coord["lon"])
            return lat.to_numpy(dtype=float), lon.to_numpy(dtype=float)

        olat, olon = _xy("origin")
        dlat, dlon = _xy("dest")

        rest = ~got & ~same
        no_coord = rest & (np.isnan(olat) | np.isnan(dlat))
        minutes[no_coord], tier[no_coord] = med[no_coord], "default"

        rest = rest & ~no_coord
        km = np.zeros(n)
        km[rest] = haversine_km(olat[rest], olon[rest], dlat[rest], dlon[rest]) * DETOUR

        zero = rest & (km <= 0)
        minutes[zero], tier[zero] = intra[zero], "intra"
        rest = rest & ~zero

        # 2순위 — 거리대 속도
        sp = pd.DataFrame({"band": band_of_array(km), "period": df["period"],
                           "is_weekend": df["is_weekend"]})
        kmh = sp.merge(self.speed, on=["band", "period", "is_weekend"],
                       how="left")["kmh"].to_numpy()
        use_speed = rest & ~np.isnan(kmh)
        minutes[use_speed] = km[use_speed] / kmh[use_speed] * 60
        tier[use_speed] = "speed"

        # 3순위 — 시간대 평균속도
        use_line = rest & np.isnan(kmh)
        minutes[use_line] = km[use_line] / kmh_p[use_line] * 60
        tier[use_line] = "line"

        return pd.DataFrame({"minutes": np.clip(minutes, MIN_MINUTES, MAX_MINUTES),
                             "tier": tier})

    # ── 생성·캐시 ─────────────────────────────────────────

    @classmethod
    def build(cls, *, use_cache: bool = True, calls_path: Path = None,
              centroid_path: Path = None) -> "TravelTime":
        """테이블 3종을 캐시에서 읽거나(없으면) 원본에서 만들어 저장한다."""
        src = Path(calls_path) if calls_path else DATA_DIR / CALLS_FILE
        paths = _cache_paths(src)
        centroids = load_centroids(centroid_path)

        if use_cache and all(p.exists() for p in paths.values()):
            tables = {k: pd.read_parquet(p) for k, p in paths.items()}
            return cls(tables["od"], tables["speed"], tables["profile"], centroids)

        rides = load_rides(src)
        tables = {"od": build_od_table(rides), "speed": build_speed_table(rides),
                  "profile": build_profile_table(rides)}
        if use_cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            for k, p in paths.items():
                tables[k].to_parquet(p, index=False)

        tt = cls(tables["od"], tables["speed"], tables["profile"], centroids)
        tt.rides = rides          # 검증용 — 캐시로 읽었을 땐 없다
        return tt


def _cache_paths(src: Path) -> dict:
    """원본 mtime·크기와 산출 파라미터를 파일명에 박는다.

    원본이 바뀌거나 위 상수(신뢰 건수·시간대·거리대)를 손보면 캐시가 저절로
    빗나가 다시 만든다 — 상수만 바꾸고 옛 테이블을 계속 쓰는 사고를 막는 장치다.
    """
    st = src.stat()
    knobs = repr((MIN_TRIPS, MIN_SPEED_TRIPS, PERIOD_CUTS, DIST_CUTS, RIDE_MIN_RANGE,
                  SPEED_RANGE)).encode()
    tag = f"{int(st.st_mtime)}_{st.st_size}_{zlib.crc32(knobs):08x}"
    return {k: CACHE_DIR / f"travel_{k}_{tag}.parquet" for k in ("od", "speed", "profile")}


def _clip(minutes: float) -> float:
    return float(min(max(minutes, MIN_MINUTES), MAX_MINUTES))


# ─────────────────────────────────────────────────────────────
# 모듈 진입점
# ─────────────────────────────────────────────────────────────

_DEFAULT: TravelTime = None


def default_table() -> TravelTime:
    """모듈 공용 조회기(최초 1회만 생성)."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = TravelTime.build()
    return _DEFAULT


def get_travel_time(출발동: str, 목적동: str, hour: int, is_weekend: bool = False) -> float:
    """동A → 동B 이동시간(분). 명세의 공개 함수.

    hour 0~23, is_weekend 는 토·일. 내부에서 시간대 5구간으로 매핑한다.
    반환값은 항상 1~120분 사이 양수다.
    """
    return default_table().minutes(출발동, 목적동, hour, is_weekend)


# ─────────────────────────────────────────────────────────────
# 검증 출력
# ─────────────────────────────────────────────────────────────

# 상식 점검용 구간 — 실측·인접·중거리·횡단·동 내부를 하나씩
SAMPLES = [
    ("하계1동", "상계67동", 12, False, "최다 실측 구간(3,390건) — 1순위"),
    ("역삼1동", "논현1동", 8, False, "강남 인접"),
    ("이태원제1동", "한남동", 14, False, "용산 인접 — '제1동' 표기로 조회"),
    ("서교동", "여의동", 17, False, "마포→영등포 한강 건너"),
    ("방화1동", "길동", 10, False, "강서 서단→강동 동단(횡단)"),
    ("신림동", "신림동", 2, True, "동 내부(심야·주말)"),
]

# 명세 참고 수치(실측) — 산출값이 여기서 크게 벗어나면 필터를 의심할 것
REF_KMH = {"late_night": 20.0, "morning_peak": 14.7, "daytime": 16.0,
           "afternoon_peak": 13.5, "evening": 18.0}


def _report():
    import time
    t0 = time.perf_counter()
    tt = TravelTime.build()
    cached = not hasattr(tt, "rides")
    print(f"테이블 준비 {time.perf_counter() - t0:.1f}초 "
          f"({'캐시 로드' if cached else '원본 산출 후 캐시 저장'})")

    rides = getattr(tt, "rides", None)
    if rides is None:
        rides = load_rides()
    print(f"실측 운행     {len(rides):>10,}건  {rides.attrs.get('funnel', {})}")
    print(f"  중앙        {rides['minutes'].median():.1f}분 / "
          f"{rides['km'].median():.2f}km / {rides['kmh'].median():.1f}km/h"
          f"   (명세 참고: 20분 / 5.5km / 16km/h)")
    print(f"OD 셀        {len(tt.od):>10,}개 (10건 이상) / 전체 {tt.od.attrs.get('n_cells_all', 0):,}")
    print(f"속도 셀      {len(tt.speed):>10,}개 / 프로필 {len(tt.profile)}개")
    amb = tt.centroids.attrs.get("n_ambiguous", 0)
    kinds = tt.centroids["key_kind"].value_counts().to_dict()
    print(f"중심점       {len(tt.centroids):>10,}개 키 {kinds} (구 중복 동명 {amb}개는 좌표 평균)")

    print("\n[시간대 평균속도 km/h] 평일 / 주말   (명세 참고값)")
    prof = tt.profile.set_index(["period", "is_weekend"])
    for p in PERIODS:
        wd = prof.loc[(p, False), "kmh"]
        we = prof.loc[(p, True), "kmh"]
        print(f"  {PERIOD_KO[p]:<16} {wd:5.1f} / {we:5.1f}   ({REF_KMH[p]})")

    # 거리대를 쪼개면 셀이 얇아질 수 있다 — 10개 프로필 × 거리대가 다 찼는지 본다
    print(f"\n[속도 테이블 셀] 거리대 {len(DIST_BANDS)} × 시간대 5 × 요일 2 "
          f"= {len(DIST_BANDS) * 10}칸 중 {len(tt.speed)}칸 채움 "
          f"(기준 {MIN_SPEED_TRIPS}건 이상)")
    cells = (tt.speed.assign(band=pd.Categorical(tt.speed["band"], DIST_BANDS))
                     .groupby("band", observed=False)
                     .agg(칸=("n", "size"), 최소건수=("n", "min"),
                          중앙속도=("kmh", "median")))
    for b, r in cells.iterrows():
        flag = "" if r["칸"] == 10 else "  ⚠ 빈 칸 있음(3순위로 흘러감)"
        print(f"  {str(b):<8} {int(r['칸']):>2}/10칸  최소 {int(r['최소건수']):>6,}건  "
              f"중앙 {r['중앙속도']:4.1f}km/h{flag}")
    thin = tt.speed[tt.speed["n"] < MIN_SPEED_TRIPS]
    print(f"  → 모든 셀 {MIN_SPEED_TRIPS}건 이상: "
          f"{'예' if thin.empty and len(tt.speed) == len(DIST_BANDS) * 10 else '아니오'}")

    print("\n[동 내부 이동 분] 평일 / 주말")
    print("  " + "  ".join(
        f"{PERIOD_KO[p].split('(')[0]} {prof.loc[(p, False), 'intra_min']:.0f}/"
        f"{prof.loc[(p, True), 'intra_min']:.0f}" for p in PERIODS))

    # ── 커버율 ──
    print("\n[커버율] 실제 콜이 어느 순위로 처리되는지 (표 산출에 쓴 데이터 기준·in-sample)")
    res = tt.lookup_many(rides["origin"], rides["dest"], rides["hour"], rides["is_weekend"])
    chk = rides.assign(pred=res["minutes"].to_numpy(), tier=res["tier"].to_numpy())
    chk["err"] = chk["pred"] - chk["minutes"]

    share = res["tier"].value_counts(normalize=True)
    for tier in TIERS:
        pct, g = share.get(tier, 0.0), chk[chk["tier"] == tier]
        err = f"절대중앙오차 {g['err'].abs().median():5.1f}분" if len(g) else "—"
        print(f"  {TIER_KO[tier]:<14} {pct:6.1%}  ({len(g):>9,}건)  {err}")
    print(f"  실측 대비 오차  중앙 {chk['err'].median():+.1f}분 / "
          f"절대중앙 {chk['err'].abs().median():.1f}분")

    # 2순위는 거리대 안에서 속도를 하나로 뭉개므로 거리대 양끝이 치우친다.
    # 특히 10km+ 는 10~15km 와 25km+ 가 한 칸에 들어가 장거리를 과대추정한다.
    band = chk[chk["tier"] == "speed"].assign(band=band_of_array(chk.loc[chk["tier"] == "speed", "km"]))
    print("  2순위 거리대별 오차(분)  " + "  ".join(
        f"{b} {g['err'].median():+.1f}(n={len(g):,})"
        for b, g in band.groupby("band", observed=True)))
    far = band[band["km"] >= 25]
    if len(far):
        print(f"    └ 그중 25km 이상 {len(far):,}건은 중앙 {far['err'].median():+.1f}분 "
              f"— 실측 {far['minutes'].median():.0f}분 vs 추정 {far['pred'].median():.0f}분")

    # ── 우회계수 검산 ──
    # DETOUR 는 직선거리를 도로 주행거리로 올리는 계수다. 실측 승차거리(도로)를
    # 중심점 직선거리로 나눠 되짚는다 — 표본이 두꺼운 OD쌍만 본다.
    pair = (rides.groupby(["origin", "dest"], observed=True)
                 .agg(km=("km", "median"), n=("km", "size"))
                 .reset_index())
    pair = pair[(pair["n"] >= MIN_SPEED_TRIPS) & (pair["origin"] != pair["dest"])]
    coord = tt.centroids.set_index("dong")[["lat", "lon"]]
    olat, olon = pair["origin"].map(coord["lat"]), pair["origin"].map(coord["lon"])
    dlat, dlon = pair["dest"].map(coord["lat"]), pair["dest"].map(coord["lon"])
    line = haversine_km(olat.to_numpy(float), olon.to_numpy(float),
                        dlat.to_numpy(float), dlon.to_numpy(float))
    ok = np.isfinite(line) & (line > 0.1)          # 좌표 없거나 중심점이 겹치는 쌍은 뺀다
    ratio = pair["km"].to_numpy()[ok] / line[ok]
    print(f"\n[우회계수 검산] {MIN_SPEED_TRIPS}건 이상 OD쌍 {len(ratio):,}개  "
          f"중앙 승차거리 ÷ 직선거리 = 중앙 {np.median(ratio):.3f} / 평균 {ratio.mean():.3f}"
          f"   (설정값 DETOUR={DETOUR})")

    # ── 샘플 조회 ──
    print("\n[샘플 조회]")
    for o, d, h, wk, note in SAMPLES:
        minutes, tier = tt.lookup(o, d, h, wk)
        co, cd = tt.coord_of(canon_dong(o)), tt.coord_of(canon_dong(d))
        km = haversine_km(co[0], co[1], cd[0], cd[1]) if co and cd else 0.0
        day = "주말" if wk else "평일"
        print(f"  {o:>7} → {d:<7} {h:>2}시 {day}  {minutes:5.1f}분  "
              f"[{TIER_KO[tier]}] 직선 {km:4.1f}km  — {note}")

    # ── 이상값 ──
    print("\n[이상값 체크]")
    m = res["minutes"].to_numpy()
    bad = {"NaN": int(np.isnan(m).sum()), "0 이하": int((m <= 0).sum()),
           "120분 초과": int((m > MAX_MINUTES).sum())}
    print(f"  전체 조회 {len(m):,}건  최소 {m.min():.1f}분 / 중앙 {np.median(m):.1f}분 / "
          f"최대 {m.max():.1f}분")
    print(f"  {bad}  →  {'정상' if sum(bad.values()) == 0 else '⚠ 확인 필요'}")

    # 모든 동쌍 전수 조회 — 실측에 없는 조합까지 폴백이 값을 내는지
    dongs = tt.centroids.loc[tt.centroids["key_kind"] == "canon", "dong"].to_numpy()
    grid_o = np.repeat(dongs, len(dongs))
    grid_d = np.tile(dongs, len(dongs))
    for h, wk in ((8, False), (2, True)):
        g = tt.lookup_many(grid_o, grid_d, np.full(len(grid_o), h), np.full(len(grid_o), wk))
        gm = g["minutes"].to_numpy()
        print(f"  전 동쌍 {len(gm):,}건 ({h}시 {'주말' if wk else '평일'})  "
              f"결측 {int(np.isnan(gm).sum())} / 범위 {gm.min():.1f}~{gm.max():.1f}분  "
              f"{g['tier'].value_counts(normalize=True).round(3).to_dict()}")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    _report()
