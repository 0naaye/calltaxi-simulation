"""
load.py — 데이터 로딩 및 전처리

원천 CSV/GeoJSON을 읽어 분석·시뮬레이션에 사용할 공통 형태로 변환한다.

데이터 경로
기본: ../data/
환경변수 `DATA_DIR`로 변경 가능
필요한 파일 목록은 README 참고

대용량 콜 데이터는 전처리 결과를 `cache/`에 parquet으로 저장하며,
원본이나 전처리 조건이 바뀌면 캐시를 다시 생성한다.

반환 컬럼명은 모듈 간 일관성을 위해 snake_case 영문으로 통일한다.
원천 컬럼과의 대응은 각 함수 docstring을 따른다.
"""

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

# 저장소 상위의 data/ — clone 위치가 어디든 이 파일 기준으로 잡힌다.
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

# 저장소에 함께 두는 **소형 공개 자료**. `data/` 가 없는 팀원(대시보드만 보는
# 경우)을 위한 폴백이다 — clone 만으로 화면이 뜨게 하려는 것.
#
# **콜 원본은 여기 두지 않는다(개인정보).** 여기 들어가는 것은 차고지 좌표·후보
# 풀처럼 공개된 소형 파일과, 진단 지도용으로 미리 단순화한 경계뿐이다.
# 재실행(evaluate·metrics)은 여전히 `data/` 의 콜 원본을 필요로 한다.
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"

# 진단 지도 **표시 전용** 경계. 0.0002도(약 22m)로 단순화돼 있어 면적·거리·
# 점-폴리곤 판정에 쓰면 안 된다 — `candidates.py` 는 면적(geometry.area)과
# sjoin, EPSG:5179 격자에 경계를 쓰므로 **반드시 원본을 봐야 한다.**
# 그래서 이 파일은 `load_dong(with_boundary=True)` 가 아니라 대시보드의
# `load_boundary()` 만 읽는다. 다시 만들려면 `build_display_boundary()`.
DISPLAY_BOUNDARY = "seoul_dong_display_only.geojson"


def data_or_asset(name: str) -> Path:
    """`data/` 에 있으면 그것, 없으면 저장소 사본(`assets/`).

    **원본이 늘 이긴다** — 데이터가 갱신됐는데 저장소 사본이 낡아 조용히
    옛 값을 읽는 일을 막는다. 사본은 어디까지나 `data/` 가 없을 때의 폴백이다.
    """
    p = DATA_DIR / name
    return p if p.exists() else ASSETS_DIR / name

# 콜 원본 — 특장차 한정본(1,390,969행). 원본 명세에서 흡수한 사실은
# docs/calibration.md 첫 절("원본 명세에서 흡수한 사실").
#
# 구 `calltaxi_2025_merged.csv`(172.9만 행)를 대체한다. 저쪽은 특장차 + 임차택시가
# 섞여 있었는데, 임차택시는 차고지 기반 교대 운영이 아니라 거점 배치의 영향을
# 받지 않아 시뮬 모집단에서 뺀다(A-16). 파일명 상수를 한 곳에 두어
# load / travel_time / idle 이 같은 원본을 보게 한다.
CALLS_FILE = "calls_2025_replay.csv"


def resolve_data_dir() -> Path:
    """데이터 폴더 결정. 환경변수 DATA_DIR 이 있으면 그쪽, 없으면 ../data/.

    팀원마다 clone 구조가 달라도 경로만 지정하면 돌아가게 하려는 것이다.
    import 시점에 한 번 읽어 DATA_DIR 에 담는다 — 실행 중에 환경변수를 바꿨다면
    이 함수를 다시 불러 받아야 한다.
    """
    env = os.environ.get("DATA_DIR", "").strip()
    return Path(env).expanduser().resolve() if env else DEFAULT_DATA_DIR


DATA_DIR = resolve_data_dir()

# 캐시는 산출물이라 데이터 폴더가 아니라 저장소 안에 둔다(읽기전용 데이터 폴더 대응).
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"

# 서울 25개 자치구. 원본에 경기·인천 건이 섞여 있어 화이트리스트로 거른다.
SEOUL_GU = (
    "종로구", "중구", "용산구", "성동구", "광진구", "동대문구", "중랑구", "성북구",
    "강북구", "도봉구", "노원구", "은평구", "서대문구", "마포구", "양천구", "강서구",
    "구로구", "금천구", "영등포구", "동작구", "관악구", "서초구", "강남구", "송파구",
    "강동구",
)

# 즉시콜 판정: |예정 − 접수| 이하면 즉시콜
IMMEDIATE_TOLERANCE_MIN = 2

# 대기시간 상한(분). 초과분은 기록 오류로 보고 제외 — 실측 p99가 127분이라 여유를 뒀다.
MAX_WAIT_MIN = 360

# 2023 경계 기준으로 통폐합된 동.
# step3_match.py(저장소 밖 탐색 스크립트, docs/external_sources.md 5절)와 동일한 판정.
MANUAL_DONG = {
    ("종로구", "명륜3가동"): "혜화동",
    ("동대문구", "용두동"): "용신동",   # 용신동 = 용두동 + 신설동
    ("동대문구", "신설동"): "용신동",
}

# 시간대 구분 — travel_time 의 OD 테이블 키와 공유한다.
PERIOD_BINS = [0, 6, 10, 17, 22, 24]
PERIOD_LABELS = ["심야", "아침", "낮", "저녁", "심야"]


# ─────────────────────────────────────────────────────────────
# 동 이름 정규화
# ─────────────────────────────────────────────────────────────

def canon_dong(name: str) -> str:
    """동 이름 표준형. '고덕제1동' → '고덕1동'.

    콜 원본은 '제N동', 경계 GeoJSON 은 'N동' 표기를 써서 그냥 두면
    441개 중 219개가 어긋난다. 공백·가운뎃점·마침표를 지우고
    숫자 앞의 '제'를 반복 제거한다(성수2가제3동 같은 중첩 대응).
    """
    s = str(name).replace(" ", "").replace("·", "").replace(".", "")
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"제(?=\d)", "", s)
    return s


def _base_dong(canon: str) -> str:
    """숫자와 꼬리 '동'을 뗀 기본명. '화곡1동' → '화곡'. 분동/합동 대응용."""
    b = re.sub(r"\d", "", canon)
    return b[:-1] if b.endswith("동") else b


def build_dong_lookup(path: Path = None) -> pd.DataFrame:
    """(구, 동) → 중심좌표 조회표.

    3단계로 맞춘다:
      exact   표준형이 그대로 일치 (콜 원본 441개 조합 중 413)
      manual  행정구역 개편으로 이름이 바뀐 동 (3)
      approx  분동/합동이라 1:N 이면 같은 기본명 그룹의 중심점 평균 (16)
    맞지 않는 나머지는 표에 넣지 않는다(인천 중구 동이 서울 중구로 섞여 들어온 건 등).

    exact/manual 은 표준형(key_kind='canon')으로, approx 는 숫자를 뗀
    기본명(key_kind='base')으로 색인한다. 조회는 canon 먼저, 실패 시 base 순.

    반환 컬럼: gu, key, key_kind, adm_nm, lat, lon, match_type
    """
    path = Path(path) if path else DATA_DIR / "행정동_중심점.csv"
    cen = pd.read_csv(path, encoding="utf-8-sig")

    parts = cen["adm_nm"].str.split(" ", expand=True)
    cen["gu"] = parts[1]
    # '서울특별시 종로구 사직동' → '사직동'. 동명에 공백이 있으면 붙여준다.
    cen["key"] = parts.iloc[:, 2:].fillna("").agg("".join, axis=1).map(canon_dong)

    exact = cen[["gu", "key", "adm_nm", "lat", "lon"]].copy()
    exact["key_kind"] = "canon"
    exact["match_type"] = "exact"

    manual = []
    for (gu, src), target in MANUAL_DONG.items():
        hit = exact[(exact["gu"] == gu) & (exact["key"] == canon_dong(target))]
        if not hit.empty:
            r = hit.iloc[0]
            manual.append({"gu": gu, "key": canon_dong(src), "adm_nm": r["adm_nm"],
                           "lat": r["lat"], "lon": r["lon"],
                           "key_kind": "canon", "match_type": "manual"})
    manual = pd.DataFrame(manual, columns=exact.columns)

    # 기본명 그룹 평균 — 원본의 '상일동'이 경계상 '상일1동+상일2동'으로 갈린 경우
    grp = (cen.assign(base=cen["key"].map(_base_dong))
              .groupby(["gu", "base"])
              .agg(adm_nm=("adm_nm", lambda s: "+".join(x.split()[-1] for x in s)),
                   lat=("lat", "mean"), lon=("lon", "mean"))
              .reset_index()
              .rename(columns={"base": "key"}))
    grp["key_kind"] = "base"
    grp["match_type"] = "approx"

    cols = ["gu", "key", "key_kind", "adm_nm", "lat", "lon", "match_type"]
    out = pd.concat([exact[cols], manual[cols], grp[cols]], ignore_index=True)
    return out.drop_duplicates(subset=["gu", "key", "key_kind"], keep="first")


def time_period(hour) -> pd.Categorical:
    """시각(0~23) → 아침/낮/저녁/심야. travel_time 의 시간대 키와 동일 기준."""
    return pd.cut(pd.Series(hour), bins=PERIOD_BINS, labels=PERIOD_LABELS,
                  right=False, ordered=False)


# ─────────────────────────────────────────────────────────────
# 콜 원본
# ─────────────────────────────────────────────────────────────

_TIME_COLS = {
    "접수일시": "received_at", "예정일시": "scheduled_at", "배차일시": "assigned_at",
    "승차일시": "boarded_at", "하차일시": "alighted_at", "취소일시": "canceled_at",
}
_PLAIN_COLS = {
    "출발구": "origin_gu", "출발동": "origin_dong", "목적구": "dest_gu", "목적동": "dest_dong",
    "이용목적": "purpose", "요금": "fare", "승차거리": "ride_distance_m",
    "차량구분": "vehicle_type", "장애유형": "disability_type",
}


def load_calls(path: Path = None, *, require_seoul_dest: bool = False,
               use_cache: bool = True, chunksize: int = 500_000) -> pd.DataFrame:
    """콜 탑승내역 로딩.

    - 즉시콜 필터(|예정−접수| <= 2분)
    - 서울 25구 필터(출발 기준. require_seoul_dest=True 면 목적지도 서울로 한정)
    - 대기시간 = 승차 − 접수 계산
    - 출발동·목적동에 중심좌표를 붙인다(좌표를 못 찾는 출발동은 제외)

    원본은 특장차 한정본 1,390,969행(A-16). 즉시콜 88.0%, 서울 출발이 그 대부분이라
    필터를 통과하는 건 1,222,330건이다.
    결과는 cache/calls_*.parquet 에 저장하고 원본 mtime·크기가 같으면 재사용한다.

    **`is_canceled` 는 승차 기록의 유무다** — 정산 미기록 616건(승차·취소 없이 하차만
    있는 건 — calibration.md 흡수 사실 ⑵)이 여기 함께 잡힌다. 운행은 이루어졌지만 승차 시각이 없어
    대기시간을 산출할 수 없고, 취소가 아니므로 별도 플래그 `is_unsettled` 로 갈라둔다.
    취소 4구간(metrics.cancel_kind)은 이 건들을 'unsettled' 로 따로 뺀다.

    반환 컬럼(원천 한글 → 영문):
      접수/예정/배차/승차/하차/취소일시 → received_at ... canceled_at
      출발구·출발동 → origin_gu, origin_dong  (+ _canon, _lat, _lon)
      목적구·목적동 → dest_gu, dest_dong      (+ _canon, _lat, _lon)
      요금·승차거리·차량구분·장애유형·이용목적 → fare, ride_distance_m, ...
      파생: wait_min(승차−접수), assign_min(배차−접수), ride_min(하차−승차),
            is_canceled, is_unsettled, date, hour, weekday, period
    """
    path = Path(path) if path else DATA_DIR / CALLS_FILE
    cache = _cache_path(path, require_seoul_dest)

    if use_cache and cache.exists():
        return pd.read_parquet(cache)

    lookup = build_dong_lookup()
    usecols = list(_TIME_COLS) + list(_PLAIN_COLS)
    kept = []
    n_raw = 0

    reader = pd.read_csv(path, encoding="utf-8-sig", usecols=usecols,
                         chunksize=chunksize, low_memory=False)
    for chunk in reader:
        n_raw += len(chunk)
        kept.append(_prepare_chunk(chunk, lookup, require_seoul_dest))

    calls = pd.concat(kept, ignore_index=True)
    calls.insert(0, "call_id", np.arange(len(calls), dtype=np.int32))
    calls = _shrink(calls)
    calls.attrs["n_raw"] = n_raw

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        calls.to_parquet(cache, index=False)
    return calls


def _prepare_chunk(chunk: pd.DataFrame, lookup: pd.DataFrame,
                   require_seoul_dest: bool) -> pd.DataFrame:
    """청크 하나에 필터·파생을 적용. 메모리 때문에 나눠 처리한다."""
    df = chunk.rename(columns={**_TIME_COLS, **_PLAIN_COLS})
    for col in _TIME_COLS.values():
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # 즉시콜
    gap = (df["scheduled_at"] - df["received_at"]).abs().dt.total_seconds() / 60
    df = df[gap <= IMMEDIATE_TOLERANCE_MIN]

    # 서울
    df = df[df["origin_gu"].isin(SEOUL_GU)]
    if require_seoul_dest:
        df = df[df["dest_gu"].isin(SEOUL_GU)]
    if df.empty:
        return df

    df["origin_dong_canon"] = df["origin_dong"].map(canon_dong)
    df["dest_dong_canon"] = df["dest_dong"].map(canon_dong)

    # 좌표 부착
    df = _attach_coords(df, lookup, "origin")
    df = _attach_coords(df, lookup, "dest")
    # 출발 좌표는 배차 계산에 필수 — 못 찾으면 버린다(인천 중구 등 오적재 건)
    df = df[df["origin_lat"].notna()]

    # 파생
    df["wait_min"] = (df["boarded_at"] - df["received_at"]).dt.total_seconds() / 60
    df["assign_min"] = (df["assigned_at"] - df["received_at"]).dt.total_seconds() / 60
    df["ride_min"] = (df["alighted_at"] - df["boarded_at"]).dt.total_seconds() / 60

    # 정산 미기록(calibration.md 흡수 사실 ⑵) — 승차·취소가 없는데 하차만 있는 건. 배차→하차 중앙
    # 56.6분으로 운행은 이루어졌다. 승차 시각이 없어 대기시간을 낼 수 없지만
    # 취소도 아니다. 섞어두면 취소율이 0.04%p 부풀고 '배차 후 취소'에 얹힌다.
    df["is_unsettled"] = (df["boarded_at"].isna() & df["canceled_at"].isna()
                          & df["alighted_at"].notna())
    df["is_canceled"] = df["boarded_at"].isna() & ~df["is_unsettled"]

    # 음수·비현실 대기는 기록 오류로 보고 결측 처리(행은 남긴다 — 취소율 분모 유지)
    bad = (df["wait_min"] < 0) | (df["wait_min"] > MAX_WAIT_MIN)
    df.loc[bad, "wait_min"] = np.nan

    df["date"] = df["received_at"].dt.normalize()
    df["hour"] = df["received_at"].dt.hour.astype("int8")
    df["weekday"] = df["received_at"].dt.weekday.astype("int8")
    df["period"] = time_period(df["hour"].to_numpy()).astype(str)

    df["dest_in_seoul"] = df["dest_gu"].isin(SEOUL_GU)
    return df.reset_index(drop=True)


def _attach_coords(df: pd.DataFrame, lookup: pd.DataFrame, side: str) -> pd.DataFrame:
    """출발/목적 동에 중심좌표를 붙인다. side 는 'origin' 또는 'dest'.

    표준형으로 먼저 맞추고, 남은 건만 기본명으로 다시 맞춘다.
    (분동/합동 동은 표준형이 경계 파일에 없어서 이 2단계가 필요하다)
    """
    gu, canon = f"{side}_gu", f"{side}_dong_canon"
    lat, lon, match = f"{side}_lat", f"{side}_lon", f"{side}_match"

    def _right(kind, key_col):
        return (lookup[lookup["key_kind"] == kind]
                .rename(columns={"gu": gu, "key": key_col,
                                 "lat": lat, "lon": lon, "match_type": match})
                [[gu, key_col, lat, lon, match]])

    df = df.merge(_right("canon", canon), on=[gu, canon], how="left")

    unresolved = df[lat].isna()
    if unresolved.any():
        base_col = f"{side}_dong_base"
        df[base_col] = df[canon].map(_base_dong)
        fb = df.loc[unresolved, [gu, base_col]].merge(
            _right("base", base_col), on=[gu, base_col], how="left")
        for col in (lat, lon, match):
            df.loc[unresolved, col] = fb[col].to_numpy()
        df = df.drop(columns=base_col)

    return df


# 저카디널리티 문자열 — object 로 두면 150만 행에서 1.4GB 를 먹는다
_CATEGORICAL = (
    "origin_gu", "origin_dong", "origin_dong_canon", "origin_match",
    "dest_gu", "dest_dong", "dest_dong_canon", "dest_match",
    "purpose", "vehicle_type", "disability_type", "period",
)
_FLOAT32 = ("fare", "ride_distance_m", "wait_min", "assign_min", "ride_min",
            "origin_lat", "origin_lon", "dest_lat", "dest_lon")


def _shrink(df: pd.DataFrame) -> pd.DataFrame:
    """dtype 축소. 청크별로 하면 카테고리가 어긋나므로 합친 뒤 한 번에 한다."""
    for col in _CATEGORICAL:
        df[col] = df[col].astype("category")
    for col in _FLOAT32:
        df[col] = df[col].astype("float32")
    return df


# 캐시 스키마 판. 파생 컬럼이나 필터 정의를 손대면 올린다 — 원본이 그대로여도
# 옛 parquet 을 계속 읽어 컬럼이 비는 사고를 막는다(is_unsettled 추가 때 겪었다).
CACHE_SCHEMA = 2


def _cache_path(src: Path, require_seoul_dest: bool) -> Path:
    """원본 mtime·크기 + 스키마 판을 파일명에 박는다.

    원본이 바뀌거나 파생 컬럼 구성이 바뀌면 캐시가 자동으로 빗나간다.
    """
    st = src.stat()
    tag = "od" if require_seoul_dest else "o"
    return (CACHE_DIR /
            f"calls_{tag}_v{CACHE_SCHEMA}_{int(st.st_mtime)}_{st.st_size}.parquet")


# ─────────────────────────────────────────────────────────────
# 거점·경계·후보지
# ─────────────────────────────────────────────────────────────

def load_depots(path: Path = None) -> pd.DataFrame:
    """현행 차고지 44개 좌표·**배속 정원** 로딩.

    반환 컬럼: depot_id, name, gu, address, lat, lon, capacity, operator, is_paid
    (차량대수 합 691대)

    **⚠ `capacity` 는 면수가 아니라 배속 정원이다.** 공단 공개 파일
    `서울시설공단_장애인콜택시 차고지 정보_20250724.csv` 의 `주차대수`(합 **699**)는
    **면수**이고, 여기서 읽는 `차고지44_좌표.csv` 의 `차량대수`(합 **691**)는
    **배속 정원**이다. **둘은 다른 양이라 합이 갈리는 것이 정상이다.**

    44곳 전부 이름·주소가 일치하지만 **11곳에서 값이 갈린다**(순차 +8). 시차 운행
    (A-09) 때문에 **배속이 면수를 넘을 수 있다** — 종묘지하주차장 배속 46대 vs 면수
    34면. 반대로 공단 값을 배속으로 읽으면 용산차고지가 26대에 근무인원 16명이 되어
    깨진다. 대조 표와 판단 근거는 docs/calibration.md 「차고지 배속 정원」 절.

    **시뮬은 이 값을 배분 비율로만 쓴다**(`simulator.allocate_fleet`) — 그 날 나온
    가동 대수(평일 574 / 주말 374)를 거점별로 나누는 가중치다.
    """
    path = Path(path) if path else data_or_asset("차고지44_좌표.csv")
    df = pd.read_csv(path, encoding="utf-8-sig")
    out = df.rename(columns={
        "연번": "depot_id", "차고지명": "name", "소속구": "gu", "상세주소": "address",
        "차량대수": "capacity", "차고지_관리주체": "operator",
    })[["depot_id", "name", "gu", "address", "lat", "lon", "capacity", "operator"]]
    out["is_paid"] = df["차고지_유무료"].eq("유료")
    out["capacity"] = out["capacity"].astype(int)
    return out


def load_dong(path: Path = None, *, with_boundary: bool = False, crs: int = 4326):
    """행정동 중심점·경계 로딩.

    with_boundary=False (기본) → 중심점 DataFrame (gu, dong_canon, adm_nm, lat, lon)
    with_boundary=True         → 경계 GeoDataFrame (geopandas 필요, 34MB GeoJSON)

    원본 GeoJSON 은 EPSG:4326 이다. 거리 계산을 미터로 하려면 crs=5179 를 넘겨
    투영좌표로 받는다(README 의 EPSG:5179 기준).

    **여기는 언제나 원본을 읽는다 — 표시용 단순화본으로 폴백하지 않는다.**
    `candidates.py` 가 이 경계로 면적(`geometry.area`)·`sjoin`·EPSG:5179 격자를
    계산하는데, 22m 단순화본을 쓰면 면적이 틀리고 경계 근처 점 판정이 뒤집힌다.
    `data/` 가 없으면 **조용히 대체하지 말고 그대로 실패해야 한다.**
    화면용 경계가 필요한 자리는 `load_display_boundary()` 를 쓴다.
    """
    if not with_boundary:
        cen = build_dong_lookup(path)
        cen = cen[cen["match_type"] == "exact"].rename(columns={"key": "dong_canon"})
        return cen[["gu", "dong_canon", "adm_nm", "lat", "lon"]].reset_index(drop=True)

    import geopandas as gpd  # 무거워서 경계가 필요할 때만 부른다

    path = Path(path) if path else DATA_DIR / "HangJeongDong_ver20230701.geojson"
    gdf = gpd.read_file(path, engine="pyogrio", where="adm_nm LIKE '서울특별시%'")
    gdf["gu"] = gdf["adm_nm"].str.split(" ").str[1]
    gdf["dong_canon"] = gdf["adm_nm"].str.split(" ").str[2].map(canon_dong)
    if crs and gdf.crs and gdf.crs.to_epsg() != crs:
        gdf = gdf.to_crs(crs)
    return gdf[["adm_nm", "adm_cd2", "gu", "dong_canon", "geometry"]]


# 표시 전용 경계를 만들 때 쓰는 단순화 허용오차(도). 0.0002 ≈ 22m.
# 대시보드가 런타임에 쓰던 값과 같다 — 그 일을 미리 해 두는 것뿐이다.
DISPLAY_SIMPLIFY_TOL = 0.0002


def build_display_boundary(out: Path = None, tol: float = DISPLAY_SIMPLIFY_TOL):
    """진단 지도용 경계를 만들어 `assets/` 에 저장한다. **원본이 필요하다.**

    34.8MB 원본을 저장소에 넣지 않으려고 둔 장치다 — 서울만·필요 컬럼만 남기고
    22m 로 단순화하면 **0.40MB** 가 된다(87배). 대시보드는 어차피 이 정밀도로
    그리므로 화면에서 달라지는 것이 없다.

    경계 원본이 갱신되면 이 함수를 다시 돌려야 한다. 사본이 낡으면 지도만
    옛 경계로 그려지고 계산은 원본을 보므로 **둘이 갈릴 수 있다.**
    """
    gdf = load_dong(with_boundary=True).copy()          # 원본 — 폴백 없음
    gdf["geometry"] = gdf.geometry.simplify(tol, preserve_topology=True)
    out = Path(out) if out else ASSETS_DIR / DISPLAY_BOUNDARY
    out.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out, driver="GeoJSON")
    return out


def load_display_boundary():
    """진단 지도용 경계. **표시 전용 — 면적·거리 계산에 쓰지 말 것.**

    `data/` 에 원본이 있으면 원본을 읽어 단순화하고(원본이 늘 이긴다), 없으면
    저장소 사본을 그대로 쓴다. 사본은 이미 단순화돼 있어 다시 줄이지 않는다.
    """
    import geopandas as gpd

    src = DATA_DIR / "HangJeongDong_ver20230701.geojson"
    if src.exists():
        gdf = load_dong(with_boundary=True).copy()
        gdf["geometry"] = gdf.geometry.simplify(DISPLAY_SIMPLIFY_TOL,
                                                preserve_topology=True)
        return gdf

    asset = ASSETS_DIR / DISPLAY_BOUNDARY
    if not asset.exists():
        raise FileNotFoundError(
            f"경계가 없다 — {src} 도 {asset} 도 찾지 못했다. "
            f"data/ 를 두거나 load.build_display_boundary() 로 사본을 만들 것")
    return gpd.read_file(asset, engine="pyogrio")


_POOL_COLS = {
    "주차장명": "name", "주소": "address", "자치구": "gu",
    "법정동": "bdong", "법정동코드": "bdong_cd",
    "면수": "capacity", "위도": "lat", "경도": "lon",
    "소스": "source", "공공근거": "public_basis",
    "UPIS대분류": "upis_lclas", "UPIS시설명": "upis_name", "고시번호": "notice_no",
    "옥내외": "enclosure", "옥내자주": "indoor_self", "옥외자주": "outdoor_self",
    "현행차고지": "current_depot", "현행차고지거리m": "current_depot_dist_m",
    "운영시간": "open_hours", "급지": "price_grade", "이용효율": "usage_efficiency",
}

# **실가용 면수(피크시간 잔여구획)는 쓰지 않는다 — 2026.08.10 철회.**
# 정보공개청구 17078071 의 `2025 피크시간 잔여` 시트에서 시영 65곳에 붙이던
# `capacity_available` · `capacity_available_ratio` 열과 `load_peak_residual()`
# 을 걷어냈다. **639곳 중 65곳(10%)에만 있는 값이라** 참고 컬럼으로 남겨 둬도
# 후보를 좁힐 때 쓸 수가 없다 — 그 65곳만 다른 자로 재는 셈이 된다.
# 원본 xlsx 는 그대로 있으므로 되살리려면 git 이력을 볼 것.


def load_candidates(path: Path = None, *, outdoor_only: bool = True) -> pd.DataFrame:
    """시뮬 거점 후보 풀 로딩 — `sim_pool_v4.csv` (639곳 · 55,798면).

    공영·시영 원본 목록을 직접 읽던 방식을 대체한다. v4 풀은 공영·시영에 더해
    KOTSA 주차장을 UPIS 도시계획시설 폴리곤과 공간조인해 공공성을 판정하고,
    건축물대장으로 민간·기계식을 걸러낸 결과다. 만들어진 과정은
    docs/calibration.md 의 흡수 사실 ⑷ 참조.

    **기본은 옥외 348곳 24,392면만 돌려준다(A-15).** 리프트 특장차(전고 2.6m)가
    들어갈 수 있어야 거점이 되는데, 주차장 진입 유효고를 주는 소스가 없어
    옥내 163곳·혼합 128곳은 진입 가능 여부를 판정할 수 없다(파이프라인 8-1,
    미해결 #2). 기계식은 풀 단계에서 이미 빠졌지만 자주식 옥내라도 유효고가
    낮으면 리프트 차량은 못 들어간다. 혼합도 뺀다 — 파이프라인이 혼합 111곳은
    사실상 옥내이고 "옥외분만 골라 쓴다"는 계산은 신뢰할 수 없다고 못박았다.

    **제외분은 버리지 않는다.** `outdoor_only=False` 로 부르면 639곳 전체가
    나온다. 현장에서 유효고를 확인하면 그대로 재편입할 후보다.

    **용량은 `capacity` = 총 면수 하나뿐이다.** 시영 65곳에만 있던 실측 잔여면
    (`capacity_available`)은 08.10 에 걷어냈다 — 639곳 중 10%에만 있는 값이라
    참고로도 쓸 수 없었다. 위 상수 자리의 주석 참조.

    cand_id 는 원본 CSV 의 행 순서(1부터)다. 풀을 다시 만들면 번호가 바뀐다 —
    저장 키가 아니라 실행 안에서의 참조용이다.

    반환 컬럼: cand_id, name, address, gu, bdong, bdong_cd, lat, lon,
              capacity, source, public_basis, upis_lclas, upis_name, notice_no,
              enclosure, is_outdoor, indoor_self, outdoor_self,
              current_depot, current_depot_dist_m,
              open_hours, price_grade, usage_efficiency
    """
    path = Path(path) if path else data_or_asset("sim_pool_v4.csv")
    df = pd.read_csv(path, encoding="utf-8-sig")

    missing = set(_POOL_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} 에 없는 컬럼: {sorted(missing)}")

    out = df.rename(columns=_POOL_COLS)[list(_POOL_COLS.values())].copy()
    out.insert(0, "cand_id", np.arange(1, len(out) + 1, dtype=np.int32))
    out["capacity"] = out["capacity"].round().astype(int)
    out["is_outdoor"] = out["enclosure"].eq("옥외")

    n_all = len(out)
    n_by_enclosure = out["enclosure"].value_counts().to_dict()
    cap_by_enclosure = out.groupby("enclosure")["capacity"].sum().to_dict()

    if not out["gu"].isin(SEOUL_GU).all():
        bad = sorted(set(out["gu"]) - set(SEOUL_GU))
        raise ValueError(f"서울 25구가 아닌 자치구가 섞여 있다: {bad}")

    if outdoor_only:
        out = out[out["is_outdoor"]]
    out = out.reset_index(drop=True)

    out.attrs["n_pool"] = n_all
    out.attrs["n_by_enclosure"] = n_by_enclosure
    out.attrs["capacity_by_enclosure"] = cap_by_enclosure
    out.attrs["n_excluded_indoor_mixed"] = (n_by_enclosure.get("옥내", 0)
                                            + n_by_enclosure.get("혼합", 0))

    cols = ["cand_id", "name", "address", "gu", "bdong", "bdong_cd", "lat", "lon",
            "capacity",
            "source", "public_basis", "upis_lclas", "upis_name", "notice_no",
            "enclosure", "is_outdoor", "indoor_self", "outdoor_self",
            "current_depot", "current_depot_dist_m",
            "open_hours", "price_grade", "usage_efficiency"]
    return out[cols]


def load_disabled_population(path: Path = None) -> pd.DataFrame:
    """동별 등록 장애인 수(장애유형 합계 · 성별 계).

    통계표를 그대로 받은 파일이라 정형이 아니다. 세 가지를 흡수한다:
      - 행 끝에 쉼표가 남아 빈 컬럼이 하나 더 붙고 열 수가 어긋난 행이 섞여 있다
        → on_bad_lines='skip' + engine='python'
      - 구 컬럼이 따로 없고 '동별' 한 컬럼에 총계·구·동·'기타'가 섞여 있다.
        구 행이 그 구의 동들 앞에 오는 순서라 ffill 로 구를 채운다.
      - 값이 '-' 인 칸이 있다(집계 없음) → NaN

    '기타'(구마다 1행)는 동 단위가 없는 잔여분이라 동 지표에 못 쓴다. 총계·구 행과
    함께 뺀다. 남는 427개 동의 합은 384,921명으로 표 전체 합계(384,934)와 13명 차이다.

    반환 컬럼: gu, dong, dong_canon, dong_base, n_disabled
    """
    path = Path(path) if path else DATA_DIR / "서울시_장애인_통계_2025.csv"
    df = pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip", engine="python")

    df = df[(df["장애유형별"] == "합계") & (df["성별"] == "계")].copy()
    # 구 행이 동 행보다 먼저 나오는 순서에 기댄다 — 원본 정렬이 바뀌면 여기가 깨진다.
    df["gu"] = df["동별"].where(df["동별"].isin(SEOUL_GU)).ffill()

    # 연도 컬럼명이 갱신마다 바뀐다('2025 년' → '2026 년'). 마지막 연도 컬럼을 쓴다.
    year_cols = [c for c in df.columns if re.fullmatch(r"\d{4}\s*년", str(c))]
    if not year_cols:
        raise ValueError(f"연도 컬럼을 찾지 못했다: {list(df.columns)}")

    out = df[~df["동별"].isin(SEOUL_GU) & ~df["동별"].isin(("합계", "기타"))].copy()
    out = out.rename(columns={"동별": "dong"})
    out["n_disabled"] = pd.to_numeric(out[year_cols[-1]], errors="coerce")
    out["dong_canon"] = out["dong"].map(canon_dong)
    out["dong_base"] = out["dong_canon"].map(_base_dong)

    cols = ["gu", "dong", "dong_canon", "dong_base", "n_disabled"]
    return out[cols].reset_index(drop=True)


_IDLE_GAP_COLS = ["차량번호", "배차일시", "승차일시", "하차일시"]


def load_idle_gaps(path: Path = None, *, use_cache: bool = True) -> np.ndarray:
    """당일 내 유휴 구간 길이(분) 전수. 시뮬의 휴게 임계 θ 캘리브레이션 입력.

    정의는 `idle.py` 와 같다 — 어떤 운행의 **하차** ~ 같은 차량의 **다음 배차**.
    자정을 넘긴 구간은 근무 종료~다음 근무 시작이 섞여 있어 뺀다(그쪽은 '유휴'가
    아니다). 음수(중복 배차)도 뺀다.

    **이 배열이 θ 의 근거다.** 시뮬은 'θ 분 동안 배차가 없으면 휴게'로 공급을
    줄이는데, 휴게 길이를 자유 파라미터로 두면 θ 가 대기를 맞추는 손잡이가 된다.
    대신 여기서 `gap > θ` 인 구간의 **초과분 분포**를 그대로 뽑아 쓴다 — 파라미터는
    θ 하나뿐이고 휴게 길이는 실측에서 나온다.

    한계: 실측의 긴 구간에는 '쉬어서 안 받은 것'과 '부를 콜이 없어서 못 받은 것'이
    섞여 있다. 둘을 가를 자료가 없어 전부 휴게로 읽으므로 **θ 는 공급 감소를
    과대 추정하는 쪽으로 치우친다.**

    반환: float32 1차원 배열(정렬됨)
    """
    path = Path(path) if path else DATA_DIR / CALLS_FILE
    st = path.stat()
    cache = CACHE_DIR / f"idlegap_v1_{int(st.st_mtime)}_{st.st_size}.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)["idle_min"].to_numpy(dtype="float32")

    df = pd.read_csv(path, encoding="utf-8-sig", usecols=_IDLE_GAP_COLS, low_memory=False)
    assign = pd.to_datetime(df["배차일시"], format="ISO8601", errors="coerce")
    board = pd.to_datetime(df["승차일시"], format="ISO8601", errors="coerce")
    alight = pd.to_datetime(df["하차일시"], format="ISO8601", errors="coerce")
    vehicle = pd.to_numeric(df["차량번호"], errors="coerce")

    ok = board.notna() & alight.notna() & assign.notna() & vehicle.notna()
    t = pd.DataFrame({"vehicle_id": vehicle[ok].astype("int64"),
                      "assigned_at": assign[ok], "alighted_at": alight[ok]})
    t = t.sort_values(["vehicle_id", "assigned_at"], kind="stable")
    nxt = t.groupby("vehicle_id", sort=False)["assigned_at"].shift(-1)

    gap = (nxt - t["alighted_at"]).dt.total_seconds() / 60
    same_day = t["alighted_at"].dt.normalize() == nxt.dt.normalize()
    out = np.sort(gap[(gap >= 0) & same_day].to_numpy(dtype="float32"))

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"idle_min": out}).to_parquet(cache, index=False)
    return out


_RESERVATION_COLS = ["접수일시", "예정일시", "배차일시", "하차일시"]


def load_reservation_occupancy(path: Path = None, *, use_cache: bool = True) -> pd.DataFrame:
    """예약콜이 시간대별로 묶어두는 차량 대수. 가정 A-11 의 차감량.

    **예약콜은 시뮬에서 재생하지 않는다.** 쿼터제(성수기 평일 시간대별 100명, 심야
    3명)로 검열된 수요라 그대로 재생하면 "쿼터 때문에 못 한 콜"이 없는 것처럼 된다.
    대신 그 운행이 실제로 차량을 묶어둔 만큼을 가용 대수에서 뺀다 — 차감하지 않으면
    같은 차량을 쓰는 예약 운행이 시뮬에 안 잡혀 **가용 차량이 과대**해진다.

    차감량을 '건수'가 아니라 **동시 점유 대수**로 낸다. 건수는 운행 길이를 무시해서
    한 시간에 20건이 각각 10분짜리인 경우와 90분짜리인 경우를 같게 만든다. 구간
    [배차, 하차)를 시간 칸에 잘라 넣고 (60분 × 일수)로 나눈 값이라, 그 시각에 평균
    몇 대가 예약콜에 묶여 있었나가 된다(idle.hourly_concurrency 와 같은 산식).

    점유 시작을 **배차일시**로 잡는다. 승차가 아니라 배차부터 그 차량은 이미 그 콜에
    매여 공차 이동 중이다(idle.py 의 유휴 정의와 같은 이유).

    평일 07시 102대 · 08시 113대로 오전에 몰린다 — 574대 기준 20%다. **검증에서
    오전 시간대가 어긋나면 1순위 용의자가 여기다**(A-11).

    반환: hour(0~23) × is_weekend 2열 — 컬럼 `weekday`, `weekend` (대수, float)
    """
    path = Path(path) if path else DATA_DIR / CALLS_FILE
    cache = CACHE_DIR / f"reservation_v1_{int(path.stat().st_mtime)}_{path.stat().st_size}.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)

    df = pd.read_csv(path, encoding="utf-8-sig", usecols=_RESERVATION_COLS, low_memory=False)
    received = pd.to_datetime(df["접수일시"], format="ISO8601", errors="coerce")
    scheduled = pd.to_datetime(df["예정일시"], format="ISO8601", errors="coerce")
    assigned = pd.to_datetime(df["배차일시"], format="ISO8601", errors="coerce")
    alighted = pd.to_datetime(df["하차일시"], format="ISO8601", errors="coerce")

    gap = (scheduled - received).abs().dt.total_seconds() / 60
    ok = ((gap > IMMEDIATE_TOLERANCE_MIN) & assigned.notna() & alighted.notna()
          & (alighted > assigned))
    start, end = assigned[ok], alighted[ok]

    epoch = start.min().normalize()
    s = (start - epoch).dt.total_seconds().to_numpy() / 60
    e = (end - epoch).dt.total_seconds().to_numpy() / 60
    is_we = (start.dt.weekday >= 5).to_numpy()

    def occupied(t, h):
        whole, rem = np.divmod(t, 1440)
        return whole * 60 + np.clip(rem - 60 * h, 0, 60)

    out = pd.DataFrame({"hour": np.arange(24, dtype="int8")})
    for col, mask in (("weekday", ~is_we), ("weekend", is_we)):
        n_days = start[mask].dt.normalize().nunique()
        out[col] = [float((occupied(e[mask], h) - occupied(s[mask], h)).sum())
                    / (60 * n_days) for h in range(24)]
    out.attrs["n_reservations"] = int(ok.sum())

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        out.to_parquet(cache, index=False)
    return out


_VEHICLE_COLS = ["차량번호", "접수일시", "승차일시", "하차일시", "승차거리"]


def load_vehicle_trips(path: Path = None) -> pd.DataFrame:
    """차량별 운행 기록. 조 편성 실측 재구성용(A-09 — metrics.vehicle_day_table).

    travel_time.load_rides 와 같은 원본이지만 목적이 달라 필터가 다르다. 저쪽은
    동A→동B 이동시간 표를 만드는 게 목적이라 출발·목적이 모두 서울인 건만 쓰지만,
    여기서는 서울 밖 운행도 차량이 묶여 있던 시간이라 구를 가리지 않는다.

    필터: 승차·하차 시각이 모두 기록된 완료 건, 운행시간 1~120분, 차량번호 있음.
    차량번호가 비어 있는 건(미탑승 건 — 원본이 완료 운행에만 차량을 붙인다)은
    차량-일 집계의 키가 없어 뺀다. 제외 건수는 attrs['n_no_vehicle'] 에 남긴다 —
    groupby 가 조용히 떨어뜨리게 두지 않으려는 것이다.

    **정산 미기록 616건(calibration.md 흡수 사실 ⑵)은 여기에 들어오지 않는다.** 승차 시각이 없어
    필터에 걸리고, 애초에 차량번호도 없어 어느 차량의 하루였는지 알 수 없다.
    배차→하차 중앙 56.6분만큼 차량이 묶여 있었지만 귀속시킬 차량이 없다.

    차량번호는 723개다(특장차 한정본 · A-16). 차고지 정원 합 691대보다 많은데,
    1년치라 중간에 교체·증차된 차량이 모두 잡히기 때문이다. 특정 시점의 가동
    대수가 아니다 — **일별 가동은 중앙 561대(범위 246~626)로, 시뮬 차량 수는
    723이 아니라 이쪽을 써야 한다(calibration.md 흡수 사실 ⑶).**

    service_date 는 승차 시각 기준이라 자정을 넘긴 운행은 다음 날로 넘어간다.
    심야 비중이 1.7%라 근무 길이 집계에는 영향이 거의 없다.

    **가동 대수를 셀 때는 request_date(접수일 기준)를 쓴다.** 자정 직전 접수 →
    자정 직후 승차인 건이 승차일 기준에서는 다음 날의 '가동'으로 넘어가, 그 차량이
    실제로 근무한 날과 어긋난다. 접수일 기준 중앙 561대 · 승차일 기준 540대로
    20대가 갈린다. calibration.md 흡수 사실 ⑶ 의 561 은 접수일 기준이다.

    반환 컬럼: vehicle_id, service_date, request_date, boarded_at, alighted_at,
              ride_min, km
    """
    path = Path(path) if path else DATA_DIR / CALLS_FILE
    df = pd.read_csv(path, encoding="utf-8-sig", usecols=_VEHICLE_COLS, low_memory=False)
    n_raw = len(df)

    board = pd.to_datetime(df["승차일시"], format="ISO8601", errors="coerce")
    alight = pd.to_datetime(df["하차일시"], format="ISO8601", errors="coerce")
    ride_min = (alight - board).dt.total_seconds() / 60

    ok = board.notna() & alight.notna() & ride_min.between(1.0, 120.0)
    df, board, alight, ride_min = df[ok], board[ok], alight[ok], ride_min[ok]

    vehicle = pd.to_numeric(df["차량번호"], errors="coerce")
    has_vehicle = vehicle.notna()
    n_no_vehicle = int((~has_vehicle).sum())
    df, board, alight, ride_min, vehicle = (
        df[has_vehicle], board[has_vehicle], alight[has_vehicle],
        ride_min[has_vehicle], vehicle[has_vehicle])

    received = pd.to_datetime(df["접수일시"], format="ISO8601", errors="coerce")

    out = pd.DataFrame({
        "vehicle_id": vehicle.astype("int64").to_numpy(),
        "service_date": board.dt.normalize().to_numpy(),
        "request_date": received.dt.normalize().to_numpy(),
        "boarded_at": board.to_numpy(),
        "alighted_at": alight.to_numpy(),
        "ride_min": ride_min.astype("float32").to_numpy(),
        # 승차거리 0 은 결측 대체값이다(calibration.md 흡수 사실 ⑴). 완료 건에도 4,636건 섞여 있고
        # 요금·운행시간은 정상이라 미터기·GPS 미기록으로 보인다. 0 을 그대로 두면
        # 평균이 눌리므로 NaN 으로 돌린다 — 행은 남긴다(운행 자체는 있었다).
        "km": (pd.to_numeric(df["승차거리"], errors="coerce")
                 .replace(0, np.nan) / 1000).astype("float32").to_numpy(),
    })
    out.attrs["n_raw"] = n_raw
    out.attrs["n_no_vehicle"] = n_no_vehicle
    return out.reset_index(drop=True)


def load_undersupplied(path: Path = None, *, max_vehicles_3km: int = 10) -> pd.DataFrame:
    """과소공급 동(3km내 차량 10대 이하) 로딩.

    동별_거점용량_접근성.csv 는 분석 단계(step4_distance.py — 저장소 밖 탐색
    스크립트, docs/external_sources.md 5절)의 산출물로,
    동별 실측 대기와 반경별 차량대수를 이미 붙여둔 표다. 기본 임계 10대에서 68개 동.

    반환 컬럼: gu, dong, dong_canon, mean_wait, long_wait_ratio, n_calls,
              nearest_depot, nearest_km, vehicles_3km, supply_level, lat, lon
    """
    path = Path(path) if path else DATA_DIR / "동별_거점용량_접근성.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")

    out = df.rename(columns={
        "출발구": "gu", "출발동": "dong", "평균대기": "mean_wait",
        "장기대기율": "long_wait_ratio", "건수": "n_calls",
        "최근접차고지명": "nearest_depot", "최근접거리km": "nearest_km",
        "3km내차량대수": "vehicles_3km", "공급수준": "supply_level",
    })
    out["dong_canon"] = out["dong"].map(canon_dong)
    out["long_wait_ratio"] = out["long_wait_ratio"] / 100   # 원본이 퍼센트 표기

    cols = ["gu", "dong", "dong_canon", "mean_wait", "long_wait_ratio", "n_calls",
            "nearest_depot", "nearest_km", "vehicles_3km", "supply_level", "lat", "lon"]
    out = out[out["vehicles_3km"] <= max_vehicles_3km]
    return out[cols].sort_values("vehicles_3km").reset_index(drop=True)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    src = "환경변수 DATA_DIR" if os.environ.get("DATA_DIR", "").strip() else "기본값 ../data/"
    print(f"DATA_DIR: {DATA_DIR}  ({src})")
    print("존재:", DATA_DIR.exists())
    if not DATA_DIR.exists():
        print("→ 폴더 구조가 다르면 환경변수 DATA_DIR 로 지정한다. 파일 목록은 README 참조.")
        raise SystemExit(1)
    print()

    lookup = build_dong_lookup()
    print(f"동 조회표      {len(lookup):>10,}개  "
          f"{lookup['match_type'].value_counts().to_dict()}")

    depots = load_depots()
    print(f"차고지         {len(depots):>10,}개  차량 {depots['capacity'].sum()}대")

    cands = load_candidates()
    enc, cap = cands.attrs["n_by_enclosure"], cands.attrs["capacity_by_enclosure"]
    print(f"후보 풀        {cands.attrs['n_pool']:>10,}곳  "
          + " / ".join(f"{k} {enc[k]}곳 {cap[k]:,.0f}면" for k in ("옥외", "옥내", "혼합")))
    print(f"후보(옥외만)   {len(cands):>10,}곳  {cands['capacity'].sum():,}면"
          f"  — 옥내·혼합 {cands.attrs['n_excluded_indoor_mixed']}곳 제외(A-15)")

    under = load_undersupplied()
    print(f"과소공급 동    {len(under):>10,}개")

    print("\n콜 로딩 중(첫 실행은 1~2분, 이후 캐시)...")
    calls = load_calls()
    n_raw = calls.attrs.get("n_raw")
    print(f"콜             {len(calls):>10,}건" + (f"  (원본 {n_raw:,}행)" if n_raw else "  (캐시)"))
    print(f"  기간         {calls['date'].min():%Y-%m-%d} ~ {calls['date'].max():%Y-%m-%d}")
    print(f"  취소율       {calls['is_canceled'].mean():.2%}"
          f"  (정산 미기록 {int(calls['is_unsettled'].sum()):,}건 제외)")
    print(f"  목적지 서울  {calls['dest_in_seoul'].mean():.1%}")
    print(f"  출발동 매칭  {calls['origin_match'].value_counts().to_dict()}")
    n_dest_gap = int(calls["dest_lat"].isna().sum())
    print(f"  목적동 좌표없음 {n_dest_gap:,}건 ({n_dest_gap/len(calls):.1%}) "
          f"— 대부분 서울 밖, travel_time 에서 거리 보조로 처리")

    w = calls["wait_min"].dropna()
    print(f"\n대기시간(분)   평균 {w.mean():.1f} / 중앙값 {w.median():.1f} / "
          f"p90 {w.quantile(.9):.1f} / 60분초과 {(w > 60).mean():.1%}")
    print("검증 기준      평균 40.8 / 중앙값 32.0 / p90 80.4 / 60분초과 19.2%")
