"""
metrics.py — 동별 진단 지표 계산

콜 데이터에서 행정동별 대기·운행·미이행·수요·공급·형평 지표를 계산한다.
진단 대시보드와 시뮬레이션 before/after 비교에서 동일한 계산 로직을 사용한다.

이 모듈은 파일을 직접 읽지 않고 데이터프레임을 입력받아 지표만 계산한다.
데이터 로딩과 전처리는 `load.py`가 담당한다.

동 키는 `(구, 표준화된 동명)`을 사용한다.
행정구역 기준 차이로 조인이 되지 않는 값은 제거하지 않고 결측값과 플래그로 남긴다.

실행
python src/metrics.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from travel_time import RIDE_MIN_RANGE, SPEED_RANGE, haversine_km

# ─────────────────────────────────────────────────────────────
# 검증 기준 · 상수
# ─────────────────────────────────────────────────────────────

# 검증 기준(현행 배치 재현 목표) — **특장차 한정 모집단**(A-16).
#
# 분모가 지표마다 다르다 — 대조할 때 헷갈리기 쉬우니 명시해 둔다.
#   대기시간 4개 : 승차 완료 건 1,033,135 (취소 건은 wait=NaN 이라 자동 제외)
#   취소율       : 시뮬 입력 모수 1,222,330 (즉시콜+서울 전체)
#   지역 지니    : 100건 이상 동 430개 (A-12), 동 균등가중
#
# 임차택시를 뺀 값이다. 구 기준(전체 152.7만)은 39.3 / 30.8 / 77.2 / 17.9% /
# 13.3% / 0.0947 이었다 — 취소율이 4.8%에 불과한 임차택시가 빠지면서 대기·취소가
# 함께 올라갔다. 근거는 A-16(docs/assa_log.md).
#
# cancel_ratio 의 분자에서 정산 미기록(is_unsettled — calibration.md 흡수 사실 ⑵)은 뺀다. 승차
# 기록이 없어 취소처럼 보이지만 배차→하차 구간이 실재해 운행은 이루어진 건이다.
# 넣으면 15.48%, 빼면 15.44% — 원 산출과 맞는 쪽은 후자다.
# **판정 대상은 픽업·취소·형평이다**(2026.08.09 재정의).
#
# 대기는 매칭(배차−접수)과 픽업(승차−배차)으로 갈리는데, **거점 배치가 실제로
# 바꾸는 것은 픽업뿐이다** — 매칭은 배차 시스템이 콜을 언제 차에 붙이느냐이고
# 위치와 상관이 적다. 그런데 시뮬은 매칭을 재현하지 못한다(실측 22.0분 vs 시뮬
# 3.7분). 원인이 기사 수락 절차 쪽으로 보이지만 데이터로 가릴 수 없어 이식하지
# 않는다 — 실측 분포를 상수로 박으면 배치가 매칭을 줄이는 경로가 0이 되고,
# 재현이 아니라 답을 베껴 넣는 것이 된다(A-19).
#
# 그래서 **총 대기 4개를 판정에서 빼고 참고로 돌린다.** 총 대기 = 매칭 + 픽업이라
# 매칭을 못 맞추면 총 대기는 구조적으로 짧게 나오고, 그 미달을 이유로 검증을
# 떨어뜨리면 정작 배치가 바꾸는 구간의 재현도를 볼 수 없다.
# **형평도 픽업 기준이다.** 총 대기 지니(0.092)는 매칭 지니(0.150)와 픽업 지니
# (0.070)의 혼합인데, 매칭을 재현하지 않으므로(A-19) 총 대기 지니로 채점하면
# 재현하지 않기로 한 구간을 채점하는 셈이 된다. 픽업 지니는 필터 없이도 실측과
# 가깝다(시뮬 0.0805 vs 실측 0.0751 — 허용 0.005 를 근소하게 넘긴다).
# 필터를 켜면 0.0519 로 멀어지므로 방향은 분명하다 — 공간 구조는 대체로 재현된다.
TARGETS = {
    "pickup_mean": 18.8,
    "pickup_p50": 17.8,
    "pickup_p90": 29.4,
    "cancel_ratio": 0.154,
    "pickup_gini": 0.070,
}

# 판정 허용오차. 시간은 분, 비율은 절대비율.
TARGET_TOL = {
    "pickup_mean": 0.5, "pickup_p50": 0.5, "pickup_p90": 1.0,
    "cancel_ratio": 0.002, "pickup_gini": 0.005,
}

# 참고값 — **판정하지 않는다.** 시뮬이 여기서 미달하는 것은 알려진 한계(A-19)이지
# 배치안의 성능이 아니다. 값이 움직였는지 보려고 함께 산출만 한다.
REFERENCE_TARGETS = {
    "mean_wait": 40.8,
    "median_wait": 32.0,
    "p90_wait": 80.4,
    "long_wait_ratio": 0.192,   # 60분 초과
    "match_mean": 22.0,         # 배차 − 접수. 시뮬이 재현하지 못하는 구간
    "abandoned_ratio": 0.0854,  # 대기 중 포기 — 인내심 곡선(A-14)의 검증 목표
    "gini_dong": 0.092,         # 총 대기 지니 — 매칭이 섞여 있어 참고로만
}

# A-12(정의는 docs/assa_log.md): 콜수가 적은 동은 평균이 크게 흔들려 지니를 왜곡한다.
# 100건 미만 제외.
# 원본 분석도 같은 기준이라 432개 동 중 2개(창신제3동 74건, 반포본동 24건)가 빠진다.
# 동별 표에서는 버리지 않고 is_reliable=False 로 표시만 한다.
MIN_CALLS_PER_DONG = 100

# 장기대기 판정 기준(분)
LONG_WAIT_MIN = 60

# 즉시 취소 판정(분) — 배차 전 취소 중 이 시간 안에 끊은 건은 변심으로 본다.
CANCEL_IMMEDIATE_MIN = 1.0

# 공급 근접도 반경(km)
SUPPLY_RADIUS_KM = 3.0

# 동 키 — 콜 데이터 컬럼명
DONG_KEYS = ("origin_gu", "origin_dong_canon")

# 대기 3구간. 배차가 늦은 건지(매칭) 차가 멀리 있는 건지(픽업) 갈라 보려는 것이다.
#   매칭 = 배차 − 접수 / 픽업 = 승차 − 배차 / 총 = 승차 − 접수
WAIT_KINDS = ("match", "pickup", "total")

# 미이행 4구간. abandoned 가 공급 부족의 핵심 신호다.
CANCEL_KINDS = ("immediate", "abandoned", "after_assign", "other")

# 표·대시보드 표기용 이름. travel_time.PERIOD_KO 와 같은 용도.
CANCEL_KIND_KO = {
    "immediate": "즉시 취소", "abandoned": "대기 중 포기",
    "after_assign": "배차 후 취소", "other": "기타(불명)",
}

# load.py 의 4구간 시간대(심야/아침/낮/저녁) → 컬럼명용 ASCII 키.
# travel_time 의 5구간과는 기준이 다르다. 섞어 쓰지 말 것.
PERIOD_EN = {"심야": "night", "아침": "morning", "낮": "day", "저녁": "evening"}

# 구/서울 평균 대비를 붙일 지표(기본)
COMPARE_COLS = ("wait_total_mean", "long_wait_ratio", "cancel_abandoned_ratio",
                "usage_rate", "vehicles_3km")


# ─────────────────────────────────────────────────────────────
# 콜 단위 파생 — 원본을 건드리지 않고 Series 로 돌려준다
# ─────────────────────────────────────────────────────────────

def pickup_min(calls: pd.DataFrame) -> pd.Series:
    """픽업 대기(분) = 승차 − 배차. 차가 얼마나 멀리 있었나."""
    return (calls["boarded_at"] - calls["assigned_at"]).dt.total_seconds() / 60


def ride_km(calls: pd.DataFrame) -> pd.Series:
    """승차거리(km). 원본이 미터라 나눈다."""
    return calls["ride_distance_m"] / 1000


def ride_kmh(calls: pd.DataFrame) -> pd.Series:
    """차내 평균속도(km/h) = 승차거리 ÷ 소요시간."""
    return ride_km(calls) / (calls["ride_min"] / 60)


def ride_valid(calls: pd.DataFrame) -> pd.Series:
    """차내 지표에 쓸 만한 운행인지. travel_time 의 이상치 컷과 같은 기준이다.

    운행시간 1~120분 · 거리 > 0 · 속도 2~60km/h.

    **거리 > 0 이 calibration.md 흡수 사실 ⑴ 의 함정을 막는 컷이다.** 승차거리 0 은 결측 대체값이라
    미탑승 건뿐 아니라 완료 건 4,636건에도 섞여 있다(미터기·GPS 미기록. 요금은
    정상). 정상 완료 건의 중앙이 5,578m 인데 0 을 섞으면 거리·속도 평균이 눌린다.
    정산 미기록 건(calibration.md 흡수 사실 ⑵)도 승차 시각이 없어 ride_min 이 NaN 이라 여기서 걸린다.
    """
    return (calls["ride_min"].between(*RIDE_MIN_RANGE)
            & (calls["ride_distance_m"] > 0)
            & ride_kmh(calls).between(*SPEED_RANGE))


def is_unsettled(calls: pd.DataFrame) -> pd.Series:
    """정산 미기록 — 승차·취소가 없는데 하차만 있는 건(calibration.md 흡수 사실 ⑵, 필터 후 524건).

    배차→하차 중앙 56.6분이라 운행은 이루어졌다. 요금·거리는 0으로 남지 않았다.
    **취소가 아니다** — 미이행 집계에서 빼고, 요금·거리 집계에서도 뺀다.
    승차 시각이 없어 대기시간은 애초에 산출되지 않는다(wait_min=NaN).

    load.load_calls 가 같은 규칙으로 `is_unsettled` 컬럼을 붙여두지만, 이 모듈의
    함수들은 시뮬 로그도 받으므로 컬럼에 기대지 않고 여기서 다시 판정한다.
    """
    canceled_at = pd.to_datetime(calls["canceled_at"], errors="coerce")
    return (calls["boarded_at"].isna() & canceled_at.isna()
            & calls["alighted_at"].notna())


def is_canceled(calls: pd.DataFrame) -> pd.Series:
    """미이행 판정. 승차 기록이 없고 정산 미기록도 아닌 건."""
    return calls["boarded_at"].isna() & ~is_unsettled(calls)


def cancel_kind(calls: pd.DataFrame) -> pd.Series:
    """미이행 4구간 분류. 승차한 건과 정산 미기록 건은 NaN.

      immediate    배차 전 + 접수 후 1분 이내 취소 — 오입력·변심
      abandoned    배차 전 + 1분 초과 — **기다리다 포기. 공급 부족의 핵심 신호**
      after_assign 배차는 됐는데 승차 기록이 없음 — 차가 오다 어긋난 건
      other        배차 전 취소 중 취소 시각이 결측이라 즉시/포기를 판별할 수 없는
                   건. 근거 없는 분류를 피해 별도 구간으로 둔다.
                   특장차 한정본에서는 0건이다(정제 단계에서 걸러졌다).

    정산 미기록(is_unsettled)은 배차·하차만 있어 그냥 두면 after_assign 으로
    들어간다. 운행이 이루어진 건이라 미이행이 아니므로 앞에서 뺀다 — 안 빼면
    취소율이 15.44% → 15.48%, 배차 후 취소가 3.74% → 3.78% 로 부푼다.
    """
    canceled = is_canceled(calls)
    assigned = calls["assigned_at"].notna()
    # 취소가 한 건도 없는 로그(시뮬 결과 등)는 canceled_at 이 전부 결측이라 datetime
    # 으로 안 잡힌다. 실측 datetime64 에는 사실상 무비용이라 그냥 한 번 통과시킨다.
    canceled_at = pd.to_datetime(calls["canceled_at"], errors="coerce")
    delay = (canceled_at - calls["received_at"]).dt.total_seconds() / 60

    kind = pd.Series(np.nan, index=calls.index, dtype=object)
    pre = canceled & ~assigned
    kind[canceled & assigned] = "after_assign"
    kind[pre & (delay <= CANCEL_IMMEDIATE_MIN)] = "immediate"
    kind[pre & (delay > CANCEL_IMMEDIATE_MIN)] = "abandoned"
    kind[pre & delay.isna()] = "other"
    return pd.Categorical(kind, categories=CANCEL_KINDS)


# ─────────────────────────────────────────────────────────────
# 동별 집계
# ─────────────────────────────────────────────────────────────

def _by_dong(df: pd.DataFrame, keys=DONG_KEYS):
    return df.groupby(list(keys), observed=True)


def _tidy(out: pd.DataFrame) -> pd.DataFrame:
    """집계 결과의 동 키를 표 컬럼으로 되돌린다. category 는 str 로 풀어 조인에 쓴다."""
    out = out.reset_index().rename(columns={
        "origin_gu": "gu", "origin_dong_canon": "dong_canon", "origin_dong": "dong"})
    for col in ("gu", "dong_canon", "dong"):
        if col in out.columns:
            out[col] = out[col].astype(str)
    return out


def _stats(grouped, col: str, prefix: str) -> pd.DataFrame:
    """그룹별 평균 · 중앙(p50) · p90 세 컬럼."""
    g = grouped[col]
    out = g.agg(**{f"{prefix}_mean": "mean", f"{prefix}_p50": "median"})
    out[f"{prefix}_p90"] = g.quantile(0.90)
    return out


def dong_wait(calls: pd.DataFrame, *, keys=DONG_KEYS) -> pd.DataFrame:
    """동별 대기 지표. 매칭/픽업/총 각각 평균·p50·p90 + 장기대기율.

    분모는 대기시간이 산출된 건(=승차 완료)이다. 취소 건은 대기시간이 NaN 이라
    평균에서 빠지고 n_served 에도 잡히지 않는다 — 원본 분석의 동별 집계와 같은 정의다.

    매칭·픽업은 배차 기록이 있는 건만이라 총 대기보다 분모가 작다(n_assigned).
    """
    df = calls.assign(pickup_min=pickup_min(calls),
                      is_long=calls["wait_min"] > LONG_WAIT_MIN)
    served = df[df["wait_min"].notna()]

    g = _by_dong(served, keys)
    out = _stats(g, "wait_min", "wait_total")
    out["n_served"] = g.size()
    out["long_wait_ratio"] = g["is_long"].mean()

    for kind, col in (("match", "assign_min"), ("pickup", "pickup_min")):
        sub = served[served[col].notna()]
        out = out.join(_stats(_by_dong(sub, keys), col, f"wait_{kind}"))
    out["n_assigned"] = _by_dong(served[served["assign_min"].notna()], keys).size()

    return _tidy(out)


def dong_ride(calls: pd.DataFrame, *, keys=DONG_KEYS) -> pd.DataFrame:
    """동별 차내 지표. 소요시간(분) · 속도(km/h) · 거리(km) 평균.

    이상치를 뺀 정상 운행만 쓴다(ride_valid). 분모는 n_ride 로 따로 남긴다 —
    n_served 와 다르다.
    """
    df = calls.assign(ride_km=ride_km(calls), ride_kmh=ride_kmh(calls))
    ok = df[ride_valid(calls)]

    g = _by_dong(ok, keys)
    out = g.agg(n_ride=("ride_min", "size"),
                ride_min_mean=("ride_min", "mean"),
                ride_kmh_mean=("ride_kmh", "mean"),
                ride_km_mean=("ride_km", "mean"))
    return _tidy(out)


def dong_cancel(calls: pd.DataFrame, *, keys=DONG_KEYS) -> pd.DataFrame:
    """동별 미이행 4구간. 즉시 취소 / 대기 중 포기 / 배차 후 취소 / 기타(불명).

    같은 분해를 분모 둘로 낸다. 관점이 다르지 각각 답하는 질문이 다르다.

      cancel_{종류}_ratio  콜 대비 — 분모는 전체 접수 콜(취소 포함).
                           '이 동에서 콜을 넣으면 얼마나 미이행되나'
      cancel_{종류}_share  취소 내 구성비 — 분모는 그 동의 전체 취소 건.
                           '미이행된 콜이 왜 미이행됐나'

    ratio 4개의 합 = cancel_ratio, share 4개의 합 = 1 로 정확히 맞는다.
    **검증 채점과 전체 취소율 15.44%는 ratio(콜 대비) 기준이다.** share 는 해석용
    보조 지표라 분모를 바꾸지 않는다.

    ratio 의 분모(n_calls)에는 정산 미기록 건이 남아 있다 — 접수된 콜인 것은
    맞아서다. 분자에서만 빠진다.

    취소가 한 건도 없는 동은 share 가 NaN 이다(0/0). 0으로 채우면 '전부 즉시
    취소였다'와 구분이 안 된다.
    """
    df = calls.assign(cancel_kind=cancel_kind(calls))
    n = _by_dong(df, keys).size().rename("n_calls")

    counts = (_by_dong(df, keys)["cancel_kind"]
              .value_counts().unstack(fill_value=0)
              .reindex(columns=list(CANCEL_KINDS), fill_value=0))

    out = counts.div(n, axis=0).add_prefix("cancel_").add_suffix("_ratio")
    out["cancel_ratio"] = out.sum(axis=1)

    n_canceled = counts.sum(axis=1)
    share = counts.div(n_canceled.replace(0, np.nan), axis=0)
    out = out.join(share.add_prefix("cancel_").add_suffix("_share"))
    out["n_canceled"] = n_canceled
    return _tidy(out)


def dong_demand(calls: pd.DataFrame, *, keys=DONG_KEYS) -> pd.DataFrame:
    """동별 수요. 접수 콜 수 + 시간대(4구간)·평일/주말 분해 + 동 중심좌표.

    좌표는 load.load_calls 가 행마다 붙여둔 출발동 중심점이다(동 안에서 같은 값).
    """
    df = calls.assign(is_weekend=calls["weekday"] >= 5)
    g = _by_dong(df, keys)

    out = g.agg(n_calls=("call_id", "size"),
                lat=("origin_lat", "first"), lon=("origin_lon", "first"),
                dong=("origin_dong", "first"))

    period = (g["period"].value_counts().unstack(fill_value=0)
              .rename(columns=PERIOD_EN).add_prefix("n_calls_"))
    weekend = g["is_weekend"].sum()
    week = pd.DataFrame({"n_calls_weekday": out["n_calls"] - weekend,
                         "n_calls_weekend": weekend})

    return _tidy(out.join(period).join(week))


def demand_matrix(calls: pd.DataFrame, *, keys=DONG_KEYS) -> pd.DataFrame:
    """동 × 시간대 × 평일/주말 콜 수 매트릭스(long form).

    동별 표에 접는 대신 원형을 남긴다 — 대시보드에서 시간대 슬라이더를 걸거나
    시뮬 도착 프로세스를 동별로 맞출 때 쓴다.

    반환 컬럼: gu, dong_canon, period, is_weekend, n_calls
    """
    df = calls.assign(is_weekend=calls["weekday"] >= 5)
    out = (df.groupby(list(keys) + ["period", "is_weekend"], observed=True)
             .size().rename("n_calls").reset_index())
    out = out.rename(columns={"origin_gu": "gu", "origin_dong_canon": "dong_canon"})
    for col in ("gu", "dong_canon", "period"):
        out[col] = out[col].astype(str)
    return out.sort_values(["gu", "dong_canon", "period", "is_weekend"]).reset_index(drop=True)


def dong_supply(dong_xy: pd.DataFrame, depots: pd.DataFrame, *,
                radius_km: float = SUPPLY_RADIUS_KM) -> pd.DataFrame:
    """동 중심점 반경 안의 차고지 배속 차량 합(3km 근접도).

    haversine 직선거리다. 실제 접근은 도로를 타므로 낙관적인 값이고, 반경 경계에
    걸친 차고지는 들어오거나 빠지거나로 갈린다(계단 함수). 432 × 44 라 전수 계산한다.

    dong_xy: gu, dong_canon, lat, lon / depots: lat, lon, capacity
    반환: gu, dong_canon, vehicles_3km, n_depots_3km
    """
    d = haversine_km(dong_xy["lat"].to_numpy()[:, None], dong_xy["lon"].to_numpy()[:, None],
                     depots["lat"].to_numpy()[None, :], depots["lon"].to_numpy()[None, :])
    within = d <= radius_km

    out = dong_xy[["gu", "dong_canon"]].copy()
    out["vehicles_3km"] = within @ depots["capacity"].to_numpy()
    out["n_depots_3km"] = within.sum(axis=1)
    return out


def dong_population(dong_keys: pd.DataFrame, population: pd.DataFrame) -> pd.DataFrame:
    """동별 등록 장애인 수를 콜 데이터의 동 구분에 맞춰 붙인다.

    두 파일의 행정구역 판이 달라 표준형만으로는 432개 중 18개가 안 맞는다. 콜 쪽이
    더 잘게 쪼개진 경우(중구 신당1~6동 ↔ 통계 신당동)와 더 뭉뚱그린 경우(강동구
    상일동 ↔ 통계 상일1·2동)가 섞여 있다.

    **n_disabled 는 그 동 하나에 귀속되는 인구다.** 표준형으로 먼저 맞추고, 남은
    건만 기본명(숫자 뗀 이름) 그룹으로 재시도하되 **기본명 그룹을 두 개 이상의 동이
    다투면 붙이지 않는다** — 신당1~6동처럼 콜 쪽이 더 잘면 통계 한 줄을 어떻게
    나눌지 근거가 없어서다. 이미 표준형으로 붙은 통계 행은 기본명 후보에서 뺀다
    (공릉2동 이중 계상 방지). 이 컬럼의 합은 통계 총합을 넘지 않는다.

    **pop_group 은 그 동이 속한 기본명 그룹 전체의 인구다.** 위까지 하고도 못 붙은
    동이 남는 그룹에만 채운다(답십리1~4동 ↔ 통계 답십리1·2동 등). 나누는 게 아니라
    양쪽을 다 더해 단위를 맞추는 것이라 근거가 있다. 같은 그룹의 동은 같은 값을
    갖고, 이용률 계산은 build_dong_table 이 콜도 같은 그룹으로 합쳐서 한다.
    n_disabled 와 달리 그룹 안에서 중복되므로 **합계를 내면 안 된다.**

    반환: gu, dong_canon, n_disabled, pop_match ('canon' / 'base' / NaN),
          dong_base, pop_group
    """
    from load import _base_dong  # 동명 정규화 규칙은 load.py 한 곳에서만 정의한다

    out = dong_keys[["gu", "dong_canon"]].copy()
    out["dong_base"] = out["dong_canon"].map(_base_dong)
    exact = population.set_index(["gu", "dong_canon"])["n_disabled"]
    out["n_disabled"] = pd.MultiIndex.from_frame(out[["gu", "dong_canon"]]).map(exact)
    out["pop_match"] = np.where(out["n_disabled"].notna(), "canon", None)
    out["pop_group"] = np.nan

    todo = out.loc[out["n_disabled"].isna(), ["gu", "dong_canon", "dong_base"]].copy()
    if todo.empty:
        return out

    todo["n_claim"] = (todo.groupby(["gu", "dong_base"], observed=True)["dong_canon"]
                           .transform("size"))

    matched = set(zip(out.loc[out["pop_match"] == "canon", "gu"],
                      out.loc[out["pop_match"] == "canon", "dong_canon"]))
    free = population[~pd.Series(list(zip(population["gu"], population["dong_canon"])),
                                 index=population.index).isin(matched)]
    # min_count=1 — 통계값이 '-'(NaN)뿐인 그룹까지 0명으로 만들면 안 된다(반포본동).
    group_sum = free.groupby(["gu", "dong_base"])["n_disabled"].sum(min_count=1)

    solo = todo[todo["n_claim"] == 1]
    vals = pd.MultiIndex.from_frame(solo[["gu", "dong_base"]]).map(group_sum)
    out.loc[solo.index, "n_disabled"] = vals
    out.loc[solo.index, "pop_match"] = np.where(pd.notna(vals), "base", None)

    # 여기까지도 못 붙은 동이 있는 그룹 — 통계 쪽 전체를 그룹 인구로 삼는다.
    # 잔여분(free)이 아니라 그룹 전체다. 답십리3·4동을 살리려면 이미 정확히 붙은
    # 답십리1·2동의 인구까지 분모에 들어가야 하고, 그러려면 콜도 4개 동을 다 더해야
    # 한다 — 분자·분모의 단위를 같이 옮기는 것이라 이중 계상이 아니다.
    left = out.loc[out["n_disabled"].isna(), ["gu", "dong_base"]].drop_duplicates()
    if not left.empty:
        whole = population.groupby(["gu", "dong_base"])["n_disabled"].sum(min_count=1)
        keys = pd.MultiIndex.from_frame(out[["gu", "dong_base"]])
        in_group = keys.isin(pd.MultiIndex.from_frame(left))
        out["pop_group"] = np.where(in_group, keys.map(whole), np.nan)

    return out


# ─────────────────────────────────────────────────────────────
# 비교(구내 / 타구 / 서울)
# ─────────────────────────────────────────────────────────────

def add_comparisons(tbl: pd.DataFrame, cols=COMPARE_COLS, *,
                    reliable_baseline: bool = True) -> pd.DataFrame:
    """각 동 값에 '구 평균 대비 / 서울 평균 대비'를 붙인다.

    기준선은 **동 균등가중 평균**이다(콜수 가중 아님) — 형평을 동 간 격차로 보는
    지니와 관점을 맞춘다. 표본이 얇은 동이 기준선을 흔들지 않도록 기본은 신뢰 동
    (100건 이상)만으로 평균을 내되, 비율 자체는 모든 동에 계산한다.

    붙는 컬럼: {col}_gu_mean, {col}_seoul_mean, {col}_vs_gu, {col}_vs_seoul
    (vs_* 는 배수다. 1.2 면 기준선보다 20% 높다는 뜻.)

    없는 컬럼을 넘기면 바로 KeyError 다. 조용히 건너뛰면 오타 하나에 비교 컬럼이
    통째로 빠진 표가 멀쩡한 얼굴로 나온다.
    """
    missing = [c for c in cols if c not in tbl.columns]
    if missing:
        raise KeyError(f"비교할 컬럼이 표에 없다: {missing}")

    out = tbl.copy()
    base = out[out["is_reliable"]] if reliable_baseline and "is_reliable" in out else out

    for col in cols:
        gu_mean = out["gu"].map(base.groupby("gu")[col].mean())
        seoul_mean = base[col].mean()
        out[f"{col}_gu_mean"] = gu_mean
        out[f"{col}_seoul_mean"] = seoul_mean
        out[f"{col}_vs_gu"] = out[col] / gu_mean.replace(0, np.nan)
        out[f"{col}_vs_seoul"] = out[col] / (seoul_mean or np.nan)
    return out


# ─────────────────────────────────────────────────────────────
# 동별 표 조립
# ─────────────────────────────────────────────────────────────

# 표 컬럼 순서 — 식별 → 수요 → 대기 → 차내 → 미이행 → 공급 → 인구.
# 비교 컬럼(_vs_gu 등)은 여기 없는 것들과 함께 뒤에 원래 순서대로 붙는다.
_COLUMN_ORDER = (
    "gu", "dong", "dong_canon", "lat", "lon", "is_reliable",
    "n_calls", "n_served", "n_assigned", "n_ride",
    "n_calls_night", "n_calls_morning", "n_calls_day", "n_calls_evening",
    "n_calls_weekday", "n_calls_weekend",
    "wait_total_mean", "wait_total_p50", "wait_total_p90", "long_wait_ratio",
    "wait_match_mean", "wait_match_p50", "wait_match_p90",
    "wait_pickup_mean", "wait_pickup_p50", "wait_pickup_p90",
    "ride_min_mean", "ride_kmh_mean", "ride_km_mean",
    "n_canceled", "cancel_ratio",
    "cancel_immediate_ratio", "cancel_abandoned_ratio",
    "cancel_after_assign_ratio", "cancel_other_ratio",
    "cancel_immediate_share", "cancel_abandoned_share",
    "cancel_after_assign_share", "cancel_other_share",
    "vehicles_3km", "n_depots_3km",
    "n_disabled", "pop_match", "usage_pop", "usage_is_grouped", "usage_rate",
)


def _order_columns(tbl: pd.DataFrame) -> pd.DataFrame:
    head = [c for c in _COLUMN_ORDER if c in tbl.columns]
    return tbl[head + [c for c in tbl.columns if c not in head]]


def build_dong_table(calls: pd.DataFrame, *, depots: pd.DataFrame = None,
                     population: pd.DataFrame = None,
                     min_calls: int = MIN_CALLS_PER_DONG,
                     compare_cols=COMPARE_COLS) -> pd.DataFrame:
    """동별 진단 지표 표. 대시보드가 읽는 산출물이자 시뮬 before/after 의 한 쪽.

    depots / population 을 넘기지 않으면 공급·이용률 컬럼만 빠지고 나머지는 그대로다
    (시뮬 로그처럼 콜만 있는 입력에도 쓸 수 있게).

    콜이 한 건이라도 있는 동은 전부 남기고, 100건 미만은 is_reliable=False 로만
    표시한다 — 지도에서 지워버리면 '데이터가 없는 동'과 구분이 안 된다.

    attrs: n_dong, n_reliable, n_pop_unmatched(이용률 NaN 동 수), pop_unmatched(동 목록)
    """
    tbl = dong_demand(calls)
    for part in (dong_wait(calls), dong_ride(calls), dong_cancel(calls)):
        tbl = tbl.merge(part, on=["gu", "dong_canon"], how="left")

    # 신뢰 기준은 지니와 같은 분모로 잰다 — 승차 완료 건이지 접수 건이 아니다.
    # 접수 기준으로 재면 창신제3동(접수 102 / 승차 74)처럼 지니에서는 빠진 동이
    # 표에서는 신뢰로 잡혀 두 지표의 모집단이 어긋난다.
    # 접수 102 가 문턱 100 을 2건 넘긴 것이라, 원본이 조금만 바뀌면 이 예시는 사라진다.
    tbl["is_reliable"] = tbl["n_served"].fillna(0) >= min_calls

    if depots is not None:
        tbl = tbl.merge(dong_supply(tbl[["gu", "dong_canon", "lat", "lon"]], depots),
                        on=["gu", "dong_canon"], how="left")

    unmatched = []
    if population is not None:
        pop = dong_population(tbl[["gu", "dong_canon"]], population)
        tbl = tbl.merge(pop, on=["gu", "dong_canon"], how="left")

        # 이용률 = 연간 접수 콜 ÷ 등록 장애인. 1인당 몇 번 불렀나.
        # 분모는 원칙적으로 그 동의 n_disabled 다. 콜이 통계보다 잘게 쪼개진 그룹만
        # 분자·분모를 함께 기본명 그룹으로 올려 단위를 맞춘다(usage_is_grouped).
        # 그룹 안의 동들은 같은 값을 갖는다 — 동 사이를 가를 근거가 원본에 없다.
        num = tbl["n_calls"].astype(float)
        tbl["usage_pop"] = tbl["n_disabled"]
        tbl["usage_is_grouped"] = False

        grouped = tbl["pop_group"].notna()
        if grouped.any():
            num[grouped] = (tbl[grouped].groupby(["gu", "dong_base"], observed=True)
                                        ["n_calls"].transform("sum"))
            tbl.loc[grouped, "usage_pop"] = tbl.loc[grouped, "pop_group"]
            tbl.loc[grouped, "usage_is_grouped"] = True

        tbl["usage_rate"] = num / tbl["usage_pop"].replace(0, np.nan)
        tbl = tbl.drop(columns=["dong_base", "pop_group"])
        unmatched = sorted(tbl.loc[tbl["usage_rate"].isna(), "dong_canon"])

    # depots/population 을 안 넘기면 그쪽 비교는 자연히 빠진다. 그 외의 이름은
    # add_comparisons 가 오타로 보고 튕긴다.
    tbl = add_comparisons(tbl, [c for c in compare_cols if c in tbl.columns])
    tbl = _order_columns(tbl)
    tbl = tbl.sort_values(["gu", "dong_canon"]).reset_index(drop=True)

    # attrs 는 merge/sort 를 거치며 사라질 수 있어 마지막에 한 번에 붙인다.
    tbl.attrs["n_dong"] = len(tbl)
    tbl.attrs["n_reliable"] = int(tbl["is_reliable"].sum())
    tbl.attrs["min_calls"] = min_calls
    tbl.attrs["n_pop_unmatched"] = len(unmatched)
    tbl.attrs["pop_unmatched"] = unmatched
    return tbl


# ─────────────────────────────────────────────────────────────
# 차량-일 집계 (동별 아님 — 조 편성 실측 재구성의 입력, 가정 A-09)
# ─────────────────────────────────────────────────────────────

def vehicle_day_table(trips: pd.DataFrame) -> pd.DataFrame:
    """차량-일 단위 집계. trips 는 load.load_vehicle_trips() 형식.

    가동시간은 그날 첫 승차 ~ 마지막 하차 구간(span)으로 잡는다. 배차를 기다리며
    서 있던 시간은 운행 기록에 안 남아 이 방법 말고는 알 길이 없다. 그래서 실제
    출차는 이보다 이르고 귀고는 이보다 늦다.

    조별 출차 시각·근무 길이를 실측에서 역추정하는 데 쓴다(A-09 —
    docs/model_flow.md 조 편성 절).

    first_board 는 조별 출차 시각의 대리값이라 그대로 돌려준다 — A-09 의 출차
    분포를 이 컬럼의 시(hour)에서 뽑는다.

    반환 컬럼: vehicle_id, service_date, trips, occupied_min, first_board, span_h
    """
    g = trips.groupby(["vehicle_id", "service_date"])
    out = g.agg(trips=("ride_min", "size"), occupied_min=("ride_min", "sum"),
                first_board=("boarded_at", "min"), last_alight=("alighted_at", "max"))
    out["span_h"] = (out["last_alight"] - out["first_board"]).dt.total_seconds() / 3600
    return out.reset_index()[["vehicle_id", "service_date", "trips",
                              "occupied_min", "first_board", "span_h"]]


def daily_active_vehicles(trips: pd.DataFrame, *,
                          date_col: str = "request_date") -> pd.Series:
    """일별 가동 차량 수. **시뮬에 넣을 차량 대수의 실측 기준이다.**

    원본의 고유 차량번호 723대는 **연간 누적**이라 그대로 넣으면 공급이 과대해진다
    (calibration.md 흡수 사실 ⑶). 교체·증차분이 전부 잡히고, 정비·휴차로 그날 안 나온 차도 포함된다.

    실측은 접수일 기준 **중앙 561대(범위 246~626)** 지만 **하나의 값으로 접으면
    안 된다** — 평일 중앙 574대 · 주말 중앙 374대로 1.5배 갈린다. 시뮬은
    `SIM_FLEET` 처럼 **요일별로 다른 대수를 넣는다**(결정 2026.08.08). 연간 중앙
    561을 평일에 쓰면 공급이 과소해져 대기가 과대 재현되고, 검증에서 원인이 배차
    로직인지 차량 수인지 갈리지 않는다.

    date_col 기본값은 request_date(접수일)다. 승차일 기준으로 세면 자정을 넘긴
    운행 때문에 중앙 540대로 20대가 낮게 나온다 — 명세의 561 과 대조하려면
    접수일 기준이어야 한다.
    """
    return trips.groupby(date_col)["vehicle_id"].nunique()


# 시뮬에 넣을 차량 대수 — **요일별로 다르게 적용한다**(결정 2026.08.08).
#
# 연간 중앙 561대를 평일에 쓰면 공급이 과소해져 대기가 과대 재현되고, 대표 기간이
# 평일 중심이라 그 왜곡이 그대로 검증에 들어간다. 그러면 재현 실패의 원인이 배차
# 로직인지 차량 수인지 구분되지 않는다.
#
# 값은 daily_active_vehicles() 의 요일별 중앙값이다. 원본에서 운행시간 1~120분 컷
# 없이 세면 평일이 575대로 1대 높다 — 그 컷은 조 편성 분석용이라 가동 여부 판정에는
# 과하지만, 한 함수에서 나온 값으로 통일하는 쪽을 택했다.
SIM_FLEET = {"weekday": 574, "weekend": 374}


# ─────────────────────────────────────────────────────────────
# 형평
# ─────────────────────────────────────────────────────────────

def dong_wait_table(log: pd.DataFrame, *, min_calls: int = MIN_CALLS_PER_DONG,
                    gu_col: str = "origin_gu", dong_col: str = "origin_dong",
                    wait_col: str = "wait_min") -> pd.DataFrame:
    """동별 평균대기·콜수 집계표. 콜수 min_calls 미만 동은 제외.

    지니 전용의 최소 집계다 — 동별 표(build_dong_table)와 달리 원본 분석과 대조할
    수 있게 동명을 원본 표기 그대로 두고, 기준 미만 동은 아예 뺀다.

    취소 건은 wait 이 NaN 이라 평균에서 빠지지만 콜수에는 잡히면 안 된다.
    여기서 말하는 '건수'는 대기시간이 산출된 건(=승차 완료) 기준이다.
    원본 분석의 동별 집계와 같은 정의라 건수가 동별로 일치한다.

    제외 내역은 attrs['n_dong_excluded'] / attrs['excluded'] 에 남긴다.
    """
    served = log[log[wait_col].notna()]
    tbl = (served.groupby([gu_col, dong_col], observed=True)[wait_col]
                 .agg(n_calls="size", mean_wait="mean")
                 .reset_index()
                 .rename(columns={gu_col: "gu", dong_col: "dong"}))

    keep = tbl["n_calls"] >= min_calls
    out = tbl[keep].sort_values(["gu", "dong"]).reset_index(drop=True)
    out.attrs["n_dong_excluded"] = int((~keep).sum())
    out.attrs["excluded"] = tbl[~keep].sort_values("n_calls").reset_index(drop=True)
    out.attrs["min_calls"] = min_calls
    return out


def dong_gini(log: pd.DataFrame, *, min_calls: int = MIN_CALLS_PER_DONG,
              weighted: bool = False, **kw) -> float:
    """동간 대기시간 지니(**동 균등가중**). 검증 기준 0.092.

    형평을 '동 간 격차'로 정의하므로 각 동을 한 표로 세는 균등가중이 정본이다.
    콜수로 가중하면 콜이 많은 동이 지표를 지배해 정작 소외된 동의 격차가 묻힌다.
    특장차 한정(A-16) 실측은 균등가중 **0.0924**다.

    weighted=True 로 콜수 가중(0.0986)도 뽑을 수 있게 남겨뒀다 — 참고용이며
    검증 채점에는 쓰지 않는다.

    **모집단이 19.5% 줄어도 지니는 0.0947 → 0.0924 로 거의 안 움직인다.** 동 간 격차
    구조는 두 차종에서 같다는 뜻이고, 그래서 이 지표는 모집단 교체에 둔감하다.
    """
    tbl = dong_wait_table(log, min_calls=min_calls, **kw)
    return gini(tbl["mean_wait"], weights=tbl["n_calls"] if weighted else None)


def pickup_gini(log: pd.DataFrame, *, min_calls: int = MIN_CALLS_PER_DONG) -> float:
    """동간 **픽업 대기** 지니(동 균등가중). 검증 기준 0.070.

    **형평의 판정 지표는 이쪽이다.** 총 대기 지니(0.092)는 매칭 지니(0.150)와
    픽업 지니(0.070)가 섞인 값인데, 시뮬은 매칭을 재현하지 않기로 했다(A-19).
    재현하지 않는 구간이 섞인 지표로 채점하면 그 미재현이 형평 점수로 넘어온다.

    **거점 배치가 줄이려는 격차가 곧 픽업 격차다** — 차가 멀리 있어서 생기는
    차이. 동별 픽업 평균은 실측 13.9~29.3분으로 2.1배 벌어져 있다.
    """
    d = log.assign(_pickup=pickup_min(log))
    return dong_gini(d, min_calls=min_calls, wait_col="_pickup")


def gini(values, weights=None) -> float:
    """지니계수(가중치 선택 가능). 0 = 완전균등, 1 = 완전불균등.

    정렬 후 누적합으로 계산한다(쌍별 차이 이중루프는 430개 동에도 느리다).
      G = 1 − Σ wᵢ(Sᵢ₋₁ + Sᵢ) / (W · Sₙ),  S = 누적 Σ w·x, W = Σ w

    가중치가 없으면 균등가중. NaN·비양수 가중치 행은 버린다.
    """
    v = np.asarray(values, dtype=float)
    w = np.ones_like(v) if weights is None else np.asarray(weights, dtype=float)
    if v.shape != w.shape:
        raise ValueError(f"values{v.shape} 와 weights{w.shape} 길이가 다르다")

    keep = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v, w = v[keep], w[keep]
    if v.size == 0:
        return float("nan")
    if (v < 0).any():
        raise ValueError("음수 값에는 지니를 정의하지 않는다(대기시간은 0 이상)")

    order = np.argsort(v, kind="stable")
    v, w = v[order], w[order]

    cum = np.cumsum(w * v)
    total = cum[-1]
    if total <= 0:            # 전부 0 이면 완전균등
        return 0.0
    prev = np.concatenate(([0.0], cum[:-1]))
    return float(1 - np.sum(w * (prev + cum)) / (w.sum() * total))


# ─────────────────────────────────────────────────────────────
# 검증
# ─────────────────────────────────────────────────────────────

def overall_metrics(calls: pd.DataFrame) -> dict:
    """전체(서울 합산) 지표. `TARGETS`(판정) + `REFERENCE_TARGETS`(참고) 전부.

    분모가 갈린다 — 대기·픽업은 승차 완료 건, 취소율·포기율은 접수 전체,
    지니는 100건 이상 동이다.

    **픽업(승차−배차)이 판정의 중심이다.** 거점 배치가 바꾸는 구간이 여기다.
    매칭(배차−접수)은 배차 시스템이 정하는 구간이라 참고로만 낸다.
    """
    w = calls["wait_min"].dropna()
    p = pickup_min(calls).dropna()
    kind = pd.Series(cancel_kind(calls))
    return {
        # 판정 대상
        "pickup_mean": float(p.mean()),
        "pickup_p50": float(p.median()),
        "pickup_p90": float(p.quantile(0.90)),
        "cancel_ratio": float(is_canceled(calls).mean()),
        "pickup_gini": pickup_gini(calls),
        # 참고
        "mean_wait": float(w.mean()),
        "median_wait": float(w.median()),
        "p90_wait": float(w.quantile(0.90)),
        "long_wait_ratio": float((w > LONG_WAIT_MIN).mean()),
        "match_mean": float(calls["assign_min"].dropna().mean()),
        "abandoned_ratio": float((kind == "abandoned").mean()),
        "gini_dong": dong_gini(calls),
    }


def cancel_breakdown(calls: pd.DataFrame) -> pd.DataFrame:
    """전체(서울 합산) 미이행 4구간 표. 건수 · 콜 대비 · 취소 내 구성비.

    동별 표를 평균 낸 게 아니라 건수를 그대로 합친 값이다. 동 균등가중 평균과는
    다르니 인용할 때 어느 쪽 기준인지 밝힐 것 — 콜이 많은 동이 여기서는 더 무겁다.

    '콜 대비' 합 = 전체 취소율(15.44%), '취소 중 비중' 합 = 1.
    """
    kind = pd.Series(cancel_kind(calls))
    counts = kind.value_counts().reindex(list(CANCEL_KINDS), fill_value=0)
    n = counts.to_numpy()
    return pd.DataFrame({
        "구분": [CANCEL_KIND_KO[k] for k in CANCEL_KINDS],
        "건수": n,
        "콜 대비": n / len(calls),
        "취소 중 비중": n / n.sum(),
    })


def compare_to_target(result: dict, *, targets: dict = None,
                      reference: dict = None) -> pd.DataFrame:
    """검증 대조표. **판정은 `TARGETS` 만** 하고 `REFERENCE_TARGETS` 는 함께 싣는다.

    참고 행의 `판정` 은 빈칸이다 — 시뮬이 총 대기·매칭에서 미달하는 것은 알려진
    한계(A-19)이지 배치안의 성능이 아니라, 여기서 X 를 매기면 그 한계가 후보
    비교를 가로막는다.

    `targets`/`reference` 를 주면 그것으로 대조한다 — **대표 기간 재생은 연간
    기준이 아니라 그 구간의 실측값과 대조해야 한다**(A-03, provenance 13절).
    """
    targets = TARGETS if targets is None else targets
    reference = REFERENCE_TARGETS if reference is None else reference

    rows = []
    for key, target in targets.items():
        got = result.get(key, np.nan)
        tol = TARGET_TOL.get(key, np.nan)
        rows.append({"구분": "판정", "지표": key, "산출": got, "기준": target,
                     "차이": got - target, "허용": tol,
                     "판정": "OK" if abs(got - target) <= tol else "X"})
    for key, target in reference.items():
        got = result.get(key, np.nan)
        rows.append({"구분": "참고", "지표": key, "산출": got, "기준": target,
                     "차이": got - target, "허용": np.nan, "판정": ""})
    return pd.DataFrame(rows)


def period_targets(calls: pd.DataFrame) -> tuple[dict, dict]:
    """어떤 구간의 실측에서 (판정 기준, 참고 기준)을 뽑는다.

    대표 기간 재생을 채점할 때 쓴다. 연간 기준을 그대로 대면 안 되는 이유는
    **지니가 기간 길이에 편향되기 때문**이다 — 동별 평균의 표본오차가 동 간
    분산에 얹혀, 30일 창에서는 같은 동을 놓고도 연간 0.0887 → 0.1062 로 오른다.
    """
    got = overall_metrics(calls)
    return ({k: got[k] for k in TARGETS}, {k: got[k] for k in REFERENCE_TARGETS})


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    import load

    OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("콜 로딩 중(첫 실행은 1~2분, 이후 캐시)...")
    calls = load.load_calls()
    n_dong = len(calls.groupby(["origin_gu", "origin_dong_canon"], observed=True))
    print(f"콜 {len(calls):,}건 / 출발동 {n_dong}개\n")

    depots = load.load_depots()
    pop = load.load_disabled_population()
    print(f"차고지 {len(depots)}개(차량 {depots['capacity'].sum()}대) / "
          f"등록 장애인 통계 {len(pop)}개 동, {int(pop['n_disabled'].sum()):,}명\n")

    tbl = build_dong_table(calls, depots=depots, population=pop)
    matrix = demand_matrix(calls)

    print(f"동별 표         {tbl.attrs['n_dong']}개 동 × {tbl.shape[1]}개 지표  "
          f"(신뢰 {tbl.attrs['n_reliable']} / 100건 미만 "
          f"{tbl.attrs['n_dong'] - tbl.attrs['n_reliable']})")
    n_grouped = int(tbl["usage_is_grouped"].sum())
    if n_grouped:
        n_groups = tbl[tbl["usage_is_grouped"]].groupby(
            ["gu", "usage_pop"], observed=True).ngroups
        print(f"  이용률 그룹 합산 {n_grouped}개 동 / {n_groups}개 기본명 그룹 "
              f"(콜이 통계보다 잘게 쪼개진 경우 — 같은 그룹은 같은 값)")
    n_miss = tbl.attrs.get("n_pop_unmatched", 0)
    if n_miss:
        print(f"  ⚠ 장애인 통계 미매칭 {n_miss}개 동 → 이용률 NaN: "
              f"{', '.join(tbl.attrs['pop_unmatched'][:6])}"
              f"{' 외' if n_miss > 6 else ''}")
    print(f"수요 매트릭스   {len(matrix):,}행 (동 × 시간대 × 평일/주말)")

    tbl.to_csv(OUT_DIR / "dong_metrics.csv", index=False, encoding="utf-8-sig")
    tbl.to_parquet(OUT_DIR / "dong_metrics.parquet", index=False)
    matrix.to_parquet(OUT_DIR / "dong_demand_matrix.parquet", index=False)

    print(f"\n저장: {OUT_DIR}")

    print("\n미이행 4구간 (서울 합산 — 동 균등가중 평균 아님)")
    brk = cancel_breakdown(calls)
    print(brk.to_string(index=False, formatters={
        "건수": "{:,}".format, "콜 대비": "{:.2%}".format,
        "취소 중 비중": "{:.1%}".format}))

    print("\n검증 (전체값 vs 검증 기준)")
    print(compare_to_target(overall_metrics(calls)).to_string(index=False))
