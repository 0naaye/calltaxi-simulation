"""
test_metrics.py — 지니·동별 집계 검증 (코드 점검)

지니는 값이 조용히 틀려도 그럴듯해 보여서 눈으로 못 잡는다.
해석적으로 답이 정해진 케이스로 고정해 둔다.
"""
import numpy as np
import pandas as pd
import pytest

import load
import metrics


# ─────────────────────────────────────────────────────────────
# 지니계수
# ─────────────────────────────────────────────────────────────

def test_gini_perfectly_equal():
    assert metrics.gini([5, 5, 5, 5]) == pytest.approx(0.0)


def test_gini_perfectly_unequal():
    """한 명이 전부 가지면 (n−1)/n 로 수렴한다."""
    assert metrics.gini([0, 0, 0, 100]) == pytest.approx(0.75)
    assert metrics.gini([0] * 99 + [1]) == pytest.approx(0.99)


def test_gini_known_value():
    """[1..5] 의 지니는 해석적으로 4/15."""
    assert metrics.gini([1, 2, 3, 4, 5]) == pytest.approx(4 / 15)


def test_gini_weights_equal_repetition():
    """가중치는 값 반복과 같은 결과여야 한다."""
    assert metrics.gini([1, 2], [3, 1]) == pytest.approx(metrics.gini([1, 1, 1, 2]))
    assert metrics.gini([10, 20, 30], [2, 2, 2]) == pytest.approx(
        metrics.gini([10, 20, 30]))


def test_gini_scale_invariant():
    """전부 상수배해도 지니는 그대로."""
    v = [3.0, 7.5, 12.0, 40.0]
    assert metrics.gini(v) == pytest.approx(metrics.gini(np.array(v) * 7.3))


def test_gini_ignores_nan_and_zero_weight():
    assert metrics.gini([1, 2, 3, np.nan]) == pytest.approx(metrics.gini([1, 2, 3]))
    assert metrics.gini([1, 2, 3, 999], [1, 1, 1, 0]) == pytest.approx(
        metrics.gini([1, 2, 3]))


def test_gini_rejects_negative():
    with pytest.raises(ValueError):
        metrics.gini([-1, 2, 3])


def test_gini_length_mismatch():
    with pytest.raises(ValueError):
        metrics.gini([1, 2, 3], [1, 1])


def test_gini_empty_is_nan():
    assert np.isnan(metrics.gini([]))
    assert np.isnan(metrics.gini([np.nan, np.nan]))


# ─────────────────────────────────────────────────────────────
# 동별 집계 · A-12
# ─────────────────────────────────────────────────────────────

def _toy_log(counts):
    """{(구, 동): 건수} → 최소 스키마 로그."""
    rows = []
    for (gu, dong), n in counts.items():
        rows += [{"origin_gu": gu, "origin_dong": dong, "wait_min": 30.0}] * n
    return pd.DataFrame(rows)


def test_dong_table_applies_min_calls():
    log = _toy_log({("A구", "큰동"): 150, ("A구", "작은동"): 99})
    tbl = metrics.dong_wait_table(log)
    assert list(tbl["dong"]) == ["큰동"]
    assert tbl.attrs["n_dong_excluded"] == 1
    assert list(tbl.attrs["excluded"]["dong"]) == ["작은동"]


def test_dong_table_boundary_is_inclusive():
    """정확히 100건인 동은 남아야 한다(A-12 는 '100건 미만' 제외)."""
    log = _toy_log({("A구", "경계동"): 100})
    assert len(metrics.dong_wait_table(log)) == 1


def test_dong_table_excludes_cancels_from_count():
    """취소 건(wait=NaN)은 건수에 잡히면 안 된다 — 원본 집계와 정의를 맞춘다."""
    log = pd.DataFrame({
        "origin_gu": ["A구"] * 200,
        "origin_dong": ["동"] * 200,
        "wait_min": [30.0] * 120 + [np.nan] * 80,
    })
    tbl = metrics.dong_wait_table(log, min_calls=100)
    assert tbl.loc[0, "n_calls"] == 120
    assert tbl.loc[0, "mean_wait"] == pytest.approx(30.0)


# ─────────────────────────────────────────────────────────────
# 실측 재현 (느림)
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def calls(data_dir):
    return load.load_calls()


@pytest.mark.slow
def test_dong_count_matches_original_analysis(calls):
    """A-12 적용 시 원본 분석과 같은 430개 동이 남아야 한다."""
    tbl = metrics.dong_wait_table(calls)
    assert len(tbl) == 430
    excluded = set(tbl.attrs["excluded"]["dong"])
    assert excluded == {"창신제3동", "반포본동"}


@pytest.mark.slow
def test_dong_keys_align_with_original_analysis(calls, data_dir):
    """동 키가 원본 산출물과 어긋나지 않아야 한다.

    **값은 더 이상 일치하지 않는다.** 원본 산출물(서울즉시콜_동별_대기격차.csv)은
    임차택시를 포함한 152.7만 건에서 만든 것이고 지금 모집단은 특장차 한정
    122.2만 건이다(A-16). 건수는 동마다 중앙 79.8% 로 줄고 평균대기는 중앙
    +1.25분 오른다. 재현 검사가 아니라 **동명 정규화가 깨지지 않았는지**를 보는
    검사로 남긴다 — 430개 동이 그대로 맞물리고 순위 구조가 유지되는지.
    """
    orig = pd.read_csv(load.DATA_DIR / "서울즉시콜_동별_대기격차.csv", encoding="utf-8-sig")
    orig = orig.rename(columns={"출발구": "gu", "출발동": "dong",
                                "평균": "orig_mean", "건수": "orig_n"})
    m = metrics.dong_wait_table(calls).merge(orig, on=["gu", "dong"], how="inner")
    assert len(m) == 430, "동 키가 원본과 맞물리지 않는다 — 동명 정규화를 의심할 것"
    # 특장차만 남기므로 어느 동도 건수가 늘 수 없다
    assert (m["n_calls"] < m["orig_n"]).all()
    assert (m["n_calls"] / m["orig_n"]).median() == pytest.approx(0.80, abs=0.02)
    assert m["mean_wait"].corr(m["orig_mean"]) > 0.98, "동 간 순위 구조가 뒤집혔다"


@pytest.mark.slow
def test_gini_reproduces_target(calls):
    """지니 두 종이 각각 재현되는지 — 판정은 픽업, 참고는 총 대기다."""
    # 판정 지표 — 매칭이 섞이지 않은 픽업 격차(A-19)
    assert metrics.pickup_gini(calls) == pytest.approx(
        metrics.TARGETS["pickup_gini"], abs=0.005)
    # 참고 지표 — 총 대기 지니. 매칭 지니(0.150)가 섞여 픽업보다 높다
    assert metrics.dong_gini(calls) == pytest.approx(
        metrics.REFERENCE_TARGETS["gini_dong"], abs=0.002)
    assert metrics.dong_gini(calls) > metrics.pickup_gini(calls)
    # 콜수 가중(참고용, 채점 대상 아님)은 다른 값이라는 사실을 고정해 둔다
    assert metrics.dong_gini(calls, weighted=True) == pytest.approx(0.0986, abs=0.002)


@pytest.mark.slow
def test_cancel_ratio_matches_corrected_target(calls):
    """취소율 타깃 15.44% 가 시뮬 입력 모수에서 재현되는지."""
    assert calls["is_canceled"].mean() == pytest.approx(
        metrics.TARGETS["cancel_ratio"], abs=0.002)
    # metrics 쪽 판정도 같은 값이어야 한다 — load 컬럼과 갈라지면 안 된다
    assert (metrics.is_canceled(calls).to_numpy() == calls["is_canceled"].to_numpy()).all()


# ─────────────────────────────────────────────────────────────
# 미이행 4구간 분류
# ─────────────────────────────────────────────────────────────

T0 = pd.Timestamp("2025-03-04 09:00")


def _call(*, assign=None, board=None, cancel=None, dong="가동", gu="A구",
          ride_min=20.0, dist_m=5000.0, weekday=1, period="낮"):
    """콜 한 건. 분 단위 오프셋으로 시각을 만든다(None = 기록 없음)."""
    off = lambda m: None if m is None else T0 + pd.to_timedelta(float(m), unit="m")
    return {
        "received_at": T0, "assigned_at": off(assign),
        "boarded_at": off(board), "alighted_at": off(board + ride_min) if board else None,
        "canceled_at": off(cancel),
        "origin_gu": gu, "origin_dong": dong, "origin_dong_canon": dong,
        "origin_lat": 37.5, "origin_lon": 127.0,
        "wait_min": board, "assign_min": assign, "ride_min": ride_min if board else np.nan,
        "ride_distance_m": dist_m, "weekday": weekday, "period": period,
    }


def _toy_calls(rows):
    df = pd.DataFrame(rows)
    df.insert(0, "call_id", np.arange(len(df)))
    return df


def test_cancel_kind_four_buckets():
    """배차 여부 × 취소까지 걸린 시간으로 4구간이 갈린다."""
    df = _toy_calls([
        _call(assign=5, board=30),          # 정상 승차 → 분류 없음
        _call(cancel=0.5),                  # 배차 전 30초 → 즉시 취소
        _call(cancel=40),                   # 배차 전 40분 → 기다리다 포기
        _call(assign=8, cancel=25),         # 배차 후 취소
        _call(cancel=None),                 # 배차 전인데 취소 시각 없음 → 기타(불명)
    ])
    assert list(pd.Series(metrics.cancel_kind(df)).astype(object)) == [
        np.nan, "immediate", "abandoned", "after_assign", "other"]


def test_cancel_share_sums_to_one():
    """취소 내 구성비 4개의 합 = 100%. 분모는 콜이 아니라 그 동의 전체 취소 건."""
    df = _toy_calls([_call(assign=5, board=30)] * 6 + [_call(cancel=0.5),
                                                       _call(cancel=40),
                                                       _call(assign=8, cancel=25),
                                                       _call(cancel=None)])
    out = metrics.dong_cancel(df).iloc[0]
    assert out["n_canceled"] == 4
    assert sum(out[f"cancel_{k}_share"] for k in metrics.CANCEL_KINDS) == pytest.approx(1.0)
    assert out["cancel_abandoned_share"] == pytest.approx(0.25)   # 취소 4건 중 1건
    assert out["cancel_abandoned_ratio"] == pytest.approx(0.1)    # 콜 10건 중 1건


def test_cancel_share_is_nan_without_cancels():
    """취소가 없으면 0/0 이다. 0으로 채우면 '전부 즉시 취소'와 구분이 안 된다."""
    out = metrics.dong_cancel(_toy_calls([_call(assign=5, board=30)] * 3)).iloc[0]
    assert out["n_canceled"] == 0
    assert out["cancel_ratio"] == 0.0
    assert all(pd.isna(out[f"cancel_{k}_share"]) for k in metrics.CANCEL_KINDS)


def test_cancel_breakdown_totals():
    """서울 합산 표: 콜 대비 합 = 전체 취소율, 취소 중 비중 합 = 1."""
    df = _toy_calls([_call(assign=5, board=30)] * 6 + [_call(cancel=0.5),
                                                       _call(cancel=40),
                                                       _call(assign=8, cancel=25),
                                                       _call(cancel=None)])
    brk = metrics.cancel_breakdown(df)
    assert list(brk["구분"]) == [metrics.CANCEL_KIND_KO[k] for k in metrics.CANCEL_KINDS]
    assert brk["건수"].sum() == 4
    assert brk["콜 대비"].sum() == pytest.approx(0.4)
    assert brk["취소 중 비중"].sum() == pytest.approx(1.0)


def test_cancel_other_is_not_absorbed_into_abandoned():
    """시각 결측 건을 '대기 중 포기'로 몰면 공급 부족 신호가 오염된다 — 분리 유지."""
    df = _toy_calls([_call(cancel=40), _call(cancel=None), _call(cancel=None)])
    out = metrics.dong_cancel(df).iloc[0]
    assert out["cancel_abandoned_ratio"] == pytest.approx(1 / 3)
    assert out["cancel_other_ratio"] == pytest.approx(2 / 3)


def test_cancel_kind_labels_cover_every_bucket():
    """표기용 이름이 구간과 1:1 로 맞는지 — 대시보드에서 빈 칸이 안 나오게."""
    assert set(metrics.CANCEL_KIND_KO) == set(metrics.CANCEL_KINDS)
    assert metrics.CANCEL_KIND_KO["other"] == "기타(불명)"


def test_cancel_kind_boundary_is_immediate():
    """정확히 1분이면 즉시 취소(경계 포함)."""
    df = _toy_calls([_call(cancel=metrics.CANCEL_IMMEDIATE_MIN),
                     _call(cancel=metrics.CANCEL_IMMEDIATE_MIN + 0.01)])
    assert list(pd.Series(metrics.cancel_kind(df)).astype(object)) == [
        "immediate", "abandoned"]


def test_dong_cancel_parts_sum_to_total():
    """4구간의 합이 전체 취소율과 정확히 맞아야 한다(검증 기준과 어긋나면 안 됨)."""
    df = _toy_calls([_call(assign=5, board=30)] * 6 + [_call(cancel=0.5),
                                                       _call(cancel=40),
                                                       _call(assign=8, cancel=25),
                                                       _call(cancel=None)])
    out = metrics.dong_cancel(df).iloc[0]
    parts = sum(out[f"cancel_{k}_ratio"] for k in metrics.CANCEL_KINDS)
    assert parts == pytest.approx(out["cancel_ratio"])
    assert out["cancel_ratio"] == pytest.approx(0.4)      # 10건 중 4건 취소
    assert out["cancel_abandoned_ratio"] == pytest.approx(0.1)


# ─────────────────────────────────────────────────────────────
# 대기 3구간 · 차내
# ─────────────────────────────────────────────────────────────

def test_wait_split_adds_up():
    """매칭(배차−접수) + 픽업(승차−배차) = 총 대기."""
    df = _toy_calls([_call(assign=10, board=35), _call(assign=4, board=24)])
    out = metrics.dong_wait(df).iloc[0]
    assert out["wait_match_mean"] + out["wait_pickup_mean"] == pytest.approx(
        out["wait_total_mean"])
    assert out["wait_total_mean"] == pytest.approx(29.5)


def test_wait_long_ratio_and_denominators():
    """장기대기율 분모는 승차 완료 건. 매칭·픽업은 배차된 건만이라 분모가 더 작다."""
    df = _toy_calls([_call(assign=5, board=90), _call(assign=5, board=10),
                     _call(board=30), _call(cancel=40)])
    out = metrics.dong_wait(df).iloc[0]
    assert out["n_served"] == 3           # 취소 1건 제외
    assert out["n_assigned"] == 2         # 배차 기록 있는 건만
    assert out["long_wait_ratio"] == pytest.approx(1 / 3)


def test_ride_valid_drops_outliers():
    """이상치(운행 0분·거리 0·비현실 속도)는 차내 지표에서 빠진다."""
    df = _toy_calls([
        _call(assign=5, board=30, ride_min=20, dist_m=5000),    # 15km/h 정상
        _call(assign=5, board=30, ride_min=20, dist_m=0),       # 거리 0
        _call(assign=5, board=30, ride_min=200, dist_m=5000),   # 120분 초과
        _call(assign=5, board=30, ride_min=2, dist_m=40000),    # 1200km/h
    ])
    out = metrics.dong_ride(df).iloc[0]
    assert out["n_ride"] == 1
    assert out["ride_kmh_mean"] == pytest.approx(15.0)
    assert out["ride_km_mean"] == pytest.approx(5.0)


# ─────────────────────────────────────────────────────────────
# 수요 · 공급
# ─────────────────────────────────────────────────────────────

def test_demand_breakdown_sums_to_total():
    df = _toy_calls([_call(board=30, period="낮", weekday=1),
                     _call(board=30, period="낮", weekday=5),
                     _call(board=30, period="심야", weekday=6)])
    out = metrics.dong_demand(df).iloc[0]
    assert out["n_calls"] == 3
    assert out["n_calls_day"] == 2 and out["n_calls_night"] == 1
    assert out["n_calls_weekday"] == 1 and out["n_calls_weekend"] == 2


def test_demand_matrix_preserves_every_call():
    df = _toy_calls([_call(board=30, period="낮", weekday=1),
                     _call(board=30, period="낮", weekday=1),
                     _call(board=30, period="저녁", weekday=6, dong="나동")])
    mx = metrics.demand_matrix(df)
    assert mx["n_calls"].sum() == len(df)
    assert set(mx["dong_canon"]) == {"가동", "나동"}


def test_dong_supply_counts_only_within_radius():
    """반경 3km 안 차고지의 배속 차량만 더한다. 위도 0.01도 ≈ 1.11km."""
    dong_xy = pd.DataFrame([{"gu": "A구", "dong_canon": "가동", "lat": 37.5, "lon": 127.0}])
    depots = pd.DataFrame([
        {"lat": 37.5, "lon": 127.0, "capacity": 10},      # 0km
        {"lat": 37.52, "lon": 127.0, "capacity": 5},      # ≈2.2km
        {"lat": 37.55, "lon": 127.0, "capacity": 99},     # ≈5.6km — 제외
    ])
    out = metrics.dong_supply(dong_xy, depots).iloc[0]
    assert out["vehicles_3km"] == 15
    assert out["n_depots_3km"] == 2


# ─────────────────────────────────────────────────────────────
# 등록 장애인 매칭
# ─────────────────────────────────────────────────────────────

def _pop(rows):
    df = pd.DataFrame(rows, columns=["gu", "dong_canon", "n_disabled"])
    df["dong_base"] = df["dong_canon"].map(load._base_dong)
    return df


def test_population_exact_match():
    keys = pd.DataFrame([{"gu": "A구", "dong_canon": "사직동"}])
    out = metrics.dong_population(keys, _pop([("A구", "사직동", 250)])).iloc[0]
    assert out["n_disabled"] == 250 and out["pop_match"] == "canon"


def test_population_base_fallback_sums_split_dong():
    """콜이 뭉뚱그린 '상일동' ↔ 통계가 쪼갠 '상일1·2동' → 합쳐서 붙인다."""
    keys = pd.DataFrame([{"gu": "강동구", "dong_canon": "상일동"}])
    pop = _pop([("강동구", "상일1동", 700), ("강동구", "상일2동", 300)])
    out = metrics.dong_population(keys, pop).iloc[0]
    assert out["n_disabled"] == 1000 and out["pop_match"] == "base"


def test_population_contested_base_stays_unmatched():
    """콜이 더 잘게 쪼갠 경우(신당1~3동 ↔ 통계 신당동)는 나눌 근거가 없어 붙이지 않는다.

    동 하나에 귀속되는 n_disabled 는 비워두고, 그룹 인구는 pop_group 으로 따로 낸다.
    """
    keys = pd.DataFrame([{"gu": "중구", "dong_canon": f"신당{i}동"} for i in (1, 2, 3)])
    out = metrics.dong_population(keys, _pop([("중구", "신당동", 900)]))
    assert out["n_disabled"].isna().all()
    assert out["pop_match"].isna().all()
    assert (out["pop_group"] == 900).all()


def test_population_group_pop_includes_already_matched_rows():
    """그룹 인구는 잔여분이 아니라 그룹 전체다.

    답십리3·4동을 살리려면 이미 정확히 붙은 답십리1·2동의 인구까지 분모에 들어가야
    한다. 대신 이용률 분자도 4개 동을 다 더한다(build_dong_table).
    """
    keys = pd.DataFrame([{"gu": "동대문구", "dong_canon": f"답십리{i}동"}
                         for i in (1, 2, 3, 4)])
    pop = _pop([("동대문구", "답십리1동", 1000), ("동대문구", "답십리2동", 400)])
    out = metrics.dong_population(keys, pop).set_index("dong_canon")
    assert out.loc["답십리1동", "n_disabled"] == 1000    # 귀속 인구는 그대로
    assert (out["pop_group"] == 1400).all()             # 그룹 인구는 전원 동일


def test_population_group_pop_empty_when_all_resolved():
    """전부 붙은 그룹에는 그룹 분모를 만들지 않는다(상일동 ↔ 상일1·2동)."""
    keys = pd.DataFrame([{"gu": "강동구", "dong_canon": "상일동"}])
    pop = _pop([("강동구", "상일1동", 700), ("강동구", "상일2동", 300)])
    assert metrics.dong_population(keys, pop)["pop_group"].isna().all()


def test_population_base_fallback_skips_already_matched():
    """기본명 그룹에서 이미 정확히 붙은 통계 행은 빼고 더한다(이중 계상 방지)."""
    keys = pd.DataFrame([{"gu": "노원구", "dong_canon": "공릉13동"},
                         {"gu": "노원구", "dong_canon": "공릉2동"}])
    pop = _pop([("노원구", "공릉1동", 400), ("노원구", "공릉2동", 600)])
    out = metrics.dong_population(keys, pop).set_index("dong_canon")
    assert out.loc["공릉2동", "n_disabled"] == 600
    assert out.loc["공릉13동", "n_disabled"] == 400      # 공릉2동을 다시 더하지 않는다


def test_population_all_nan_group_is_not_zero():
    """통계값이 '-' 뿐인 그룹을 0명으로 만들면 이용률이 무한대로 튄다."""
    keys = pd.DataFrame([{"gu": "서초구", "dong_canon": "반포본동"}])
    out = metrics.dong_population(keys, _pop([("서초구", "반포본동", np.nan)])).iloc[0]
    assert pd.isna(out["n_disabled"])


# ─────────────────────────────────────────────────────────────
# 비교 (구내 / 서울)
# ─────────────────────────────────────────────────────────────

def _cmp_table(reliable=True):
    return pd.DataFrame({
        "gu": ["A구", "A구", "B구"],
        "dong_canon": ["가", "나", "다"],
        "wait_total_mean": [10.0, 30.0, 40.0],
        "is_reliable": [True, True, reliable],
    })


def test_add_comparisons_ratios():
    out = metrics.add_comparisons(_cmp_table(), ["wait_total_mean"])
    assert out["wait_total_mean_gu_mean"].tolist() == [20.0, 20.0, 40.0]
    assert out["wait_total_mean_vs_gu"].tolist() == [0.5, 1.5, 1.0]
    # 서울 평균은 동 균등가중 — (10+30+40)/3
    assert out["wait_total_mean_vs_seoul"][0] == pytest.approx(10 / (80 / 3))


def test_add_comparisons_baseline_excludes_unreliable():
    """표본이 얇은 동은 기준선에서 빠지지만 자기 비율은 계산된다."""
    out = metrics.add_comparisons(_cmp_table(reliable=False), ["wait_total_mean"])
    assert out["wait_total_mean_seoul_mean"][0] == pytest.approx(20.0)   # 40 제외
    assert out["wait_total_mean_vs_seoul"][2] == pytest.approx(2.0)


def test_add_comparisons_rejects_unknown_column():
    """오타를 조용히 넘기면 비교 컬럼이 통째로 빠진 표가 나온다."""
    with pytest.raises(KeyError):
        metrics.add_comparisons(_cmp_table(), ["cancel_abandon_ratio"])


def test_compare_cols_default_exists_in_built_table():
    """기본 비교 대상 이름이 실제 산출 컬럼과 맞는지 — 오타 방지."""
    df = _toy_calls([_call(assign=5, board=30)] * 3)
    tbl = metrics.build_dong_table(df)
    known = set(tbl.columns) | {"usage_rate", "vehicles_3km"}   # 선택 입력분
    assert set(metrics.COMPARE_COLS) <= known


# ─────────────────────────────────────────────────────────────
# 차량-일 집계 (조 편성 실측 재구성의 입력 — A-09)
# ─────────────────────────────────────────────────────────────

def _trips(rows):
    """(차량, 승차 오프셋 분, 운행 분) → 운행 기록."""
    out = []
    for vid, start, dur in rows:
        b = T0 + pd.to_timedelta(float(start), unit="m")
        out.append({"vehicle_id": vid, "service_date": b.normalize(),
                    "boarded_at": b, "alighted_at": b + pd.to_timedelta(float(dur), unit="m"),
                    "ride_min": float(dur), "km": dur / 3})
    return pd.DataFrame(out)


def test_vehicle_day_table_span_and_counts():
    """첫 승차~마지막 하차가 span 이다 — 9시 승차, 13시 하차면 4시간.

    조 편성이 이 span 에서 근무 길이를 역추정하므로(A-09) 정의가 바뀌면 안 된다.
    """
    vd = metrics.vehicle_day_table(_trips([(1, 0, 60), (1, 180, 60)]))
    assert len(vd) == 1
    r = vd.iloc[0]
    assert r["trips"] == 2
    assert r["occupied_min"] == pytest.approx(120.0)
    assert r["span_h"] == pytest.approx(4.0)


def test_vehicle_day_table_splits_by_vehicle_and_date():
    """차량-일이 키다 — 차량이 다르면 다른 행."""
    vd = metrics.vehicle_day_table(
        pd.concat([_trips([(1, 0, 60), (1, 180, 60)]), _trips([(2, 0, 10)])]))
    assert len(vd) == 2
    assert set(vd["vehicle_id"]) == {1, 2}
    assert vd.loc[vd["vehicle_id"] == 2, "span_h"].iloc[0] == pytest.approx(10 / 60)


# ─────────────────────────────────────────────────────────────
# 동별 표 조립 (느림)
# ─────────────────────────────────────────────────────────────

def test_build_dong_table_without_optional_inputs():
    """거점·인구 없이도 돌아야 한다(시뮬 로그처럼 콜만 있는 입력)."""
    df = _toy_calls([_call(assign=5, board=30)] * 3 + [_call(cancel=40, dong="나동")])
    tbl = metrics.build_dong_table(df)
    assert set(tbl["dong_canon"]) == {"가동", "나동"}
    assert "vehicles_3km" not in tbl.columns and "usage_rate" not in tbl.columns
    assert "wait_total_mean_vs_gu" in tbl.columns


def test_usage_rate_groups_contested_base():
    """콜 신당1~3동 ↔ 통계 신당동 — 콜을 그룹으로 합쳐 이용률을 낸다.

    분자·분모가 같이 그룹으로 올라가므로 세 동은 같은 값을 갖는다.
    """
    rows = ([_call(assign=5, board=30, gu="중구", dong="신당1동")] * 60
            + [_call(assign=5, board=30, gu="중구", dong="신당2동")] * 30
            + [_call(assign=5, board=30, gu="중구", dong="신당3동")] * 10)
    tbl = metrics.build_dong_table(_toy_calls(rows),
                                   population=_pop([("중구", "신당동", 200)]))
    tbl = tbl.set_index("dong_canon")
    assert tbl["usage_is_grouped"].all()
    assert (tbl["usage_pop"] == 200).all()
    assert tbl["usage_rate"].tolist() == pytest.approx([100 / 200] * 3)   # 콜 100건 ÷ 200명
    assert tbl["n_disabled"].isna().all()      # 동 하나에 귀속할 근거는 여전히 없다


def test_usage_rate_not_grouped_when_matched():
    """정확히 붙는 동은 자기 콜 ÷ 자기 인구 그대로다."""
    rows = ([_call(assign=5, board=30, gu="A구", dong="사직동")] * 40
            + [_call(assign=5, board=30, gu="A구", dong="삼청동")] * 10)
    tbl = metrics.build_dong_table(
        _toy_calls(rows),
        population=_pop([("A구", "사직동", 100), ("A구", "삼청동", 50)])
    ).set_index("dong_canon")
    assert not tbl["usage_is_grouped"].any()
    assert tbl.loc["사직동", "usage_rate"] == pytest.approx(0.4)
    assert tbl.loc["삼청동", "usage_rate"] == pytest.approx(0.2)


def test_build_dong_table_keeps_thin_dong_with_flag():
    """100건 미만 동도 표에는 남는다 — 지우면 '데이터 없음'과 구분이 안 된다."""
    df = _toy_calls([_call(assign=5, board=30)] * 100 + [_call(assign=5, board=30,
                                                               dong="작은동")])
    tbl = metrics.build_dong_table(df).set_index("dong_canon")
    assert len(tbl) == 2
    assert bool(tbl.loc["가동", "is_reliable"]) and not bool(tbl.loc["작은동", "is_reliable"])


@pytest.mark.slow
def test_build_dong_table_reproduces_dong_universe(calls, data_dir):
    """실측으로 조립했을 때 동 수·신뢰 동 수가 A-12 와 맞는지."""
    tbl = metrics.build_dong_table(calls, depots=load.load_depots(),
                                   population=load.load_disabled_population())
    assert tbl.attrs["n_dong"] == 432
    assert tbl.attrs["n_reliable"] == 430          # 지니 모집단과 동일
    # 4구간 취소 분해가 전체 취소율과 어긋나면 검증 기준 13.3%가 깨진다
    parts = tbl[[f"cancel_{k}_ratio" for k in metrics.CANCEL_KINDS]].sum(axis=1)
    assert (parts - tbl["cancel_ratio"]).abs().max() < 1e-9
    # 취소 내 구성비는 취소가 있는 동에서 반드시 100%
    shares = tbl[[f"cancel_{k}_share" for k in metrics.CANCEL_KINDS]].sum(axis=1)
    assert (shares[tbl["n_canceled"] > 0] - 1).abs().max() < 1e-9
    # 인구를 이중 계상하면 붙은 합이 통계 총합을 넘는다.
    # usage_pop 은 그룹 안에서 중복되므로 여기 쓰면 안 된다 — n_disabled 로 잰다.
    pop_total = load.load_disabled_population()["n_disabled"].sum()
    assert tbl["n_disabled"].sum() <= pop_total
    # 그룹 합산으로 이용률이 살아나 NaN 은 명륜3가동·반포본동 둘만 남는다
    assert sorted(tbl.loc[tbl["usage_rate"].isna(), "dong_canon"]) == ["명륜3가동", "반포본동"]
    grp = tbl[tbl["usage_is_grouped"]]
    assert len(grp) == 24
    assert all(g["usage_rate"].nunique() == 1
               for _, g in grp.groupby(["gu", "usage_pop"], observed=True))


@pytest.mark.slow
def test_overall_metrics_pass_verification(calls):
    """판정 대상(픽업·취소·지니)이 전부 통과하고, 참고 행은 판정되지 않아야 한다."""
    cmp = metrics.compare_to_target(metrics.overall_metrics(calls))
    judged = cmp[cmp["구분"] == "판정"]
    assert len(judged) == len(metrics.TARGETS)
    assert (judged["판정"] == "OK").all(), judged.to_string(index=False)
    # 참고 행은 판정을 매기지 않는다 — A-19 의 미재현이 실점이 되면 안 된다
    ref = cmp[cmp["구분"] == "참고"]
    assert len(ref) == len(metrics.REFERENCE_TARGETS)
    assert (ref["판정"] == "").all()
    # 참고값도 실측과는 맞아야 한다(정의가 어긋나면 여기서 걸린다)
    assert (ref["차이"].abs() < 0.5).all()


def test_judged_gini_is_pickup_not_total_wait():
    """판정 지니는 **픽업**이다. 총 대기 지니는 참고로만 둔다.

    둘은 값이 비슷해(0.070 vs 0.092) 문서에서 뒤바뀌어도 눈에 안 띈다. 실제로
    한 번 뒤바뀐 적이 있어(calibration 표가 총 대기 값을 판정 기준으로 실었다)
    여기서 고정한다. 총 대기 지니로 채점하면 재현하지 않기로 한 매칭 구간이
    형평 점수로 넘어온다(A-19).
    """
    assert "pickup_gini" in metrics.TARGETS
    assert "gini_dong" not in metrics.TARGETS
    assert "gini_dong" in metrics.REFERENCE_TARGETS
    assert "pickup_gini" not in metrics.REFERENCE_TARGETS
    # 두 값을 뒤바꿔 적으면 걸리게 — 픽업 쪽이 더 낮다
    assert metrics.TARGETS["pickup_gini"] < metrics.REFERENCE_TARGETS["gini_dong"]
    assert metrics.TARGETS["pickup_gini"] == pytest.approx(0.070, abs=0.001)
    assert metrics.REFERENCE_TARGETS["gini_dong"] == pytest.approx(0.092, abs=0.001)
    # 판정 지표는 전부 허용오차가 있어야 한다
    assert set(metrics.TARGET_TOL) == set(metrics.TARGETS)


def test_period_targets_round_trip(calls):
    """구간 실측에서 뽑은 기준으로 그 구간을 채점하면 차이가 0이다."""
    tg, ref = metrics.period_targets(calls)
    cmp = metrics.compare_to_target(metrics.overall_metrics(calls),
                                    targets=tg, reference=ref)
    assert (cmp["차이"].abs() < 1e-9).all()
    assert (cmp[cmp["구분"] == "판정"]["판정"] == "OK").all()
