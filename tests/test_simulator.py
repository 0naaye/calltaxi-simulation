# -*- coding: utf-8 -*-
"""시뮬 엔진 점검 — **설계대로 도는가**만 본다.

현실 재현(대기 40.8분 · 취소 15.44%)은 여기서 판정하지 않는다. 배차 후 취소와
유휴≠가용이 아직 없어 재현이 될 수 없고(simulator.KNOWN_GAPS), 그 판정은 채점
함수(metrics.compare_to_target)가 따로 한다.

여기서 막는 것은 **조용히 틀리는 것**들이다 — 차량이 동시에 두 콜을 처리하거나,
배차된 콜이 승차도 취소도 아닌 채 사라지거나, 대기시간에 음수가 섞이는 경우.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import load          # noqa: E402
import metrics       # noqa: E402
import simulator as S  # noqa: E402
import travel_time as T  # noqa: E402


# ─────────────────────────────────────────────────────────────
# 순수 함수 — 원본 없이 돈다
# ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n_total", [44, 100, 374, 574, 701])
def test_allocate_fleet_sums_exactly(n_total):
    depots = pd.DataFrame({"capacity": [10, 12, 15, 46, 45, 34, 30] + [10] * 37})
    got = S.allocate_fleet(depots, n_total)
    assert got.sum() == n_total, "배분 합이 목표 대수와 어긋난다"
    assert (got >= 1).all(), "정원이 있는 거점은 최소 1대를 보장해야 한다"


def test_allocate_fleet_follows_capacity_ratio():
    """큰 거점이 더 많이 받아야 한다 — 배속 정원이 배분 비율이다."""
    depots = pd.DataFrame({"capacity": [10, 20, 40]})
    got = S.allocate_fleet(depots, 700)
    assert got[0] < got[1] < got[2]
    assert got[2] == pytest.approx(700 * 40 / 70, abs=1)


def test_allocate_fleet_rejects_impossible_fleet():
    depots = pd.DataFrame({"capacity": [10] * 44})
    with pytest.raises(ValueError):
        S.allocate_fleet(depots, 10)          # 거점 44곳에 10대는 최소 1대 불가


def test_build_shifts_within_bounds():
    start, end = S.build_shifts(np.arange(5000), 42)
    assert (end > start).all(), "근무 종료가 출차보다 이를 수 없다"
    assert (end - start >= 60).all()
    # 출차 시각이 실측 봉우리(07·08·12시)에 몰려야 한다.
    # ±15분 흔들림이 있으므로 가장 가까운 정시로 되돌려 센다 — 07:00 − 12분은
    # 6시대로 떨어지지만 1조다(실제 출차는 첫 승차보다 이르다, A-09 한계).
    nominal = np.round(start / 60).astype(int)
    top = pd.Series(nominal).value_counts(normalize=True)
    assert set(top.head(3).index) == {7, 8, 12}
    assert top[7] == pytest.approx(S.SHIFT_TABLE[7], abs=0.02)


def test_build_shifts_is_stable_under_fleet_growth():
    """A-10 증차 — 가동 대수가 574 → 582 로 늘어도 앞 574 슬롯의 조 편성은 그대로.

    이게 깨지면 증차 순간 전 차량의 편성이 달라져 before/after 차이에 난수가
    섞인다. 실측한 크기가 작지 않다 — 순서 기반이던 시절 동별 픽업이 시드만
    바꿔도 표준편차 0.465분 움직였고, 후보 하나의 효과는 −0.06분이다.
    """
    a_start, a_end = S.build_shifts(np.arange(574), 42)
    b_start, b_end = S.build_shifts(np.arange(582), 42)
    assert np.array_equal(a_start, b_start[:574])
    assert np.array_equal(a_end, b_end[:574])
    # 날짜가 다르면 편성도 달라야 한다(슬롯 ID 에 날짜가 섞여 있다)
    d1 = S.build_shifts(np.arange(574), 42)[0]
    d2 = S.build_shifts(S.SLOTS_PER_DAY + np.arange(574), 42)[0]
    assert not np.array_equal(d1, d2)


def test_patience_sample_is_monotone_in_u():
    """역CDF — u 가 크면 인내심이 짧다(생존율이 먼저 u 아래로 떨어진다)."""
    p = S.Patience()
    rng = np.random.default_rng(0)
    v = p.sample(rng, 20_000)
    assert (v >= 0).all() and np.isfinite(v).all()
    assert v.max() <= p.max_min
    # 곡선의 중앙(186분) 근처에서 절반이 갈려야 한다
    assert 0.4 < (v <= 186.0).mean() < 0.6


def test_uniform_by_id_is_stable_and_uniform():
    """콜 ID 스트림 — 같은 (id, seed, stream)이면 항상 같은 값. 부분집합에도 안전."""
    ids = np.arange(50_000)
    a = S.uniform_by_id(ids, 42, S.STREAM_PATIENCE)
    b = S.uniform_by_id(ids, 42, S.STREAM_PATIENCE)
    assert np.array_equal(a, b)

    # 부분집합을 따로 뽑아도 같은 값이 나와야 한다(순서 의존 금지)
    pick = np.array([7, 999, 31_415])
    assert np.allclose(S.uniform_by_id(pick, 42, S.STREAM_PATIENCE), a[pick])

    # 스트림·시드가 다르면 값이 갈린다
    assert not np.allclose(a, S.uniform_by_id(ids, 42, S.STREAM_IMMEDIATE))
    assert not np.allclose(a, S.uniform_by_id(ids, 43, S.STREAM_PATIENCE))

    assert a.min() >= 0.0 and a.max() < 1.0
    assert abs(a.mean() - 0.5) < 0.01
    # 십분위가 고르게 차야 한다
    counts = np.histogram(a, bins=10, range=(0, 1))[0]
    assert counts.min() > len(ids) / 10 * 0.9


def test_idle_break_length_comes_from_data():
    """휴게 길이는 자유 파라미터가 아니라 실측 초과분에서 나온다."""
    gaps = np.concatenate([np.full(5000, 5.0), np.linspace(60, 300, 5000)])
    ib = S.IdleBreak(gaps, 60.0)
    rng = np.random.default_rng(0)
    v = np.array([ib.sample(rng) for _ in range(2000)])
    assert v.min() >= 0 and v.max() <= 240.0 + 1e-6, "초과분 범위를 벗어났다"
    assert 0 < ib.share_of_idle_minutes < 1
    with pytest.raises(ValueError):
        S.IdleBreak(gaps, 10_000.0)          # 꼬리 표본이 없으면 만들지 않는다


def test_minmax_and_rank_helpers():
    assert S._minmax(np.array([5.0])).tolist() == [1.0]
    assert S._minmax(np.array([2.0, 2.0])).tolist() == [1.0, 1.0]
    assert S._minmax(np.array([0.0, 5.0, 10.0])).tolist() == [0.0, 0.5, 1.0]
    assert S._rank_score(np.array([3.0, 1.0, 2.0])).tolist() == [1.0, 0.0, 0.5]


# ─────────────────────────────────────────────────────────────
# 엔진 — 작은 합성 입력
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def matrix():
    return S.TravelMatrix(T.TravelTime.build())


def _synthetic_calls(n, dongs, *, start="2025-03-03 08:00:00", gap_min=3.0):
    """최소 컬럼만 갖춘 콜 프레임. load.load_calls 의 부분집합이다."""
    t = pd.Timestamp(start) + pd.to_timedelta(np.arange(n) * gap_min, unit="m")
    return pd.DataFrame({
        "call_id": np.arange(n, dtype=np.int32),
        "received_at": t, "scheduled_at": t,
        "assigned_at": pd.NaT, "boarded_at": pd.NaT,
        "alighted_at": pd.NaT, "canceled_at": pd.NaT,
        "origin_gu": "종로구", "origin_dong": dongs[0],
        "dest_gu": "종로구", "dest_dong": dongs[1],
        "origin_dong_canon": dongs[0], "dest_dong_canon": dongs[1],
        "ride_distance_m": 5000.0, "fare": 1500.0,
        "wait_min": np.nan, "assign_min": np.nan, "ride_min": np.nan,
        "is_unsettled": False, "is_canceled": True,
        "weekday": t.weekday, "hour": t.hour,
    })


def _depots(n=1):
    return pd.DataFrame({
        "depot_id": np.arange(1, n + 1), "name": [f"d{i}" for i in range(n)],
        "gu": "종로구", "lat": [37.5735] * n, "lon": [126.9788] * n,
        "capacity": [10] * n})


def test_empty_calls_does_not_crash(matrix):
    """콜 0건 — 빈 로그가 나오고 예외가 없어야 한다."""
    empty = _synthetic_calls(0, ("사직동", "혜화동"))
    log = S.run_placement(None, empty, _depots(), matrix)
    assert len(log) == 0
    assert log.attrs["stats"]["served"] == 0


def test_single_vehicle_serves_sequentially(matrix, monkeypatch):
    """차량 1대 — 동시에 두 콜을 처리하지 않는다(운행 구간이 겹치지 않는다)."""
    monkeypatch.setattr(S, "FLEET_WEEKDAY", 1)
    monkeypatch.setattr(S, "FLEET_WEEKEND", 1)
    monkeypatch.setattr(S, "IMMEDIATE_CANCEL_P", 0.0)

    calls = _synthetic_calls(40, ("사직동", "혜화동"), gap_min=5.0)
    log = S.run_placement(None, calls, _depots(), matrix)

    served = log[log["boarded_at"].notna()].sort_values("boarded_at")
    assert len(served) >= 5, "1대라도 몇 건은 태워야 한다"
    # 같은 차량이므로 [승차, 하차] 구간이 겹치면 안 된다
    assert (served["boarded_at"].to_numpy()[1:]
            >= served["alighted_at"].to_numpy()[:-1]).all()
    assert served["vehicle_id"].nunique() == 1


def test_every_call_ends_boarded_or_canceled(matrix):
    """콜은 반드시 승차·취소 중 하나로 끝난다(사라지지 않는다)."""
    calls = _synthetic_calls(300, ("사직동", "혜화동"), gap_min=2.0)
    log = S.run_placement(None, calls, _depots(3), matrix, after_assign_cancel_p=0.0)

    boarded = log["boarded_at"].notna()
    canceled = log["canceled_at"].notna()
    assert not (boarded & canceled).any(), "승차와 취소가 동시에 잡힌 콜"
    unresolved = int((~boarded & ~canceled).sum())
    assert unresolved == log.attrs["stats"]["unresolved"]
    assert unresolved == 0, "warmdown 안에 전부 해소돼야 한다"

    # 배차 후 취소를 끄면 배차 = 승차다. 켜면 그만큼만 갈린다(별도 검사).
    assert (log["assigned_at"].notna() == boarded).all()


def test_time_columns_are_ordered_and_finite(matrix):
    calls = _synthetic_calls(300, ("사직동", "혜화동"), gap_min=2.0)
    log = S.run_placement(None, calls, _depots(3), matrix)

    w = log["wait_min"].dropna()
    assert (w >= 0).all(), "대기시간에 음수가 있다"
    assert np.isfinite(w).all(), "대기시간에 NaN·inf 가 섞였다"
    r = log["ride_min"].dropna()
    assert (r > 0).all() and np.isfinite(r).all()

    ok = log[log["boarded_at"].notna()]
    assert (ok["received_at"] <= ok["assigned_at"]).all()
    assert (ok["assigned_at"] <= ok["boarded_at"]).all()
    assert (ok["boarded_at"] <= ok["alighted_at"]).all()

    cx = log[log["canceled_at"].notna()]
    assert (cx["received_at"] <= cx["canceled_at"]).all()


def test_event_log_order_is_consistent(matrix):
    """이벤트 순서 — 접수 → 배차 → 승차 → 하차 가 콜마다 그 순서로 찍힌다."""
    calls = _synthetic_calls(120, ("사직동", "혜화동"), gap_min=3.0)
    sim = S.CallTaxiSim(calls, _depots(2), matrix)
    sim.run()

    seq = {}
    for t, kind, row, _vid in sim.events:
        if row >= 0:
            seq.setdefault(row, []).append((t, kind))
    checked = 0
    for row, ev in seq.items():
        kinds = [k for _, k in ev]
        times = [t for t, _ in ev]
        assert times == sorted(times), f"콜 {row} 이벤트 시각이 역전됐다"
        if kinds[0] == "arrive" and len(kinds) > 1:
            assert kinds[:4] in (["arrive", "assign", "board", "alight"],
                                 ["arrive", "assign", "after_assign_cancel"],
                                 ["arrive", "abandon"]), f"콜 {row}: {kinds}"
            checked += 1
    assert checked > 50


def test_placement_adds_ten_vehicles(matrix):
    """A-10 — 신규 거점은 10대 증차이고 기존 배속은 그대로다."""
    base = _depots(2)
    out = S.resolve_placement({"name": "신규", "lat": 37.49, "lon": 127.02}, base)
    assert len(out) == len(base) + 1
    assert out["capacity"].iloc[-1] == S.NEW_DEPOT_VEHICLES
    assert out["capacity"].iloc[:-1].tolist() == base["capacity"].tolist()

    # **배속은 인자로 못 바꾼다.** 증차량이 상수라야 후보별 차이가 위치에서만
    # 나온다는 해석이 성립한다(A-10). 넘겨받은 값이 있어도 무시해야 한다.
    forced = S.resolve_placement(
        {"name": "신규", "lat": 37.49, "lon": 127.02, "capacity": 30}, base)
    assert forced["capacity"].iloc[-1] == S.NEW_DEPOT_VEHICLES


def test_reservation_deduction_holds_vehicles_back(matrix, monkeypatch):
    """A-11 — 차감 대수만큼은 배차되지 않고 남는다."""
    monkeypatch.setattr(S, "FLEET_WEEKDAY", 4)
    monkeypatch.setattr(S, "IMMEDIATE_CANCEL_P", 0.0)
    calls = _synthetic_calls(60, ("사직동", "혜화동"), gap_min=1.0)

    free = S.run_placement(None, calls, _depots(), matrix)
    held = pd.DataFrame({"hour": range(24), "weekday": [3.0] * 24,
                         "weekend": [3.0] * 24})
    gated = S.run_placement(None, calls, _depots(), matrix, reservation=held)
    assert gated["boarded_at"].notna().sum() < free["boarded_at"].notna().sum()


def test_patience_is_fixed_per_call_across_placements(matrix):
    """공통난수 — 배치안이 달라도 같은 콜은 같은 인내심을 받는다.

    이게 깨지면 후보 간 차이에 난수 흔들림이 섞여, 위치 효과를 잴 수 없다.
    """
    calls = _synthetic_calls(200, ("사직동", "혜화동"), gap_min=2.0)
    a = S.CallTaxiSim(calls, _depots(1), matrix)
    a._prepare_calls()
    b = S.CallTaxiSim(calls, S.resolve_placement(
        {"name": "신규", "lat": 37.49, "lon": 127.02}, _depots(3)), matrix)
    b._prepare_calls()
    assert np.array_equal(a._patience_min, b._patience_min)
    assert np.array_equal(a._immediate, b._immediate)
    assert np.array_equal(a._after_assign_cancel, b._after_assign_cancel)

    # 콜 부분집합을 넣어도 그 콜의 값은 그대로여야 한다
    half = calls.iloc[100:].reset_index(drop=True)
    c = S.CallTaxiSim(half, _depots(1), matrix)
    c._prepare_calls()
    assert np.array_equal(c._patience_min, a._patience_min[100:])


def test_after_assign_cancel_raises_cancel_ratio(matrix):
    """A-16 이후 취소 4구간 — 배차 후 취소가 켜지면 그만큼만 늘어야 한다."""
    calls = _synthetic_calls(600, ("사직동", "혜화동"), gap_min=1.5)
    off = S.run_placement(None, calls, _depots(3), matrix, after_assign_cancel_p=0.0)
    on = S.run_placement(None, calls, _depots(3), matrix, after_assign_cancel_p=0.20)

    assert pd.Series(metrics.cancel_kind(off)).eq("after_assign").sum() == 0
    n_after = pd.Series(metrics.cancel_kind(on)).eq("after_assign").sum()
    assert n_after > 0
    assert on["is_canceled"].mean() > off["is_canceled"].mean()
    # 배차 후 취소된 콜은 승차 기록이 없고 배차 기록은 있다
    hit = on[on["assigned_at"].notna() & on["boarded_at"].isna()]
    assert len(hit) == n_after
    assert hit["canceled_at"].notna().all()


def test_idle_break_reduces_supply(matrix):
    """θ 를 켜면 공급이 줄어 대기가 늘어야 한다 — 방향만 본다."""
    calls = _synthetic_calls(400, ("사직동", "혜화동"), gap_min=3.0)
    gaps = load.load_idle_gaps()
    free = S.run_placement(None, calls, _depots(2), matrix, after_assign_cancel_p=0.0)
    held = S.run_placement(None, calls, _depots(2), matrix, after_assign_cancel_p=0.0,
                           idle_break=S.IdleBreak(gaps, 20.0))
    assert held.attrs["stats"]["breaks"] > 0
    assert held["wait_min"].dropna().mean() >= free["wait_min"].dropna().mean()


def test_log_schema_matches_metrics_contract(matrix):
    """시뮬 로그가 metrics 함수를 그대로 통과해야 한다 — 이 모듈의 계약."""
    calls = _synthetic_calls(400, ("사직동", "혜화동"), gap_min=2.0)
    log = S.run_placement(None, calls, _depots(3), matrix)

    res = metrics.overall_metrics(log)
    assert set(res) == set(metrics.TARGETS) | set(metrics.REFERENCE_TARGETS)
    assert np.isfinite(res["pickup_mean"]) and res["pickup_mean"] > 0
    assert np.isfinite(res["mean_wait"]) and res["mean_wait"] >= 0
    assert 0.0 <= res["cancel_ratio"] <= 1.0

    kind = pd.Series(metrics.cancel_kind(log))
    assert kind.notna().sum() == int(log["is_canceled"].sum())
    brk = metrics.cancel_breakdown(log)
    assert brk["건수"].sum() == int(log["is_canceled"].sum())


@pytest.mark.slow
def test_real_calls_run_and_stay_sane():
    """실측 하루치 — 실행되고 시간 컬럼이 성립하는지까지만 본다."""
    calls = load.load_calls()
    day = S.slice_period(calls, days=1)
    mx = S.TravelMatrix(T.TravelTime.build(),
                        set(calls["origin_dong_canon"].astype(str))
                        | set(calls["dest_dong_canon"].astype(str)))
    log = S.run_placement(None, day, load.load_depots(), mx,
                          reservation=load.load_reservation_occupancy())

    assert len(log) == len(day)
    assert list(log.columns)[:len(day.columns)] == list(day.columns), \
        "실측 컬럼 구조가 유지돼야 한다"
    w = log["wait_min"].dropna()
    assert (w >= 0).all() and np.isfinite(w).all()
    assert log.attrs["stats"]["unresolved"] == 0
    # 차량 하나가 같은 시각에 두 운행을 하지 않는다
    served = log[log["boarded_at"].notna()]
    for _vid, g in served.groupby("vehicle_id"):
        g = g.sort_values("boarded_at")
        assert (g["boarded_at"].to_numpy()[1:]
                >= g["alighted_at"].to_numpy()[:-1]).all()


# ─────────────────────────────────────────────────────────────
# 민감도 손잡이 — 기본값이 기존 동작을 바꾸지 않는가
# ─────────────────────────────────────────────────────────────

def test_predawn_radius_defaults_to_night(matrix):
    """평일 심야 반경을 따로 뗐어도 **기본값은 야간과 같아야 한다**(A-17).

    갈라 둔 것은 민감도로 갈아끼우기 위해서지 동작을 바꾸려는 게 아니다.
    갈아끼우면 **평일 심야만** 움직이고 주말 심야·저녁은 그대로여야 한다.
    """
    calls = _synthetic_calls(4, ("사직동", "혜화동"))
    base = S.CallTaxiSim(calls, _depots(), matrix)
    tight = S.CallTaxiSim(calls, _depots(), matrix, radius_predawn=7.0)

    predawn, evening = 3 * 60.0, 21 * 60.0
    assert base._radius_km(3, 0, predawn) == S.DISPATCH_RADIUS_NIGHT
    assert tight._radius_km(3, 0, predawn) == 7.0
    # 원문에 있는 구간은 손대지 않는다 — 주말 심야와 19시 이후
    for sim in (base, tight):
        assert sim._radius_km(3, 1, predawn) == S.DISPATCH_RADIUS_NIGHT
        assert sim._radius_km(21, 0, evening) == S.DISPATCH_RADIUS_NIGHT
        assert sim._radius_km(10, 0, 10 * 60.0) == S.DISPATCH_RADIUS_DAY


def test_score_weights_default_to_module_constant(matrix):
    """`score_w` 를 안 주면 `SCORE_W` 그대로여야 한다 — 기본 경로 불변."""
    calls = _synthetic_calls(4, ("사직동", "혜화동"))
    assert S.CallTaxiSim(calls, _depots(), matrix).score_w == S.SCORE_W
    w = {"order": 12.0, "wait": 28.0, "dist": 50.0}
    assert S.CallTaxiSim(calls, _depots(), matrix, score_w=w).score_w == w
    # 인스턴스 값이지 모듈 상수를 덮어쓰는 게 아니다(설정 간 오염 금지)
    assert S.SCORE_W == {"order": 15.0, "wait": 35.0, "dist": 40.0}


def test_new_knobs_leave_default_run_unchanged(matrix, monkeypatch):
    """기본값을 명시로 넘긴 실행과 생략한 실행이 **같아야 한다.**

    민감도를 붙이면서 기존 244곳 결과가 재현되지 않으면 그 결과 전체를 다시
    돌려야 한다. 여기가 그 회귀를 막는 자리다.
    """
    monkeypatch.setattr(S, "FLEET_WEEKDAY", 3)
    monkeypatch.setattr(S, "FLEET_WEEKEND", 3)
    calls = _synthetic_calls(30, ("사직동", "혜화동"))

    a = S.run_placement(None, calls, _depots(), matrix)
    b = S.run_placement(None, calls, _depots(), matrix,
                        score_w=S.SCORE_W,
                        radius_predawn=S.DISPATCH_RADIUS_NIGHT)
    pd.testing.assert_frame_equal(a, b)
