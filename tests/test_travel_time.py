"""
test_travel_time.py — 이동시간 테이블 검증 (코드 점검)

이동시간은 시뮬 전 구간(공차이동·운행)에 곱해지는 값이라 조용히 틀리면
대기지표가 통째로 어긋난다. 여기서는 세 가지를 지킨다:

  1. 키 매핑(시각 → 시간대, 거리 → 거리대)이 경계에서 밀리지 않는다
  2. 폴백 사슬이 어떤 동쌍·시각에도 양수 값을 낸다(결측·0·음수 금지)
  3. 실측 테이블이 있으면 그걸 먼저 쓴다(폴백이 실측을 덮지 않는다)

1·3 은 원본 없이 합성 데이터로, 2 는 실제 테이블로 본다.
원본 CSV 를 읽는 테스트는 느려서 slow 마커: `pytest -m "not slow"` 로 제외.
"""
import numpy as np
import pandas as pd
import pytest

import travel_time as T


# ─────────────────────────────────────────────────────────────
# 키 매핑
# ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("hour, expected", [
    (0, "late_night"), (6, "late_night"),        # 심야 0~6
    (7, "morning_peak"), (9, "morning_peak"),    # 오전피크 7~9
    (10, "daytime"), (15, "daytime"),            # 낮 10~15
    (16, "afternoon_peak"), (18, "afternoon_peak"),
    (19, "evening"), (23, "evening"),
])
def test_period_boundaries(hour, expected):
    """구간 경계 시각이 옆 구간으로 밀리면 안 된다."""
    assert T.period_of(hour) == expected


def test_period_covers_all_hours():
    assert {T.period_of(h) for h in range(24)} == set(T.PERIODS)


def test_period_array_matches_scalar():
    hours = np.arange(24)
    assert list(T.period_of_array(hours)) == [T.period_of(h) for h in hours]


@pytest.mark.parametrize("hour", [-1, 24, 100])
def test_period_rejects_bad_hour(hour):
    with pytest.raises(ValueError):
        T.period_of(hour)


@pytest.mark.parametrize("km, expected", [
    (0.0, "~2km"), (1.99, "~2km"), (2.0, "2-5km"),
    (4.9, "2-5km"), (5.0, "5-10km"), (9.9, "5-10km"),
    (10.0, "10-20km"), (19.9, "10-20km"),    # 20km 컷 — 장거리 과대추정 교정
    (20.0, "20km+"), (40.0, "20km+"),
])
def test_band_boundaries(km, expected):
    assert T.band_of(km) == expected


def test_band_array_matches_scalar():
    km = np.array([0.0, 1.9, 2.0, 4.9, 5.0, 9.9, 10.0, 19.9, 20.0, 30.0])
    assert list(T.band_of_array(km)) == [T.band_of(k) for k in km]


def test_band_labels_match_cuts():
    """구간을 손볼 때 라벨을 같이 안 고치면 여기서 걸린다."""
    assert len(T.DIST_BANDS) == len(T.DIST_CUTS) + 1


def test_haversine_known_distance():
    """서울시청 → 강남역 직선 약 8.8km (남북 7.6 + 동서 4.4)."""
    d = T.haversine_km(37.5665, 126.9780, 37.4979, 127.0276)
    assert 8.6 < d < 9.0


def test_haversine_zero_for_same_point():
    assert T.haversine_km(37.5, 127.0, 37.5, 127.0) == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────
# 폴백 사슬 — 합성 테이블
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def tiny():
    """실측 1셀만 있는 최소 테이블. 나머지는 전부 폴백으로 떨어진다."""
    od = pd.DataFrame([{"origin": "가동", "dest": "나동", "period": "daytime",
                        "is_weekend": False, "minutes": 12.0, "n": 50}])
    speed = pd.DataFrame([{"band": b, "period": p, "is_weekend": w, "kmh": 20.0, "n": 99}
                          for b in T.DIST_BANDS for p in T.PERIODS for w in (False, True)])
    profile = pd.DataFrame([{"period": p, "is_weekend": w, "kmh": 15.0,
                             "minutes": 20.0, "intra_min": 5.0, "n": 99, "n_intra": 99}
                            for p in T.PERIODS for w in (False, True)])
    centroids = pd.DataFrame([
        {"dong": "가동", "lat": 37.50, "lon": 127.00, "key_kind": "canon", "n_src": 1},
        {"dong": "나동", "lat": 37.55, "lon": 127.00, "key_kind": "canon", "n_src": 1},
        {"dong": "다", "lat": 37.60, "lon": 127.00, "key_kind": "base", "n_src": 2},
    ])
    return T.TravelTime(od, speed, profile, centroids)


def test_observed_wins(tiny):
    """실측이 있는 조합은 거리 계산으로 덮이지 않는다."""
    assert tiny.lookup("가동", "나동", 12, False) == (12.0, "od")


def test_falls_to_speed_when_cell_missing(tiny):
    """같은 동쌍이라도 시간대·요일이 다르면 실측 셀이 없어 2순위로 내려간다."""
    minutes, tier = tiny.lookup("가동", "나동", 12, True)
    assert tier == "speed"
    # 직선 5.56km × 우회 1.36 ÷ 20km/h = 22.7분
    expected = T.haversine_km(37.50, 127.0, 37.55, 127.0) * T.DETOUR / 20.0 * 60
    assert minutes == pytest.approx(expected, rel=1e-6)


def test_falls_to_line_when_speed_cell_missing(tiny):
    """속도 셀까지 비면 3순위(시간대 평균속도)로 내려간다."""
    tt = T.TravelTime(tiny.od, tiny.speed.iloc[0:0], tiny.profile, tiny.centroids)
    minutes, tier = tt.lookup("가동", "나동", 12, True)
    assert tier == "line"
    assert minutes == pytest.approx(
        T.haversine_km(37.50, 127.0, 37.55, 127.0) * T.DETOUR / 15.0 * 60, rel=1e-6)


def test_same_dong_gets_minimum(tiny):
    """동 내부 이동은 거리 0 이라 속도로 나누면 0분 — 최소값을 준다."""
    minutes, tier = tiny.lookup("가동", "가동", 12, False)
    assert tier == "intra"
    assert minutes >= T.MIN_INTRA_MINUTES


def test_unknown_dong_falls_back(tiny):
    """좌표를 못 찾는 동이 와도 예외 없이 값을 낸다."""
    minutes, tier = tiny.lookup("없는동", "나동", 12, False)
    assert tier == "default"
    assert minutes > 0


def test_base_name_fallback(tiny):
    """'다1동' 은 중심점에 없지만 기본명 '다' 로 좌표를 찾는다."""
    assert tiny.coord_of("다1동") == (37.60, 127.00)
    assert tiny.lookup("가동", "다1동", 12, False)[1] == "speed"


def test_normalizes_dong_name(tiny):
    """'제N동' 표기로 넣어도 표준형과 같은 결과."""
    assert tiny.lookup("가동", "나동", 12, False) == tiny.lookup(" 가동", "나동", 12, False)


def test_clipped_into_range(tiny):
    """실측이 아무리 튀어도 반환값은 1~120분 안에 든다."""
    od = tiny.od.assign(minutes=9999.0)
    tt = T.TravelTime(od, tiny.speed, tiny.profile, tiny.centroids)
    assert tt.minutes("가동", "나동", 12, False) == T.MAX_MINUTES


def test_lookup_many_matches_scalar(tiny):
    """벡터 조회와 스칼라 조회는 같은 답을 내야 한다(커버율 집계가 이걸 믿는다)."""
    cases = [("가동", "나동", 12, False), ("가동", "나동", 12, True),
             ("가동", "가동", 3, True), ("없는동", "나동", 20, False),
             ("가동", "다1동", 8, False)]
    res = tiny.lookup_many([c[0] for c in cases], [c[1] for c in cases],
                           [c[2] for c in cases], [c[3] for c in cases])
    for i, c in enumerate(cases):
        minutes, tier = tiny.lookup(*c)
        assert res.loc[i, "tier"] == tier
        assert res.loc[i, "minutes"] == pytest.approx(minutes, rel=1e-9)


# ─────────────────────────────────────────────────────────────
# 테이블 산출
# ─────────────────────────────────────────────────────────────

def _synthetic_rides(n_per_cell=12):
    rows = []
    for i in range(n_per_cell):
        rows.append({"origin": "가동", "dest": "나동", "minutes": 10.0 + i,
                     "km": 5.0, "kmh": 20.0, "hour": 12, "period": "daytime",
                     "is_weekend": False})
    rows.append({"origin": "가동", "dest": "다동", "minutes": 30.0, "km": 8.0,
                 "kmh": 16.0, "hour": 12, "period": "daytime", "is_weekend": False})
    return pd.DataFrame(rows)


def test_od_table_drops_thin_cells():
    """10건 미만 조합은 신뢰하지 않고 표에서 뺀다."""
    od = T.build_od_table(_synthetic_rides())
    assert list(od["dest"]) == ["나동"]          # 1건짜리 다동은 빠진다
    assert od.loc[0, "n"] == 12


def test_od_table_uses_median():
    """평균이 아니라 중앙값 — 장시간 이상치에 끌려가지 않게."""
    od = T.build_od_table(_synthetic_rides())
    assert od.loc[0, "minutes"] == pytest.approx(15.5)   # 10~21 의 중앙


def test_profile_intra_defaults_when_thin():
    """동일-동 표본이 없으면 최소값 5분으로 채운다."""
    prof = T.build_profile_table(_synthetic_rides())
    assert (prof["intra_min"] >= T.MIN_INTRA_MINUTES).all()
    assert prof["n_intra"].sum() == 0


# ─────────────────────────────────────────────────────────────
# 실제 테이블 (느림)
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def table(data_dir):
    return T.TravelTime.build()


@pytest.mark.slow
def test_speeds_match_observed_reference(table):
    """시간대별 속도가 실측 참고치(심야 20 / 피크 13~15 / 낮 16)와 크게 어긋나지 않는다."""
    prof = table.profile.set_index(["period", "is_weekend"])
    for period, ref in T.REF_KMH.items():
        kmh = prof.loc[(period, False), "kmh"]
        assert abs(kmh - ref) < 3.0, f"{period}: {kmh:.1f} vs 참고 {ref}"
    # 피크가 심야보다 느려야 한다 — 시간대 구분이 뒤집히면 여기서 걸린다
    assert prof.loc[("afternoon_peak", False), "kmh"] < prof.loc[("late_night", False), "kmh"]


@pytest.mark.slow
def test_speed_table_is_fully_populated(table):
    """거리대 × 시간대 × 요일 전 칸이 신뢰 건수를 넘겨야 한다.

    비면 그 조합은 조용히 3순위(시간대 평균속도)로 흘러가 거리대 구분이 무의미해진다.
    거리대를 더 쪼갤 때 안전한지 판정하는 기준이기도 하다.
    """
    assert len(table.speed) == len(T.DIST_BANDS) * len(T.PERIODS) * 2
    assert (table.speed["n"] >= T.MIN_SPEED_TRIPS).all()


@pytest.mark.slow
def test_long_band_faster_than_mid_band(table):
    """장거리일수록 평균속도가 빨라야 한다(간선·고속 이용).

    20km 컷을 넣은 이유가 이것이다 — 10km+ 를 한 칸에 두면 이 차이가 뭉개져
    장거리가 과대추정된다.
    """
    sp = table.speed.set_index(["band", "period", "is_weekend"])
    for period in T.PERIODS:
        assert (sp.loc[("20km+", period, False), "kmh"]
                > sp.loc[("10-20km", period, False), "kmh"] >
                sp.loc[("5-10km", period, False), "kmh"])


@pytest.mark.slow
def test_long_trip_estimate_near_observed(table):
    """방화1동 → 길동(직선 29km) 실측 중앙 73분. 거리대가 뭉개지면 100분을 넘긴다."""
    minutes = T.get_travel_time("방화1동", "길동", 10, False)
    assert 60 < minutes < 85


@pytest.mark.slow
def test_every_dong_pair_resolves(table):
    """전 동쌍 × 시간대 어디를 찔러도 1~120분 사이 값이 나온다(결측·0·음수 금지)."""
    dongs = table.centroids.loc[table.centroids["key_kind"] == "canon", "dong"].to_numpy()
    o = np.repeat(dongs, len(dongs))
    d = np.tile(dongs, len(dongs))
    for hour, weekend in [(8, False), (14, False), (2, True)]:
        m = table.lookup_many(o, d, hour, weekend)["minutes"].to_numpy()
        assert not np.isnan(m).any()
        assert (m >= T.MIN_MINUTES).all() and (m <= T.MAX_MINUTES).all()


@pytest.mark.slow
def test_farther_takes_longer(table):
    """같은 시각이면 먼 동이 가까운 동보다 오래 걸려야 한다."""
    near = T.get_travel_time("역삼1동", "논현1동", 10, False)
    far = T.get_travel_time("방화1동", "길동", 10, False)
    assert near < far


@pytest.mark.slow
def test_peak_slower_than_night(table):
    """같은 구간이면 오후피크가 심야보다 오래 걸린다."""
    assert (T.get_travel_time("방화1동", "길동", 17, False)
            > T.get_travel_time("방화1동", "길동", 3, False))


@pytest.mark.slow
def test_cache_roundtrip(table):
    """캐시로 다시 읽어도 같은 답이 나온다."""
    again = T.TravelTime.build()
    assert len(again.od) == len(table.od)
    assert again.lookup("역삼1동", "논현1동", 8, False) == table.lookup("역삼1동", "논현1동", 8, False)
