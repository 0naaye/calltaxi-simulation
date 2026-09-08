"""
test_candidates.py — 동별 후보 배정 검증 (코드 점검)

배정 규칙이 조용히 흔들리면 "이 동에 거점을 놓으면" 이후의 모든 계산이 다른
자리를 재게 된다. 규칙의 경계값(1km · 1.2km)과 라벨의 정합성을 못 박아 둔다.

경계 GeoJSON(34MB)을 읽으므로 모듈 스코프로 한 번만 배정한다.
"""
import numpy as np
import pytest

import candidates as C
import load

SEOUL_BBOX = (37.41, 37.71, 126.76, 127.19)


@pytest.fixture(scope="module")
def cands(data_dir):
    return load.load_candidates()


@pytest.fixture(scope="module")
def assign(cands):
    return C.assign_dong_candidates(cands)


# ─────────────────────────────────────────────────────────────
# 거리 계산
# ─────────────────────────────────────────────────────────────

def test_haversine_zero():
    assert C.haversine_m(37.5, 127.0, 37.5, 127.0) == pytest.approx(0, abs=1e-6)


def test_haversine_known_distance():
    """위도 1도 = 약 111km. 부호·라디안 변환이 뒤집히면 여기서 걸린다."""
    d = C.haversine_m(37.0, 127.0, 38.0, 127.0)
    assert d == pytest.approx(111_195, rel=0.001)


def test_haversine_broadcasts():
    lat = np.array([37.5, 37.6])[:, None]
    lon = np.array([127.0, 127.1])[:, None]
    d = C.haversine_m(lat, lon, np.array([[37.5, 37.7]]), np.array([[127.0, 127.2]]))
    assert d.shape == (2, 2)
    assert d[0, 0] == pytest.approx(0, abs=1e-6)


# ─────────────────────────────────────────────────────────────
# 배정
# ─────────────────────────────────────────────────────────────

def test_every_dong_appears_once(assign):
    """동 426개가 빠짐없이, 한 번씩. 누락되면 그 동은 조용히 시뮬에서 사라진다."""
    assert len(assign) == 426
    assert not assign.duplicated(subset=["adm_cd2"]).any()
    assert not assign.duplicated(subset=["gu", "dong_canon"]).any()
    assert assign["gu"].isin(load.SEOUL_GU).all()


def test_all_candidates_land_inside_boundary(cands):
    """후보 좌표가 서울 경계 밖으로 나가면 ⑴ 판정이 통째로 어긋난다."""
    located = C.locate_candidates(cands)
    assert len(located) == len(cands)
    assert located["adm_cd2"].notna().all()


def test_rule_labels_match_distance(assign):
    """라벨과 거리가 어긋나면 안 된다 — 규칙이 사후에 붙은 게 아니라는 확인."""
    near = assign[assign["assign_rule"] == C.RULE_NEAR]
    assert (near["nearest_dist_m"] <= C.NEAR_RADIUS_M).all()

    edge = assign[assign["assign_rule"] == C.RULE_EDGE]
    assert (edge["nearest_dist_m"] > C.NEAR_RADIUS_M).all()
    assert (edge["nearest_dist_m"] <= C.EDGE_RADIUS_M).all()

    unassigned = assign[~assign["is_assigned"]]
    assert (unassigned["nearest_dist_m"] > C.EDGE_RADIUS_M).all()


def test_inside_rule_picks_max_capacity(assign, cands):
    """⑴ 은 동 안에서 총 면수 최대 1곳. 다른 곳이 뽑히면 규칙이 깨진 것이다."""
    located = C.locate_candidates(cands)
    inside = assign[assign["assign_rule"] == C.RULE_INSIDE]
    assert len(inside) == inside["n_outdoor_in_dong"].gt(0).sum()

    for _, row in inside.iterrows():
        pool = located[located["adm_cd2"] == row["adm_cd2"]]
        assert row["capacity"] == pool["capacity"].max()
        assert row["cand_id"] in set(pool["cand_id"])


def test_unassigned_dong_has_no_candidate(assign):
    """배정 안 된 동에 후보 값이 남아 있으면 '최근접'이 '배정'으로 읽힌다."""
    un = assign[~assign["is_assigned"]]
    assert len(un) > 0
    for col in ("cand_id", "cand_name", "capacity", "dist_m"):
        assert un[col].isna().all()
    # 반대로 최근접 거리는 항상 있어야 한다 — 보류 판정의 근거다
    assert assign["nearest_dist_m"].notna().all()


def test_substitute_assignment_respects_cap(assign):
    """⑵⑶ 대체는 1.2km 를 넘을 수 없다 — 넘으면 시뮬이 엉뚱한 자리를 거점으로 쓴다.

    표시만 하고 넘기지 않는지 확인하는 게 핵심이다. 컬럼에 값이 남아 있으면
    아래 단정이 통과해도 배정 자체가 무효인 상태가 된다.
    """
    sub = assign[assign["is_assigned"] & (assign["assign_rule"] != C.RULE_INSIDE)]
    assert len(sub) > 0
    assert (sub["dist_m"] <= C.EDGE_RADIUS_M).all()
    # 상한을 넘은 동은 후보 칸이 실제로 비어 있어야 한다
    beyond = assign[assign["nearest_dist_m"] > C.EDGE_RADIUS_M]
    non_inside = beyond[beyond["assign_rule"] != C.RULE_INSIDE]
    assert non_inside["cand_id"].isna().all()
    assert not non_inside["is_assigned"].any()


def test_cap_violation_raises(cands, monkeypatch):
    """라벨링이 어긋나 상한 초과가 새어 나오면 조용히 통과하지 않고 터져야 한다."""
    monkeypatch.setattr(C, "ASSIGNED_RULES",
                        C.ASSIGNED_RULES + (C.RULE_PENDING,))
    with pytest.raises(AssertionError, match="상한"):
        C.assign_dong_candidates(cands)


def test_inside_rule_is_exempt_but_flagged(assign):
    """⑴ 내부는 상한 대상이 아니다(후보가 동 안에 있다). 대신 표시는 남는다."""
    far = assign[assign["cand_beyond_cap"]]
    assert (far["assign_rule"] == C.RULE_INSIDE).all()
    assert (far["dist_m"] > C.EDGE_RADIUS_M).all()
    assert (far["n_outdoor_in_dong"] > 0).all()
    assert not assign.loc[~assign["is_assigned"], "cand_beyond_cap"].any()


def test_assigned_dong_has_full_candidate_row(assign):
    ok = assign[assign["is_assigned"]]
    for col in ("cand_id", "cand_name", "capacity",
                "cand_lat", "cand_lon", "dist_m"):
        assert ok[col].notna().all()
    la0, la1, lo0, lo1 = SEOUL_BBOX
    assert ok["cand_lat"].between(la0, la1).all()
    assert ok["cand_lon"].between(lo0, lo1).all()
    assert (ok["dist_m"] >= 0).all()


def test_airport_greenbelt_only_when_far(assign):
    """공항·개발제한 라벨은 1.2km 밖일 때만 붙는다 — 가까우면 그 후보를 쓴다."""
    air = assign[assign["assign_rule"] == C.RULE_AIRPORT]
    assert set(zip(air["gu"], air["dong_canon"])) <= C.AIRPORT_GREENBELT
    assert (air["nearest_dist_m"] > C.EDGE_RADIUS_M).all()


def test_radius_widening_only_adds_dong(cands):
    """반경을 넓히면 배정 동이 줄어들 수 없다 — 단조성 확인."""
    wide = C.assign_dong_candidates(cands, near_m=1500, edge_m=1500)
    base = C.assign_dong_candidates(cands)
    assert wide["is_assigned"].sum() >= base["is_assigned"].sum()


def test_capacity_is_the_only_size_column(assign, cands):
    """배정 순위는 총 면수로만 낸다. 실가용 열은 08.10 에 철회했다."""
    ok = assign[assign["is_assigned"]]
    by_id = cands.set_index("cand_id")["capacity"]
    assert (ok["capacity"] == ok["cand_id"].map(by_id)).all()
    assert "capacity_available" not in assign.columns


def test_centroid_recalc_dongs_are_unassigned(assign):
    """재계산 지정 동은 미배정 상태여야 한다 — 배정돼 버리면 재계산 대상이 아니다."""
    key = set(zip(assign["gu"], assign["dong_canon"]))
    assert C.CENTROID_RECALC_DONG <= key, "지정 동 이름이 격자와 어긋난다"
    named = assign[[(g, d) in C.CENTROID_RECALC_DONG
                    for g, d in zip(assign["gu"], assign["dong_canon"])]]
    assert len(named) == len(C.CENTROID_RECALC_DONG)
    assert not named["is_assigned"].any()


def test_summary_counts_match(assign):
    summ = C.assignment_summary(assign)
    assert summ["n_dong"].sum() == len(assign)
    assigned = summ[summ["assign_rule"].isin(C.ASSIGNED_RULES)]
    assert assigned["n_dong"].sum() == int(assign["is_assigned"].sum())
    # 대체 배정은 겹칠 수 있다 — 후보 수가 동 수를 넘으면 계산이 잘못된 것이다
    assert (assigned["n_cand"] <= assigned["n_dong"]).all()
