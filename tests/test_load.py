"""
test_load.py — 데이터 로딩 검증 (코드 점검)

동 이름 정규화와 좌표 매칭이 조용히 깨지면 이후 지표가 전부 어긋난다.
콜 원본을 읽는 테스트는 느려서 slow 마커를 붙였다: `pytest -m "not slow"` 로 제외.
"""
from pathlib import Path

import pandas as pd
import pytest

import load

# 서울 대략적 경계 — 좌표가 엉뚱한 곳으로 튀는지 잡는 용도
SEOUL_BBOX = (37.41, 37.71, 126.76, 127.19)   # lat_min, lat_max, lon_min, lon_max


def in_seoul(lat, lon):
    la0, la1, lo0, lo1 = SEOUL_BBOX
    return (lat.between(la0, la1) & lon.between(lo0, lo1)).all()


# ─────────────────────────────────────────────────────────────
# 데이터 경로 해석
# ─────────────────────────────────────────────────────────────

def test_data_dir_defaults_to_sibling_data(monkeypatch):
    """환경변수가 없으면 저장소 상위의 data/ 를 본다."""
    monkeypatch.delenv("DATA_DIR", raising=False)
    assert load.resolve_data_dir() == load.DEFAULT_DATA_DIR
    assert load.DEFAULT_DATA_DIR.name == "data"


def test_data_dir_env_override(monkeypatch, tmp_path):
    """DATA_DIR 을 주면 그 폴더를 본다 — clone 구조가 달라도 돌아가게."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    assert load.resolve_data_dir() == tmp_path.resolve()


def test_data_dir_env_blank_falls_back(monkeypatch):
    """빈 값·공백만 있으면 설정 안 한 것으로 본다."""
    monkeypatch.setenv("DATA_DIR", "   ")
    assert load.resolve_data_dir() == load.DEFAULT_DATA_DIR


def test_data_dir_env_expands_user(monkeypatch):
    monkeypatch.setenv("DATA_DIR", "~/calltaxi-data")
    assert load.resolve_data_dir() == (Path.home() / "calltaxi-data").resolve()


# ─────────────────────────────────────────────────────────────
# 저장소 동봉 자료(assets/) — clone 만으로 대시보드가 뜨게 하는 최소 집합
# ─────────────────────────────────────────────────────────────

def test_assets_ship_the_files_the_dashboard_needs():
    """대시보드가 data/ 없이 뜨려면 이 셋이 저장소에 있어야 한다."""
    for name in ("차고지44_좌표.csv", "sim_pool_v4.csv",
                 load.DISPLAY_BOUNDARY):
        p = load.ASSETS_DIR / name
        assert p.exists(), f"{name} 이 assets/ 에 없다 — clone 만으로 안 뜬다"
    # 콜 원본은 개인정보라 절대 동봉하지 않는다
    assert not (load.ASSETS_DIR / load.CALLS_FILE).exists(), \
        "콜 원본이 저장소에 들어왔다"


def test_data_wins_over_asset_copy(tmp_path, monkeypatch):
    """원본이 있으면 원본을 쓴다 — 사본이 낡아 조용히 옛 값을 읽으면 안 된다."""
    monkeypatch.setattr(load, "DATA_DIR", tmp_path)
    real = tmp_path / "sim_pool_v4.csv"
    real.write_text("x", encoding="utf-8")
    assert load.data_or_asset("sim_pool_v4.csv") == real
    # 없으면 저장소 사본으로 내려간다
    assert load.data_or_asset("차고지44_좌표.csv") == \
        load.ASSETS_DIR / "차고지44_좌표.csv"


def test_boundary_for_calculation_never_falls_back(tmp_path, monkeypatch):
    """**계산용 경계는 폴백하지 않는다.**

    `assets/` 사본은 0.0002도(약 22m) 단순화본이라 면적(`geometry.area`)·
    `sjoin`·EPSG:5179 격자에 쓰면 값이 틀린다 — `candidates.py` 가 그 셋을 전부
    한다. data/ 가 없으면 **조용히 대체하지 말고 그대로 실패해야** 한다.
    """
    monkeypatch.setattr(load, "DATA_DIR", tmp_path)
    with pytest.raises(Exception):
        load.load_dong(with_boundary=True)

    src = Path(load.__file__).read_text(encoding="utf-8")
    body = src.split("def load_dong", 1)[1].split("\ndef ", 1)[0]
    assert "data_or_asset" not in body, \
        "계산용 경계 로더가 저장소 사본으로 폴백한다"


# ─────────────────────────────────────────────────────────────
# 동 이름 정규화
# ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw, expected", [
    ("고덕제1동", "고덕1동"),
    ("성수2가제3동", "성수2가3동"),      # '제'가 두 번 나오는 중첩 케이스
    ("공릉1.3동", "공릉13동"),           # 마침표 제거
    ("사직동", "사직동"),                # 숫자 없으면 그대로
    ("종로1.2.3.4가동", "종로1234가동"),
])
def test_canon_dong(raw, expected):
    assert load.canon_dong(raw) == expected


def test_canon_dong_is_idempotent():
    """이미 표준형인 이름을 다시 넣어도 바뀌지 않아야 한다."""
    for raw in ["고덕제1동", "성수2가제3동", "제기제1동", "신당제6동"]:
        once = load.canon_dong(raw)
        assert load.canon_dong(once) == once


def test_canon_dong_keeps_leading_je():
    """'제기동'의 '제'는 숫자 앞이 아니므로 지워지면 안 된다."""
    assert load.canon_dong("제기동") == "제기동"
    assert load.canon_dong("제기제1동") == "제기1동"


# ─────────────────────────────────────────────────────────────
# 조회표·보조 테이블
# ─────────────────────────────────────────────────────────────

def test_dong_lookup_keys_unique(data_dir):
    lk = load.build_dong_lookup()
    assert not lk.duplicated(subset=["gu", "key", "key_kind"]).any()
    assert set(lk["key_kind"]) == {"canon", "base"}
    assert set(lk["match_type"]) == {"exact", "approx", "manual"}


def test_dong_lookup_covers_all_seoul_gu(data_dir):
    lk = load.build_dong_lookup()
    assert set(lk[lk["match_type"] == "exact"]["gu"]) == set(load.SEOUL_GU)


def test_dong_lookup_manual_overrides_resolve(data_dir):
    """개편으로 사라진 동이 후신 동 좌표를 받아야 한다."""
    lk = load.build_dong_lookup()
    canon = lk[lk["key_kind"] == "canon"].set_index(["gu", "key"])
    for (gu, src), target in load.MANUAL_DONG.items():
        assert (gu, load.canon_dong(src)) in canon.index, f"{gu} {src} 미해결"
        got = canon.loc[(gu, load.canon_dong(src))]
        want = canon.loc[(gu, load.canon_dong(target))]
        assert got["lat"] == want["lat"] and got["lon"] == want["lon"]


def test_load_depots(data_dir):
    dep = load.load_depots()
    assert len(dep) == 44
    assert dep["capacity"].sum() == 691
    assert dep[["lat", "lon"]].notna().all().all()
    assert in_seoul(dep["lat"], dep["lon"])
    assert dep["gu"].isin(load.SEOUL_GU).all()
    assert not dep["depot_id"].duplicated().any()


def test_load_dong_centroids(data_dir):
    cen = load.load_dong()
    assert len(cen) == 426
    assert in_seoul(cen["lat"], cen["lon"])
    assert not cen.duplicated(subset=["gu", "dong_canon"]).any()


def test_load_candidates_outdoor_only(data_dir):
    """기본은 옥외만 — 옥내·혼합이 섞이면 A-15 가 깨진 것이다."""
    c = load.load_candidates()
    assert len(c) == 348
    assert c["capacity"].sum() == 24_392
    assert (c["enclosure"] == "옥외").all()
    assert c["is_outdoor"].all()
    assert in_seoul(c["lat"], c["lon"])
    assert c["gu"].isin(load.SEOUL_GU).all()
    assert (c["capacity"] >= 0).all()
    assert not c["cand_id"].duplicated().any()


def test_load_candidates_keeps_full_pool(data_dir):
    """옥내·혼합은 제외하되 목록은 보존한다 — 전고 확인 시 재편입할 후보다."""
    full = load.load_candidates(outdoor_only=False)
    assert len(full) == 639
    assert full["enclosure"].value_counts().to_dict() == {"옥외": 348, "옥내": 163, "혼합": 128}
    assert full.attrs["n_excluded_indoor_mixed"] == 291
    # cand_id 는 원본 행 순서 — 옥외만 걸러도 같은 후보는 같은 번호를 가져야 한다
    out = load.load_candidates()
    merged = out.merge(full, on="cand_id", suffixes=("", "_full"))
    assert len(merged) == len(out)
    assert (merged["name"] == merged["name_full"]).all()


def test_capacity_is_total_area_for_every_candidate(data_dir):
    """용량은 전 후보 총 면수 하나뿐이다."""
    c = load.load_candidates(outdoor_only=False)
    raw = pd.read_csv(load.DATA_DIR / "sim_pool_v4.csv", encoding="utf-8-sig")
    assert c["capacity"].sum() == round(raw["면수"].sum()) == 55_798
    assert c.loc[c["source"] == "시영", "capacity"].sum() == 6_950


def test_peak_residual_is_gone(data_dir):
    """**실가용 면수는 철회했다**(08.10) — 열도 함수도 남아 있으면 안 된다.

    639곳 중 시영 65곳(10%)에만 있는 값이라 참고 컬럼으로도 쓸 수 없었다.
    되살아나면 그 65곳만 다른 자로 재게 된다.
    """
    c = load.load_candidates(outdoor_only=False)
    for col in ("capacity_available", "capacity_available_ratio"):
        assert col not in c.columns, f"철회한 열이 살아 있다: {col}"
    for attr in ("n_capacity_available", "capacity_available_note"):
        assert attr not in c.attrs
    for fn in ("load_peak_residual", "find_foia_parking_xlsx"):
        assert not hasattr(load, fn), f"철회한 함수가 살아 있다: {fn}"


def test_load_undersupplied(data_dir):
    u = load.load_undersupplied()
    assert (u["vehicles_3km"] <= 10).all()
    assert u["long_wait_ratio"].between(0, 1).all()
    assert in_seoul(u["lat"], u["lon"])
    # 임계를 올리면 대상 동이 늘어야 한다
    assert len(load.load_undersupplied(max_vehicles_3km=30)) > len(u)


# ─────────────────────────────────────────────────────────────
# 콜 원본 (느림)
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def calls(data_dir):
    return load.load_calls()


@pytest.mark.slow
def test_calls_filters_applied(calls):
    assert calls["origin_gu"].isin(load.SEOUL_GU).all()
    gap = (calls["scheduled_at"] - calls["received_at"]).abs().dt.total_seconds() / 60
    assert (gap <= load.IMMEDIATE_TOLERANCE_MIN).all()
    assert calls["origin_lat"].notna().all(), "출발 좌표는 배차에 필수"
    assert not calls["call_id"].duplicated().any()


@pytest.mark.slow
def test_calls_wait_time_sane(calls):
    w = calls["wait_min"].dropna()
    assert (w >= 0).all(), "대기시간은 음수가 될 수 없다"
    assert (w <= load.MAX_WAIT_MIN).all()
    # 취소건은 승차 기록이 없으므로 대기시간도 없어야 한다
    assert calls.loc[calls["is_canceled"], "wait_min"].isna().all()


@pytest.mark.slow
def test_calls_reproduce_observed_targets(calls):
    """검증 기준이 실측에서 재현되는지 — 필터 정의가 맞는지 확인.

    특장차 한정본 기준이다(A-16). 임차택시를 섞으면 39.3 / 30.8 / 77.2 / 17.9% 로
    내려간다 — 취소율 4.8%의 다른 운영 체계가 평균을 끌어내린 값이다.
    """
    assert len(calls) == 1_222_330
    w = calls["wait_min"].dropna()
    assert w.mean() == pytest.approx(40.8, abs=0.3)
    assert w.median() == pytest.approx(32.0, abs=0.3)
    assert w.quantile(0.90) == pytest.approx(80.4, abs=0.5)
    assert (w > 60).mean() == pytest.approx(0.192, abs=0.005)


@pytest.mark.slow
def test_unsettled_is_not_canceled(calls):
    """정산 미기록(명세 1-5) — 승차·취소 없이 하차만 있는 건은 취소가 아니다."""
    u = calls["is_unsettled"]
    assert u.sum() == 524
    assert calls.loc[u, "boarded_at"].isna().all()
    assert calls.loc[u, "canceled_at"].isna().all()
    assert calls.loc[u, "alighted_at"].notna().all()
    assert calls.loc[u, "assigned_at"].notna().all(), "배차는 전부 기록돼 있다"
    assert not calls.loc[u, "is_canceled"].any()
    # 섞으면 취소율이 15.44% → 15.48% 로 부푼다
    assert calls["is_canceled"].mean() == pytest.approx(0.1544, abs=0.0005)
    assert calls["boarded_at"].isna().mean() == pytest.approx(0.1548, abs=0.0005)


@pytest.mark.slow
def test_calls_period_matches_hour(calls):
    """시간대 파생이 시각과 어긋나지 않아야 한다(travel_time 이 이 키를 쓴다)."""
    assert set(calls["period"].unique()) <= set(load.PERIOD_LABELS)
    assert (calls.loc[calls["hour"] == 3, "period"] == "심야").all()
    assert (calls.loc[calls["hour"] == 8, "period"] == "아침").all()
    assert (calls.loc[calls["hour"] == 14, "period"] == "낮").all()
    assert (calls.loc[calls["hour"] == 20, "period"] == "저녁").all()
    assert (calls.loc[calls["hour"] == 23, "period"] == "심야").all()
