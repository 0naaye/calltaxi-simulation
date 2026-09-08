# -*- coding: utf-8 -*-
"""대시보드 로더 점검 — 출처의 attrs 가 살아 있는가.

**attrs 는 조용히 사라진다.** pandas `merge` 는 attrs 를 물려주지 않아서,
로더가 후보에 배정 동을 붙이는 순간 `n_pool`·`n_excluded_indoor_mixed` 가
날아갔다(KeyError: 'n_pool' — 사이드바 "후보 풀 {n}곳" 문구). 타입 검사에도
테스트에도 안 걸리고 화면을 켤 때만 터진다.

그 문구는 08.10 에 걷어냈지만 **검사는 남긴다** — 로더가 출처의 메타데이터를
잃는 것 자체가 버그이고, 다음에 누가 읽을 때 또 같은 자리에서 터진다.
"""
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("streamlit", reason="대시보드 의존성 없음")

import dashboard as D   # noqa: E402
import load             # noqa: E402


# `load.load_candidates()` 가 담아 주는 값들. merge 를 지나도 살아야 한다.
SOURCE_ATTRS = ("n_pool", "n_excluded_indoor_mixed")


@pytest.fixture(scope="module")
def cand(data_dir):
    # st.cache_data 를 우회한다 — 캐시가 있으면 로더 본문을 안 타서
    # attrs 가 사라져도 통과해 버린다.
    return D.load_candidates.__wrapped__()


def test_loader_keeps_the_source_attrs(cand):
    """merge 뒤에도 원본 attrs 가 남아야 한다 — 이게 KeyError 의 원인이었다."""
    for key in SOURCE_ATTRS:
        assert key in cand.attrs, f"출처의 attrs 가 사라졌다: {key}"
    assert cand.attrs["n_pool"] == 639
    assert cand.attrs["n_excluded_indoor_mixed"] == 291
    # 로더 자신이 붙이는 값도 함께 있어야 한다(update 가 덮어쓰면 안 된다).
    assert cand.attrs["n_outdoor"] == 348


def test_loader_attrs_match_the_source(cand):
    """대시보드가 들고 있는 숫자와 load.py 가 말하는 숫자가 같아야 한다."""
    src = load.load_candidates()
    for key in SOURCE_ATTRS:
        assert cand.attrs[key] == src.attrs[key]


def test_assigned_subset_is_smaller_than_outdoor(cand):
    """배정된 곳만 남는다 — 지도가 시뮬 입력과 같은 것을 보여줘야 한다."""
    if not cand.attrs.get("assigned_only"):
        pytest.skip("dong_candidates.csv 가 없어 348곳 전체로 물러난 상태")
    assert len(cand) == 244
    assert len(cand) < cand.attrs["n_outdoor"] == 348
    assert cand["assigned_dongs"].str.len().gt(0).all()
    assert not cand["cand_id"].duplicated().any()


def test_depot_layer_click_changes_nothing(cand):
    """**차고지 점을 눌러도 지도가 변하지 않아야 한다.**

    `clickmode="event+select"` 라 어느 점을 클릭하면 Plotly 가 나머지 트레이스를
    '비선택'으로 보고 흐리게 만든다 — 차고지를 눌렀는데 동들이 회색조가 되던
    것이 이것이다. 트레이스마다 비선택 투명도를 제 값으로 못 박아 막는다.

    차고지는 태그도 `location` 도 없어 클릭 처리기가 그냥 흘린다(아래).
    """
    poly = D.resolve_polygons.__wrapped__()
    for selected in (None, poly.iloc[0]["adm_cd2"]):
        fig = D.build_map(poly, selected, True, True, D.DEFAULT_METRIC)
        for tr in fig.data:
            normal = 1.0 if tr.marker.opacity is None else tr.marker.opacity
            assert tr.unselected.marker.opacity == normal, (
                f"{tr.name} 이 선택 때문에 흐려진다")

    dep = [t for t in fig.data if t.name == "차고지"]
    assert dep, "차고지 레이어가 없다"
    # 클릭 처리기가 보는 두 갈래 어느 쪽에도 걸리지 않아야 한다
    assert dep[0].customdata is None, "차고지에 customdata 가 붙으면 후보로 오인된다"
    assert not hasattr(dep[0], "locations"), "차고지는 코로플레스가 아니다"
    assert dep[0].hovertemplate, "hover 는 살아 있어야 한다"


def test_map_customdata_carries_cand_id(cand):
    """후보 점 클릭이 3페이지로 가려면 customdata 에 cand_id 가 있어야 한다.

    칸 번호(`CAND_ID_IDX`)를 build_map 과 클릭 처리기가 함께 본다 — 실가용
    면수를 뺄 때 칸이 한 자리 밀렸고, 상수로 묶지 않았으면 엉뚱한 후보가 열렸다.
    """
    poly = D.resolve_polygons.__wrapped__()
    fig = D.build_map(poly, None, True, True, D.DEFAULT_METRIC)
    tr = [t for t in fig.data if getattr(t, "name", "") == "후보 주차장"]
    assert tr, "후보 레이어가 없다"
    cds = tr[0].customdata
    i = D.CAND_ID_IDX
    assert len(cds) == len(cand)
    assert all(len(c) == i + 1 for c in cds), "cand_id 는 마지막 칸이어야 한다"
    assert all(c[0] == D.CAND_TAG and isinstance(c[i], int) for c in cds)
    assert {c[i] for c in cds} == set(cand["cand_id"].astype(int))
    # hovertemplate 이 참조하는 칸이 실제로 다 있어야 한다
    for n in range(1, i):
        assert all(c[n] is not None for c in cds)


# ─────────────────────────────────────────────────────────────
# 3페이지 — 시뮬 결과
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def placement():
    g = D.load_placement.__wrapped__()
    if g.empty:
        pytest.skip("placement_grades.csv 가 없다")
    return g


def test_placement_drops_grade_columns(placement):
    """**등급 열은 화면 코드에 도달하면 안 된다.**

    안 쓰는 것으로는 부족하다 — 열이 남아 있으면 다음 사람이 집어 쓴다.
    로더에서 잘라 두면 3페이지가 등급에 손댈 방법 자체가 없다.
    """
    assert not [c for c in placement.columns if c.startswith(D.REF_PREFIX)]
    for banned in ("grade", "gap_prev", "threshold_prev"):
        assert banned not in placement.columns
    # 화면이 쓰는 열은 그대로 있어야 한다
    for keep in ("rate_pct", "interval", "n_overlap", "sample_ok",
                 "total_delta_min", "n_calls_scope", "n_dong_scope", "after",
                 # 총 대기는 참고 지표지만 `ref_` 가 아니다 — 화면이 쓴다.
                 # 등급과 성격이 다르다: 등급은 판단을 대신하는 표시라 잘라내고,
                 # 이쪽은 「참고」 꼬리표를 달고 보여주는 값이다.
                 "rate_total_pct", "rate_total_sd"):
        assert keep in placement.columns


def test_scope_reproduces_recorded_dong_count(placement):
    """**다시 잡은 영향권의 동 수가 CSV 의 `n_dong_scope` 와 같아야 한다.**

    영향권은 저장돼 있지 않아 `evaluate.scope_members` 로 다시 잡는다. 반경이나
    좌표 소스가 어긋나면 지도와 숫자가 조용히 갈라진다 — 여기서 잡는다.
    """
    scope = D.placement_scope.__wrapped__()
    assert len(scope) == len(placement) == 244
    bad = [(int(r.cand_id), scope[int(r.cand_id)]["n_dong"], int(r.n_dong_scope))
           for r in placement.itertuples()
           if scope[int(r.cand_id)]["n_dong"] != int(r.n_dong_scope)]
    assert not bad, f"영향권 동 수가 어긋난다: {bad[:5]}"
    for r in placement.itertuples():
        s = scope[int(r.cand_id)]
        assert s["assigned"], f"cand {r.cand_id} 담당 동이 비었다"
        assert s["lat"] is not None and s["lon"] is not None


def test_unmappable_dongs_are_counted_not_hidden(placement):
    """경계가 없거나 합쳐진 동은 **세어서 알린다.**

    경계 파일(2023)과 콜 원본의 동 구분이 달라 폴리곤 수가 동 수보다 적을 수
    있다(244곳 중 66곳). 침묵하면 지도가 "18개 동"이라 적고 17개를 칠한다.
    """
    scope = D.placement_scope.__wrapped__()
    gap = 0
    for s in scope.values():
        assert len(s["polys"]) <= s["n_dong"], "폴리곤이 동보다 많을 수는 없다"
        assert s["n_unmapped"] == s["n_dong"] - len(s["polys"])
        gap += bool(s["n_unmapped"])
    assert gap > 0, "어긋남이 사라졌다면 이 경고 문구도 걷어내야 한다"
    assert gap == 66, f"어긋나는 후보 수가 바뀌었다: {gap} (전에는 66)"


def test_sim_map_paints_exactly_the_scope(placement):
    """영향권만 칠하고 담당 동은 테두리로 가른다."""
    scope = D.placement_scope.__wrapped__()
    for cid in (380, 411, 515):
        row = placement[placement["cand_id"] == cid].iloc[0]
        fig = D.build_sim_map(cid, row)
        names = [t.name for t in fig.data]
        assert names == ["영향권 밖", "3km 영향권", "담당 동", "후보"]

        painted = sum(1 for v in fig.data[1].z if v is not None)
        assert painted == len(scope[cid]["polys"])
        n_asg = sum(1 for v in fig.data[2].z if v is not None)
        assert n_asg == len(scope[cid]["assigned"])
        # 담당 동은 영향권의 부분집합이라 채움이 같아야 한다 — 테두리만 다르다
        assert fig.data[2].colorscale == fig.data[1].colorscale
        assert fig.data[2].marker.line.width > fig.data[1].marker.line.width


def test_sim_map_uirevision_is_per_candidate(placement):
    """후보를 바꾸면 카메라를 다시 잡아야 한다 — "constant" 면 화면 밖에 남는다."""
    rows = {c: placement[placement["cand_id"] == c].iloc[0] for c in (380, 411)}
    revs = {c: D.build_sim_map(c, r).layout.uirevision for c, r in rows.items()}
    assert revs[380] != revs[411]
    # 같은 후보를 다시 그리면 값이 같아야 줌·팬이 유지된다
    assert D.build_sim_map(380, rows[380]).layout.uirevision == revs[380]


def test_improve_color_is_monotone():
    """개선율이 클수록 짙어야 한다 — 후보 간 비교가 색의 유일한 용도다."""
    light = D.improve_color(-1.5, 1.5, 8.6)
    dark = D.improve_color(-8.6, 1.5, 8.6)
    assert light == D.IMPROVE_LIGHT and dark == D.IMPROVE_DARK
    lum = lambda h: sum(int(h[i:i + 2], 16) for i in (1, 3, 5))  # noqa: E731
    assert lum(light) > lum(D.improve_color(-5.0, 1.5, 8.6)) > lum(dark)


def test_candidate_options_sort_and_flag(placement):
    """정렬 두 축 · 표본 부족 표시 · **순위 번호 없음**."""
    by_rate = D.candidate_options("rate")
    by_total = D.candidate_options("total")
    assert len(by_rate) == len(by_total) == len(placement)

    # 정렬 축이 실제로 다르다 — 같으면 둘 중 하나가 무의미하다
    assert [c for c, _ in by_rate] != [c for c, _ in by_total]
    assert by_rate[0][0] == int(placement.sort_values("rate_pct").iloc[0]["cand_id"])
    assert by_total[0][0] == int(
        placement.sort_values("total_delta_min").iloc[0]["cand_id"])

    # 두 축의 값이 라벨에 함께 있어야 한다 — 정렬 키만 보이면 그 축이 정답으로 읽힌다
    label = dict(by_rate)[by_rate[0][0]]
    assert "%" in label and "분" in label

    weak = {int(r.cand_id) for r in placement.itertuples() if not r.sample_ok}
    for cid, lab in by_rate:
        assert ("※표본부족" in lab) == (cid in weak)

    # 순위 번호를 붙이지 않는다
    for _, lab in by_rate[:5]:
        assert not lab.lstrip().split(".")[0].isdigit()
        assert "위" not in lab


# ─────────────────────────────────────────────────────────────
# 3페이지 — 시나리오 전제 · 분포 · 문구
# ─────────────────────────────────────────────────────────────

def test_fleet_effect_matches_the_simulator():
    """전제 블록의 숫자는 **화면에 적은 값이 아니라 계산한 값**이어야 한다.

    574/582/8 을 문자열로 박아 두면 `NEW_DEPOT_VEHICLES`·`FLEET_WEEKDAY`·차고지
    정원이 바뀔 때 조용히 어긋난다. `run_placement` 이 하는 계산과 같은지 본다.
    """
    import simulator as S
    import load as L

    f = D.fleet_effect.__wrapped__()
    dep = L.load_depots()
    assert f["cap_base"] == int(dep["capacity"].sum())
    assert f["cap_new"] == f["cap_base"] + S.NEW_DEPOT_VEHICLES

    for tag, base in (("weekday", S.FLEET_WEEKDAY), ("weekend", S.FLEET_WEEKEND)):
        got = f[tag]
        assert got["base"] == base
        assert got["after"] == int(round(base * f["scale"]))
        assert got["new_depot"] >= 1
        # A-10 — 기존 44곳 배속은 건드리지 않는다. 이게 깨지면 전제 문장이 거짓이 된다
        assert got["changed"] == 0, "기존 거점 배분이 바뀌면 '그대로입니다'가 거짓이다"


def test_position_text_never_prints_a_lying_percent():
    """양 끝에서 **백분율이 거짓말을 한다** — 그 구간만 말로 바꾼다.

    0% 는 "아무것도 아닌 건가"로 읽히고, 100% 는 **최대치로 읽혀 뜻이 정반대로
    전달된다**(실제로는 개선이 가장 작은 쪽 끝이다). 둘 다 화면에 나가면 안 된다.
    """
    import re
    plain = lambda s: re.sub(r"<[^>]+>", "", s)  # noqa: E731 — 굵게 표시를 걷어낸다
    v = np.linspace(-8.0, -1.0, 244)

    best = plain(D._position_text(v, -8.0, "개선"))
    worst = plain(D._position_text(v, -1.0, "개선"))
    assert best == "개선이 가장 큰 쪽 끝입니다"
    assert worst == "개선이 가장 작은 쪽 끝입니다"
    assert "%" not in best and "%" not in worst

    # 사이는 백분율 그대로. 0%·100% 는 어디서도 나오지 않는다
    for cur in v:
        t = plain(D._position_text(v, float(cur), "개선"))
        assert "약 0%" not in t and "약 100%" not in t
    assert "약 50% 지점" in plain(D._position_text(v, float(np.median(v)), "개선"))


def test_subject_particle_matches_the_final_consonant():
    """「수혜이」 같은 조사가 화면에 나가지 않는다 — 받침으로 갈린다."""
    assert D._subject("개선") == "개선이"      # 받침 ㄴ
    assert D._subject("수혜") == "수혜가"      # 받침 없음
    assert D._subject("절감") == "절감이"
    assert D._subject("대기") == "대기가"
    assert D._subject("A") == "A이"            # 한글이 아니면 조용히 넘어간다


def test_distribution_marker_sits_on_the_candidate(placement):
    """표식이 그 후보의 값에 찍히고, **순위 번호는 어디에도 없다.**"""
    rate = placement["rate_pct"].to_numpy(float)
    cur = float(placement.sort_values("rate_pct").iloc[3]["rate_pct"])
    fig = D._dist_fig(rate, cur, D._label_pct)

    lines = [s for s in fig.layout.shapes if s.type == "line"]
    assert len(lines) == 1 and lines[0].x0 == pytest.approx(cur)
    marker = [a for a in fig.layout.annotations if a.yanchor == "bottom"]
    assert [a.text for a in marker] == ["이 후보"], "순위·등수 표기가 붙으면 안 된다"
    # y축은 통째로 지운다 — 도수는 세는 값이 아니라 모양이다
    assert fig.layout.yaxis.visible is False


def test_both_distributions_label_their_ends(placement):
    """**두 차트 모두** 양 끝 값을 보인다 — 축이 없으면 분포의 폭을 모른다.

    총절감만 축이 사라졌던 원인은 눈금이 값에 **가운데 정렬**돼 넓은 라벨
    (`-2,954`)이 2px 여백에서 잘렸기 때문이다. 안쪽 정렬 주석으로 바꿨으니
    **폭과 무관하게** 두 끝이 남아야 한다.
    """
    rate = placement["rate_pct"].to_numpy(float)
    total = placement["total_delta_min"].to_numpy(float)
    scale = float(np.abs(total).max()) >= 1000

    for values, label in ((rate, D._label_pct),
                          (total, lambda v: D._label_min(v, scale))):
        fig = D._dist_fig(values, float(np.median(values)), label)
        ends = [a for a in fig.layout.annotations if a.yanchor == "top"]
        assert len(ends) == 2, "양 끝 라벨이 둘 다 있어야 한다"
        assert [a.xanchor for a in ends] == ["left", "right"], \
            "안쪽 정렬이 아니면 좁은 여백에서 잘린다"
        assert ends[0].x == pytest.approx(values.min())
        assert ends[1].x == pytest.approx(values.max())
        assert all(a.text.strip() for a in ends)
        # 눈금은 끄고 주석만 쓴다 — 둘 다 켜면 라벨이 겹친다
        assert fig.layout.xaxis.showticklabels is False
        # 라벨이 앉을 아래 여백이 있어야 한다
        assert fig.layout.margin.b >= 14


def test_minute_labels_shrink_but_keep_one_unit():
    """네 자리 분은 천 단위로 줄이고, **양 끝의 단위를 섞지 않는다.**

    한쪽만 천 단위면 폭을 눈으로 견줄 수 없다. 세 자리면 그대로가 더 읽기 쉽다.
    """
    assert D._label_min(-2954.4, True) == "-3.0천분"
    assert D._label_min(-704.8, True) == "-0.7천분"      # 섞지 않는다
    assert D._label_min(-704.8, False) == "-705분"
    # 사이드바 폭 기준 — 축약본이 원본보다 짧아야 의미가 있다
    assert len(D._label_min(-2954.4, True)) < len(D._label_min(-2954.4, False))
    assert D._label_pct(-8.57) == "-8.6%"


def _shown_strings(func_name: str = None) -> list:
    """화면에 나가는 문자열만 — 주석·도크스트링은 뺀다.

    주석은 **왜 그렇게 적었는지**를 남기는 자리라 옛 문구를 인용할 수밖에 없다.
    본문 검색으로 금지어를 찾으면 그 인용이 걸려 정정 기록을 못 남기게 된다.
    """
    import ast
    tree = ast.parse(Path(D.__file__).read_text(encoding="utf-8"))
    if func_name:
        tree = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == func_name)
    docs = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)):
            d = ast.get_docstring(node, clean=False)
            if d:
                docs.add(d)
    return [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and n.value not in docs]


def test_screen_text_has_no_one_sided_bias_wording():
    """화면 문구가 편의 방향을 단정하지 않는다(08.10 정정).

    `BIAS_NOTE` 는 상수로만 두고 실제 화면은 `sim_warnings` 가 낸다 — 상수만
    고치고 화면을 안 고치면 이 검사가 잡는다(실제로 그렇게 빠뜨렸었다).
    """
    shown = "".join(_shown_strings("sim_warnings"))
    assert "양방향 편의" in shown
    assert "후보 간 비교로 읽으십시오" in shown
    for banned in ("실제보다 작게", "과소평가", "더 클 가능성"):
        assert banned not in shown, f"화면에 방향을 단정하는 표현이 남았다: {banned}"


def test_scenario_text_carries_the_location_only_claim():
    """전제 블록의 두 번째 문장 — '후보별 차이는 위치에서만 나온다'.

    이게 "증차해서 좋아진 것 아니냐"에 답하는 자리다. 빠지면 전제 블록이
    조건 나열로만 남는다.
    """
    shown = "".join(_shown_strings("sim_scenario"))
    assert "위치에서만 나옵니다" in shown
    assert "그대로입니다" in shown
    # 주말 단서는 본문이 아니라 물음표 도움말로 접는다 — 본문에 두면 길다
    assert "주말" not in shown
    assert "주말" in D.SCENARIO_HELP
    # **차감 사유는 화면 어디에도 없다**(08.10) — 설계 근거는 문서 몫이다.
    # 툴팁은 부연을 접는 자리이지 근거를 옮겨 놓는 자리가 아니다.
    assert "차감" not in D.SCENARIO_HELP and "차감" not in shown


def test_metric_roles_are_body_ink_not_grey():
    """「개선율 — … / 총절감 — …」 은 **본문 색**이어야 한다.

    회색 주석으로 두면 눈이 지나치는데, **두 지표를 함께 봐야 한다**는 것이 좌측
    패널의 요점이다. 목록과 분포 **양쪽 모두** 그래야 한다 — 한쪽만 고치면 같은
    문장이 화면에서 두 무게로 보인다.

    뒤따르는 백분위·부연은 회색 그대로다. 전부 짙게 하면 아무것도 도드라지지 않는다.
    """
    src = Path(D.__file__).read_text(encoding="utf-8")
    for fn in ("candidate_picker", "distribution_panel"):
        shown = _shown_strings(fn)
        assert any("개선율" in s and "—" in s for s in shown), f"{fn}: 역할 표기가 없다"
        body = src.split(f"def {fn}", 1)[1].split("\ndef ", 1)[0]
        assert "role_html(" in body, f"{fn}: 역할 표기가 아직 회색(.note)이다"
        # 회색도 함께 남아야 한다 — 전부 짙게 하면 위계가 사라진다
        assert "note_html(" in body, f"{fn}: 부연까지 본문 색이 됐다"

    # `.role` 이 실제로 본문 잉크색이어야 한다 — 클래스만 있고 회색이면 소용없다
    role_css = D.CSS.split(".role {", 1)[1].split("}", 1)[0]
    assert f"color: {D.INK}" in role_css
    assert D.INK not in D.CSS.split(".note {", 1)[1].split("}", 1)[0]


def test_no_denial_of_a_claim_never_made():
    """"등수가 아닙니다" 같은 부정문을 화면에 두지 않는다(08.10).

    숫자에 등수가 안 붙어 있으면 그것으로 이미 말이 된다 — 아니라고 굳이 적으면
    없는 기대를 만들어 놓고 부정하는 꼴이다. 원칙은 주석과 명세에 남는다.
    """
    shown = "".join(_shown_strings("distribution_panel"))
    assert "등수가 아닙니다" not in shown


def test_no_top_n_wording_on_screen():
    """'상위 N곳' 류 표현이 화면에 없다.

    근접 강화 설정에서 상위권 간격이 4.12 → 0.59%p 로 좁아져, 상위권이 뚜렷하다는
    인상 자체가 배차 가중치에 의존한다(docs/model_flow.md 「민감도」). 주석·
    도크스트링은 근거를 적는 자리라 검사 대상에서 뺀다 — **화면에 나가는 문자열만** 본다.
    """
    for s in _shown_strings():
        assert "상위 " not in s, f"화면 문자열에 '상위 N' 표현이 있다: {s[:60]}"


# ─────────────────────────────────────────────────────────────
# 총 대기 참고 행 (08.12)
# ─────────────────────────────────────────────────────────────

def test_total_wait_minutes_never_appear_without_the_observed_value():
    """분을 내되 **실측을 반드시 같은 자리에** 낸다 (08.12 개정).

    예전 규칙은 "분을 아예 내지 않는다"였다. 분이 진단 화면의 실측 대기와
    견줘진다는 이유였는데, **그 대비가 지금 이 블록의 목적이다** — 시뮬 총 대기
    before 가 실측의 0.30~0.49배라는 것이 매칭 미재현(A-19)의 증거이고 그것이
    채점을 픽업으로 옮긴 근거다.

    규칙을 없앤 것이 아니라 **짝을 채우는 조건으로 바꿨다.** 분만 내고 실측을
    빼면 옛 걱정이 그대로 살아나므로, 둘은 같이 움직여야 한다.
    """
    shown = "".join(_shown_strings("sim_total_wait"))
    assert "총 대기" in shown
    assert "분" in shown, "분 단위가 사라졌다 — 개정 취지가 빠졌다"
    # 실측·배수·근거가 같은 함수 안에 있어야 한다.
    assert "실측" in shown, "분만 내고 실측을 내지 않는다"
    # 가정 번호(A-19)는 08.12 에 뗐다 — 뜻을 나르는 문장이 남아 있는지로 본다
    assert "매칭을 재현하지 않아 짧다" in shown, "짧은 이유가 화면에 없다"
    src = Path(D.__file__).read_text(encoding="utf-8")
    body = src.split("def sim_total_wait", 1)[1].split("\ndef ", 1)[0]
    assert "obs_total_min" in body, "실측 열을 읽지 않는다"


def test_total_wait_falls_back_to_percent_when_observed_is_missing(monkeypatch):
    """실측이 없으면 분을 내지 않고 개선율 한 줄로 돌아간다.

    분만 있고 실측이 없는 화면은 옛 규칙이 막으려던 바로 그 상태다 — 값이
    서울 평균 42.7분과 곧장 견줘진다. 옛 `placement_grades.csv` 로도 3페이지가
    떠야 하므로 죽지 않고 **줄어드는** 쪽으로 처리한다.
    """
    import pandas as pd
    shown = []
    monkeypatch.setattr(D.st, "markdown", lambda *a, **k: shown.append(a[0]))
    D.sim_total_wait(pd.Series({
        "rate_total_pct": -10.99, "rate_total_sd": 0.5,
        "before": 13.24, "after": 12.14, "rate_pct": -8.57,
        "before_total": 18.67, "after_total": 16.62,
        "obs_total_min": float("nan"), "sim_obs_ratio": float("nan")}))
    body = "".join(shown)
    assert "총 대기 개선율" in body, "개선율 한 줄로 돌아가지 않았다"
    assert "18.67" not in body and "16.62" not in body, \
        "실측이 없는데 분을 냈다"


def test_total_wait_caveat_is_on_screen_not_in_the_tooltip():
    """단서는 툴팁이 아니라 값 옆에 적는다.

    **툴팁은 부연을 접는 자리이지 근거를 옮겨 놓는 자리가 아니다**(명세 3절).
    숫자를 밖으로 인용할 때 반드시 따라가야 하는 말이라 접으면 안 된다.
    """
    # **어느 문자열이 나르는지는 묻지 않는다 — 화면에 있는지를 묻는다.**
    # A-19 는 실측 대비 줄이 수치와 함께 말하고(08.12), 인용 경고는 마지막
    # 문단이 맡는다. 예전에는 둘 다 TOTAL_WAIT_NOTE 한 문장에 있었는데 바로 위
    # 줄과 같은 말을 반복해 정작 지켜야 할 경고가 묻혔다.
    # **가정 번호가 아니라 뜻이 화면에 있는지를 본다.** 「(A-19)」 는 08.12 에
    # 뗐다 — 화면은 실무자가 보는 자리이고 번호는 문서·코드에 남아 추적이
    # 끊기지 않는다. 지켜야 할 것은 **문장이 숫자를 따라다니는 것**이다.
    shown = "".join(_shown_strings("sim_total_wait"))
    assert "매칭을 재현하지 않아 짧다" in shown, "매칭 미재현 단서가 화면에 없다"
    assert "실측" in shown, "대비 근거가 화면에 없다"
    assert "후보 간 비교" in D.TOTAL_WAIT_NOTE, "인용 경고가 사라졌다"
    assert D.TOTAL_WAIT_NOTE not in D.TOTAL_WAIT_HELP, "경고를 툴팁으로 옮겼다"
    # 물음표에는 정의와 "정렬 기준은 픽업"만 들어간다
    assert "픽업" in D.TOTAL_WAIT_HELP

    src = Path(D.__file__).read_text(encoding="utf-8")
    body = src.split("def sim_total_wait", 1)[1].split("\ndef ", 1)[0]
    assert "note_html(TOTAL_WAIT_NOTE)" in body, "단서가 본문에 나가지 않는다"
    assert 'title="{TOTAL_WAIT_HELP}"' in body or "TOTAL_WAIT_HELP" in body


def test_total_wait_is_not_a_metric_card():
    """카드로 만들지 않는다 — 판정 지표와 같은 무게로 읽힌다.

    이 값은 순위 기준이 아니다. 목록 정렬도 분포 패널도 픽업 그대로다
    (docs/evaluation.md 「총 대기도 함께 재되 참고 열이다」).
    """
    src = Path(D.__file__).read_text(encoding="utf-8")
    body = src.split("def sim_total_wait", 1)[1].split("\ndef ", 1)[0]
    assert ".metric(" not in body, "총 대기를 st.metric 으로 냈다"
    # 「참고」 배지는 08.12 에 뗐다. 순위 기준이 아니라는 것은 이제 아래 두 줄과
    # 물음표가 말한다 — **그 말이 화면에 남아 있는지**로 본다.
    shown = "".join(_shown_strings("sim_total_wait"))
    assert "인용하지 말고" in shown or "인용" in D.TOTAL_WAIT_NOTE, \
        "이 값이 인용 대상이 아니라는 말이 화면에서 사라졌다"
    assert "정렬" in D.TOTAL_WAIT_HELP or "픽업" in D.TOTAL_WAIT_HELP, \
        "순위 기준이 픽업이라는 안내가 없다"

    # 정렬·분포는 픽업 열만 본다
    for fn in ("candidate_options", "distribution_panel"):
        fbody = src.split(f"def {fn}", 1)[1].split("\ndef ", 1)[0]
        assert "rate_total" not in fbody, f"{fn} 이 총 대기를 기준으로 쓴다"


def test_before_after_is_not_shown_twice():
    """`before → after` 는 대비 블록에만 둔다 — 카드 넷째를 뺐다 (08.12).

    같은 숫자가 카드와 블록에 두 번 뜨면 블록의 대비 효과가 약해진다.
    before/after 는 픽업과 총 대기를 나란히 놓을 때 의미가 있는 값이라
    그쪽이 담당한다. 판정 카드는 셋(개선율 · 대비 구간 · 총절감)이다.
    """
    src = Path(D.__file__).read_text(encoding="utf-8")
    body = src.split("def render_sim_page", 1)[1].split("\ndef ", 1)[0]
    assert "st.columns(3)" in body, "판정 카드가 셋이 아니다"

    # **화면에 나가는 문자열만 본다.** 주석은 카드를 왜 뺐는지 적는 자리라 옛
    # 라벨을 인용할 수밖에 없다 — 본문 검색으로 찾으면 그 인용이 걸린다.
    shown = "".join(_shown_strings("render_sim_page"))
    assert "before → after" not in shown, "카드에 before → after 가 남아 있다"
    for label in ("픽업 개선율", "대비 구간", "총 절감"):
        assert label in shown, f"판정 카드 「{label}」 이 사라졌다"


def test_comparison_block_is_one_markdown_call_in_a_panel(monkeypatch):
    """테두리 블록을 **한 번의 markdown** 으로 낸다 (명세 8절).

    여는 태그만 따로 부르면 Streamlit 이 그 호출을 독립 요소로 감싸 빈 사각형이
    생긴다. 안에 위젯이 없어야 묶을 수 있는데, σ 물음표는 위젯이 아니라 `title`
    속성이라 문자열로 들어간다.

    실측이 없어 개선율 한 줄로 돌아갈 때도 테두리는 같이 두른다 — 패널 안에서
    이 블록만 테두리가 없으면 딸려 온 글로 보인다.
    """
    import pandas as pd
    full = pd.Series({
        "rate_total_pct": -10.99, "rate_total_sd": 1.31,
        "before": 13.17, "after": 12.04, "rate_pct": -8.57,
        "before_total": 18.67, "after_total": 16.62,
        "obs_total_min": 42.27, "sim_obs_ratio": 0.44})
    thin = full.copy()
    thin["obs_total_min"] = float("nan")

    for name, row in (("분+실측", full), ("개선율만", thin)):
        calls = []
        monkeypatch.setattr(D.st, "markdown", lambda *a, **k: calls.append(a[0]))
        D.sim_total_wait(row)
        assert len(calls) == 1, f"{name}: markdown 을 {len(calls)}번 불렀다"
        html = calls[0]
        assert html.startswith('<div class="panel">'), f"{name}: 테두리가 없다"
        assert html.count("<div") == html.count("</div>"), \
            f"{name}: div 가 짝이 안 맞아 빈 사각형이 생긴다"
        # 위젯이 아니라 title 속성으로 들어가야 한 덩이로 묶인다
        assert "qmark" in html and "title=" in html


def test_comparison_block_sits_in_the_right_panel():
    """대비 블록은 **우측 패널**에서 지도 옆에 선다 (08.12).

    본문에 두면 전체 폭을 쓰면서 지도를 아래로 밀어 결과와 지도를 한눈에 볼 수
    없다. 패널 순서는 대비 블록 → 담당 동 → 경고다.
    """
    src = Path(D.__file__).read_text(encoding="utf-8")
    body = src.split("def render_sim_page", 1)[1].split("\ndef ", 1)[0]
    head, panel = body.split("with right:", 1)
    assert "sim_total_wait(row)" not in head, "대비 블록이 아직 본문에 있다"
    for fn in ("sim_total_wait", "sim_dong_panel", "sim_warnings"):
        assert f"{fn}(" in panel, f"{fn} 이 우측 패널에 없다"
    order = [panel.index(f"{fn}(") for fn in
             ("sim_total_wait", "sim_dong_panel", "sim_warnings")]
    assert order == sorted(order), "패널 순서가 블록 → 담당 동 → 경고 가 아니다"


def test_comparison_block_keeps_its_columns_when_narrow():
    """좁은 폭에서도 두 값 줄이 같은 격자에 선다 — 정렬이 이 블록의 전부다.

    값 칸은 줄바꿈을 막고(`nowrap`), 자리가 모자라면 줄을 깨는 대신 글자를
    줄인다. σ·실측 줄만 격자 전체를 써서 아래로 흐른다.
    """
    src = Path(D.__file__).read_text(encoding="utf-8")
    css = src.split(".wcmp", 1)[1].split("/* 물음표", 1)[0]
    assert "white-space: nowrap" in css, "값 칸이 꺾일 수 있다"
    assert "tabular-nums" in css, "자릿수 폭이 고정되지 않는다"
    assert "grid-column: 1 / -1" in css, "σ·실측 줄이 격자 전체를 쓰지 않는다"
    assert "@media" in css and "font-size" in css, "좁은 폭 대비가 없다"
    # 값이 라벨보다 커야 위계가 산다 — 좁은 폭으로 줄어든 뒤에도 그렇다
    import re
    sizes = [float(m) for m in re.findall(r"font-size: ([\d.]+)rem", css)]
    assert sizes, "글자 크기가 지정돼 있지 않다"
    # 지도와 패널 비율 — 좁히면 값 줄이 꺾인다
    body = src.split("def render_sim_page", 1)[1].split("\ndef ", 1)[0]
    assert "st.columns([2, 1]" in body, "패널이 좁아져 정렬이 무너진다"


def test_total_wait_row_is_silent_without_data(monkeypatch):
    """열이 없거나 비어 있으면 아무것도 그리지 않는다.

    옛 `placement_grades.csv`(총 대기 열 이전)로도 3페이지가 떠야 한다.
    """
    import pandas as pd
    calls = []
    monkeypatch.setattr(D.st, "markdown", lambda *a, **k: calls.append(a))
    D.sim_total_wait(pd.Series({"rate_pct": -8.5}))
    D.sim_total_wait(pd.Series({"rate_total_pct": np.nan, "rate_total_sd": np.nan}))
    assert not calls, "데이터가 없는데 빈 행을 그렸다"
