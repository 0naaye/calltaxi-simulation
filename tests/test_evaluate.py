# -*- coding: utf-8 -*-
"""후보 평가 점검 — 이어 돌리기와 대비 구간.

이 둘은 **조용히 틀리면 잡히지 않는다.** 이어 돌리기가 깨지면 후보 몇 개가
말없이 빠진 채 집계되고, 구간 규칙이 어긋나면 구분되지 않는 후보가 갈라져
실무자가 근거 없는 선택을 하게 된다.

등급 규칙(`assign_grades` 안)은 남아 있고 여기서도 계속 검사한다 — **다만
보고와 CSV 에는 나오지 않는다.** 그 경계도 함께 지킨다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import evaluate as E   # noqa: E402


def _rows(spec, stage=2, calls=4000):
    """(cand_id, [시드별 개선율]) → 결과 CSV 형식 행들.

    총 대기 열도 채운다 — 비워 두면 `run_stage` 가 「열이 늘기 전 행」으로 보고
    다시 돌린다(그 규칙은 아래 `test_rows_without_total_wait_are_rerun`).
    """
    out = []
    for cid, rates in spec:
        for seed, r in enumerate(rates, start=42):
            # 총 대기는 픽업의 2배 수준에 개선율은 1.5배 — 실제 관계(1.3~2.2배)를
            # 흉내 낸 값이다. 판정에 쓰이지 않으므로 크기 자체는 중요하지 않다.
            rt = r * 1.5
            out.append({"stage": stage, "cand_id": cid, "cand_name": f"c{cid}",
                        "gu": "종로구", "dongs": "종로구 사직동", "capacity": 10,
                        "seed": seed, "n_dong_scope": 20, "n_calls_scope": calls,
                        "before": 14.0, "after": 14.0 * (1 + r / 100),
                        "delta": 14.0 * r / 100, "rate_pct": r,
                        "before_total": 30.0, "after_total": 30.0 * (1 + rt / 100),
                        "delta_total": 30.0 * rt / 100, "rate_total_pct": rt,
                        "n_boarded_before": 3800, "n_boarded_after": 3820,
                        "elapsed_s": 50.0})
    return pd.DataFrame(out, columns=E._COLUMNS)


# ─────────────────────────────────────────────────────────────
# 이어 돌리기
# ─────────────────────────────────────────────────────────────

def test_append_and_resume(tmp_path):
    """덧붙이기 → 다시 읽기. 헤더가 한 번만 붙어야 한다."""
    p = tmp_path / "eval.csv"
    assert E.load_results(p).empty

    for i in (1, 2, 3):
        E._append({c: (i if c == "cand_id" else 0) for c in E._COLUMNS}, p)
    got = E.load_results(p)
    assert len(got) == 3
    assert list(got["cand_id"]) == [1, 2, 3]
    assert list(got.columns) == E._COLUMNS


def test_resume_skips_done_pairs(tmp_path, monkeypatch):
    """이미 있는 (cand_id, seed) 는 다시 돌리지 않는다."""
    p = tmp_path / "eval.csv"
    _rows([(1, [-3.0]), (2, [-1.0])], stage=1).to_csv(
        p, index=False, encoding="utf-8-sig")

    cands = pd.DataFrame({"cand_id": [1, 2, 3], "cand_name": ["a", "b", "c"],
                          "lat": [37.5] * 3, "lon": [127.0] * 3,
                          "capacity": [10] * 3, "gu": ["종로구"] * 3,
                          "dongs": ["x"] * 3, "n_dong_assigned": [1] * 3})

    ran = []
    monkeypatch.setattr(E.S, "run_placement",
                        lambda pl, *a, **k: ran.append(pl) or pd.DataFrame())
    monkeypatch.setattr(E, "scope_members", lambda *a, **k: {("종로구", "사직동")})
    monkeypatch.setattr(E, "scope_pickup", lambda log, m: (14.0, 4000))
    monkeypatch.setattr(E, "scope_wait_total", lambda log, m: (30.0, 3800))

    E.run_stage(1, cands, (42,), calls=None, depots=None, mx=None,
                reservation=None, dong_xy=None, path=p)
    # 후보 3만 남았어야 한다 — before(None) 1회 + 후보 1회
    assert ran == [None, 3], f"이미 끝난 후보를 다시 돌렸다: {ran}"
    assert len(E.load_results(p)) == 3


def test_rows_without_total_wait_are_rerun(tmp_path, monkeypatch):
    """**총 대기 열이 없던 시절의 행은 안 끝난 것으로 본다.**

    건너뛰면 그 (후보, 시드) 만 총 대기가 NaN 인 채 집계로 들어간다 — 화면에는
    「참고」 행이 후보에 따라 비거나 시드 2개짜리 σ 가 나온다. 조용히 틀리는
    쪽이라 규칙으로 막는다.
    """
    p = tmp_path / "eval.csv"
    old = _rows([(1, [-3.0]), (2, [-1.0])], stage=1)
    old = old.drop(columns=["before_total", "after_total", "delta_total",
                            "rate_total_pct", "n_boarded_before",
                            "n_boarded_after"])
    old.to_csv(p, index=False, encoding="utf-8-sig")

    cands = pd.DataFrame({"cand_id": [1, 2], "cand_name": ["a", "b"],
                          "lat": [37.5] * 2, "lon": [127.0] * 2,
                          "capacity": [10] * 2, "gu": ["종로구"] * 2,
                          "dongs": ["x"] * 2, "n_dong_assigned": [1] * 2})
    ran = []
    monkeypatch.setattr(E.S, "run_placement",
                        lambda pl, *a, **k: ran.append(pl) or pd.DataFrame())
    monkeypatch.setattr(E, "scope_members", lambda *a, **k: {("종로구", "사직동")})
    monkeypatch.setattr(E, "scope_pickup", lambda log, m: (14.0, 4000))
    monkeypatch.setattr(E, "scope_wait_total", lambda log, m: (30.0, 3800))

    E.run_stage(1, cands, (42,), calls=None, depots=None, mx=None,
                reservation=None, dong_xy=None, path=p)
    assert ran == [None, 1, 2], f"옛 행을 끝난 것으로 봤다: {ran}"

    # 같은 (후보, 시드) 가 옛 행·새 행으로 둘씩 남는다.
    got = E.load_results(p)
    assert len(got) == 4
    assert list(got.columns) == E._COLUMNS, "옛 파일에 열을 맞춰 놓지 않았다"


def test_grades_prefer_rows_that_have_total_wait():
    """옛 행과 새 행이 같은 (후보, 시드) 로 남으면 **새 행**을 집는다.

    옛 행이 dedup 에서 이기면 총 대기가 통째로 NaN 이 되어 화면의 「참고」 행이
    말없이 사라진다.
    """
    new = _rows([(1, [-5.0, -5.1]), (2, [-1.0, -1.1])])
    old = new.copy()
    old[["before_total", "after_total", "delta_total", "rate_total_pct"]] = np.nan
    mixed = pd.concat([old, new], ignore_index=True)   # 옛 행이 앞에 온다

    g = E.assign_grades(mixed)
    assert g["rate_total_pct"].notna().all(), "옛 행이 dedup 에서 이겼다"
    # 값도 새 행 그대로여야 한다 — 픽업의 1.5배로 만들어 둔 값이다(_rows)
    assert g["rate_total_pct"].round(3).tolist() == \
        (g["rate_pct"] * 1.5).round(3).tolist()


def test_parse_seeds():
    assert E.parse_seeds("42,43") == (42, 43)
    assert E.parse_seeds(" 44 ") == (44,)
    assert E.parse_seeds(None) == ()
    assert E.parse_seeds("") == ()


def test_seed_is_the_outer_loop(tmp_path, monkeypatch):
    """시드 하나로 후보 전부를 돈 뒤 다음 시드로 넘어가야 한다.

    후보별로 시드를 다 돌면 중간에 멈췄을 때 앞쪽 후보만 반복이 쌓이고 뒤쪽은
    비어 σ 를 낼 수 없다 — 비교할 수 없는 결과가 남는다.
    """
    p = tmp_path / "eval.csv"
    cands = pd.DataFrame({"cand_id": [1, 2, 3], "cand_name": ["a", "b", "c"],
                          "lat": [37.5] * 3, "lon": [127.0] * 3,
                          "capacity": [10] * 3, "gu": ["종로구"] * 3,
                          "dongs": ["x"] * 3, "n_dong_assigned": [1] * 3})
    monkeypatch.setattr(E.S, "run_placement", lambda pl, *a, **k: pd.DataFrame())
    monkeypatch.setattr(E, "scope_members", lambda *a, **k: {("종로구", "사직동")})
    monkeypatch.setattr(E, "scope_pickup", lambda log, m: (14.0, 4000))
    monkeypatch.setattr(E, "scope_wait_total", lambda log, m: (30.0, 3800))

    E.run_stage(2, cands, (42, 43), calls=None, depots=None, mx=None,
                reservation=None, dong_xy=None, path=p)
    got = E.load_results(p)
    assert list(got["seed"]) == [42, 42, 42, 43, 43, 43],         f"시드가 바깥 루프가 아니다: {list(zip(got['seed'], got['cand_id']))}"
    assert list(got["cand_id"]) == [1, 2, 3, 1, 2, 3]


def test_shortlist_is_by_improvement_rate():
    """2단계 대상은 개선율(음수가 클수록 좋음) 상위 N."""
    res = _rows([(1, [-5.0]), (2, [-1.0]), (3, [-3.0]), (4, [+0.5])], stage=1)
    assert E.stage2_shortlist(res, 2) == [1, 3]


# ─────────────────────────────────────────────────────────────
# 등급 — σ 기반
# ─────────────────────────────────────────────────────────────

def test_grades_merge_when_gap_below_threshold():
    """차이가 2σ 에 못 미치면 같은 등급으로 묶인다."""
    # σ ≈ 0.1 → 문턱 ≈ 2·√(0.01+0.01) ≈ 0.283. 간격 0.1 은 그보다 좁다.
    res = _rows([(1, [-5.0, -5.1, -4.9]), (2, [-4.9, -5.0, -4.8])])
    g = E.assign_grades(res)
    assert g["grade"].nunique() == 1
    assert set(g["grade"]) == {"단일"}


def test_grades_split_when_gap_exceeds_threshold():
    """차이가 2σ 를 넘으면 등급이 갈린다."""
    res = _rows([(1, [-5.0, -5.1, -4.9]), (2, [-1.0, -1.1, -0.9])])
    g = E.assign_grades(res)
    assert g["grade"].nunique() == 2
    assert list(g["grade"]) == ["상", "하"]
    assert g.loc[0, "cand_id"] == 1, "개선이 큰 후보가 위에 와야 한다"
    assert g.loc[1, "gap_prev"] == pytest.approx(4.0, abs=0.01)


def test_grade_threshold_uses_combined_sigma():
    """문턱은 k·sqrt(σᵢ²+σⱼ²) — 두 후보의 σ 를 합친 값이다."""
    res = _rows([(1, [-5.0, -5.6, -4.4]), (2, [-3.6, -3.7, -3.5])])
    g = E.assign_grades(res)
    sd = g["rate_sd"].to_numpy()
    assert g.loc[1, "threshold_prev"] == pytest.approx(
        E.GRADE_SIGMA_K * np.sqrt(sd[0] ** 2 + sd[1] ** 2), rel=1e-9)


def test_grades_use_stage1_replicate_too():
    """2단계 대상은 1단계에서 돈 시드도 반복으로 센다 — σ 표본을 잃지 않는다."""
    s1 = _rows([(1, [-5.0]), (2, [-1.0])], stage=1)          # seed 42
    s2 = _rows([(1, [-5.1, -4.9]), (2, [-1.1, -0.9])], stage=2)
    s2["seed"] = s2["seed"] + 1                               # seed 43, 44
    g = E.assign_grades(pd.concat([s1, s2], ignore_index=True))
    assert set(g["n_seed"]) == {3}, "1단계 반복이 빠졌다"


def test_seeds_missing_is_per_candidate():
    """**후보별로** 따져야 한다 — "어느 하나라도 있으면 됐다"로 보면 이어 돌리는
    중간에 등급이 매겨진다."""
    # 1 은 42·43·44 를 다 채웠고 2 는 42 뿐이다.
    df = pd.concat([_rows([(1, [-5.0, -5.1, -4.9])]),
                    _rows([(2, [-3.0])])], ignore_index=True)
    assert E.seeds_missing(df, [1], (42, 43, 44)) == []
    assert E.seeds_missing(df, [1, 2], (42, 43, 44)) == [43, 44]
    # 결과에 아예 없는 후보가 섞이면 전부 미완으로 본다.
    assert E.seeds_missing(df, [1, 99], (42, 43, 44)) == [42, 43, 44]


def test_grade_labels_survive_past_z():
    """등급이 26개를 넘어도 이름이 깨지지 않아야 한다 — 244곳이면 가능하다."""
    labels = E._grade_labels(30)
    assert len(labels) == len(set(labels)) == 30
    assert labels[:2] == ["A", "B"] and labels[25] == "Z"
    assert labels[26] == "AA" and labels[27] == "AB"
    assert all(c.isalpha() for lab in labels for c in lab)


def test_grades_need_two_seeds():
    """σ 를 낼 수 없으면 등급을 매기지 않고 멈춘다 — 조용히 넘기면 안 된다."""
    with pytest.raises(ValueError):
        E.assign_grades(_rows([(1, [-5.0]), (2, [-1.0])]))


def test_grades_are_monotone_in_rate():
    """등급은 개선율 순서를 뒤집지 않는다."""
    res = _rows([(i, [-r, -r - 0.05, -r + 0.05])
                 for i, r in enumerate([6.0, 4.0, 2.0, 1.9, 0.2], start=1)])
    g = E.assign_grades(res)
    assert g["rate_pct"].is_monotonic_increasing
    codes = pd.Categorical(g["grade"], categories=list(dict.fromkeys(g["grade"])),
                           ordered=True).codes
    assert (np.diff(codes) >= 0).all(), "등급이 개선율 순서와 어긋난다"


def test_report_carries_the_warnings():
    """보고 문구에 '우열을 가릴 수 없다'와 **양방향 편의** 경고가 있어야 한다.

    편의가 한쪽이라고 적으면 안 된다 — A-19·A-20 은 절감을 줄이는 쪽이고 θ 를
    두지 않은 것(A-01)은 늘리는 쪽이라 상쇄 크기를 모른다(08.10 정정). 그래서
    "과소평가"·"더 클 가능성" 같은 방향을 단정하는 표현이 **없어야** 한다.
    """
    g = E.assign_grades(_rows([(1, [-5.0, -5.1, -4.9]), (2, [-1.0, -1.1, -0.9])]))
    txt = E.candidate_report(g)
    assert "우열을 가릴 수 없다" in txt
    assert "양방향 편의" in txt
    assert "후보 간 비교로만 읽는다" in txt
    assert "과소평가" not in txt
    assert "더 클 가능성" not in txt


# ─────────────────────────────────────────────────────────────
# 등급을 표시하지 않는다 — 규칙은 남기고 표시만 뺀다
# ─────────────────────────────────────────────────────────────

def test_report_never_shows_grades():
    """**보고에 등급이 나오면 안 된다.** 판단을 대신하는 표시라 뺐다.

    244곳에서 [상] 4곳 / [하] 240곳이 나왔고 그 [상] 4곳 중 3곳이 표본
    부족이었다 — 최고 등급이 가장 좁은 지역을 가리켰다. 되살아나면 여기서 잡힌다.
    """
    g = E.assign_grades(_rows([(1, [-5.0, -5.1, -4.9]), (2, [-1.0, -1.1, -0.9])]))
    assert set(g["grade"]) == {"상", "하"}, "등급 계산 자체는 남아 있어야 한다"

    txt = E.candidate_report(g)
    for token in ("[상]", "[하]", "[등급", "등급 1개"):
        assert token not in txt, f"보고에 등급 표시가 남았다: {token}"
    assert "등급은 표시하지 않는다" in txt


def test_csv_frame_demotes_grade_to_reference_columns():
    """CSV 에서 등급 3열은 `ref_` 접두사로 맨 뒤에 있어야 한다.

    **지우지는 않는다** — 사전 고정된 규칙을 결과를 보고 지우면 그 자체가 사후
    선택이다. 재현·감사에 필요하므로 남기되 이름과 위치로 참고용임을 못 박는다.
    """
    g = E.assign_grades(_rows([(1, [-5.0, -5.1, -4.9]), (2, [-1.0, -1.1, -0.9])]))
    cols = list(E.report_frame(g).columns)

    assert cols[-3:] == ["ref_grade", "ref_gap_prev", "ref_threshold_prev"]
    for bare in ("grade", "gap_prev", "threshold_prev"):
        assert bare not in cols, f"맨 이름의 {bare} 열이 CSV 에 남았다"
    # 실무자가 읽어야 할 열은 앞쪽에 그대로 있다.
    for keep in ("rate_pct", "interval", "n_calls_scope", "n_dong_scope",
                 "total_delta_min", "sample_ok", "n_overlap"):
        assert keep in cols


# ─────────────────────────────────────────────────────────────
# 대비 구간 — 등급이 뭉쳐도 읽을 수 있어야 한다
# ─────────────────────────────────────────────────────────────

def test_interval_non_overlap_matches_grade_split():
    """**구간이 겹친다 == 같은 등급이다.** 이게 어긋나면 표와 등급이 따로 논다.

    σ 가 같은 두 후보를 문턱 바로 아래·바로 위로 놓고 양쪽을 확인한다.
    폭 배수를 손대면 여기서 잡힌다.
    """
    sd = 0.1                       # [-r-sd, -r, -r+sd] 로 만드는 표준편차
    thr = E.GRADE_SIGMA_K * np.sqrt(2) * sd

    for factor, want_split in ((0.8, False), (1.2, True)):
        gap = thr * factor
        g = E.assign_grades(_rows([
            (1, [-5.0 - sd, -5.0, -5.0 + sd]),
            (2, [-5.0 + gap - sd, -5.0 + gap, -5.0 + gap + sd]),
        ]))
        split = g["grade"].nunique() == 2
        lo, hi = g.iloc[0], g.iloc[1]
        overlap = lo["rate_hi"] >= hi["rate_lo"]
        assert split == want_split
        assert overlap == (not want_split), "구간 겹침과 등급 분리가 어긋난다"


def test_interval_half_width_is_derived_from_grade_k():
    """반폭은 k/√2·σ 다 — 새 규칙이 아니라 등급 규칙에서 유도한 값."""
    g = E.assign_grades(_rows([(1, [-5.0, -5.1, -4.9]), (2, [-1.0, -1.2, -0.8])]))
    half = (g["rate_hi"] - g["rate_lo"]) / 2
    assert np.allclose(half, E.GRADE_SIGMA_K / np.sqrt(2) * g["rate_sd"])


def test_report_says_how_many_the_top_cannot_be_told_from():
    """구분이 안 될 때 **몇 곳과 겹치는지 명시**해야 한다 — 침묵하면 1위로 읽힌다.

    등급이 하나로 뭉치던 자리를 겹침 수가 대신한다.
    """
    g = E.assign_grades(_rows([(1, [-5.0, -5.4, -4.6]), (2, [-4.9, -5.3, -4.5]),
                               (3, [-4.8, -5.2, -4.4])]))
    assert list(g["n_overlap"]) == [2, 2, 2], "셋이 서로 다 겹쳐야 한다"
    txt = E.candidate_report(g)
    assert "다른 2곳과 구간이 겹친다" in txt
    assert "단독 1위라 말할 수 없다" in txt


def test_n_overlap_counts_only_actual_overlaps():
    """겹침 수는 **자기를 빼고** 실제로 구간이 닿는 후보만 센다."""
    g = E.assign_grades(_rows([(1, [-5.0, -5.1, -4.9]), (2, [-5.05, -5.15, -4.95]),
                               (3, [-1.0, -1.1, -0.9])]))
    n_ov = dict(zip(g["cand_id"], g["n_overlap"]))
    assert n_ov == {1: 1, 2: 1, 3: 0}, "멀리 떨어진 3 이 끼거나 자기를 셌다"
    # `overlaps` 목록 길이와 어긋나면 안 된다 — 표와 숫자가 따로 놀면 못 믿는다.
    for cid, n in n_ov.items():
        assert len(E.overlaps(g, cid)) == n + 1


def test_overlaps_lists_the_indistinguishable():
    """`overlaps` 는 구간이 겹치는 후보를 전부(자기 포함) 돌려준다."""
    g = E.assign_grades(_rows([(1, [-5.0, -5.1, -4.9]), (2, [-5.05, -5.15, -4.95]),
                               (3, [-1.0, -1.1, -0.9])]))
    got = set(E.overlaps(g, 1)["cand_id"])
    assert got == {1, 2}, "멀리 떨어진 3 이 끼거나 가까운 2 가 빠졌다"


# ─────────────────────────────────────────────────────────────
# 표본 부족 표시
# ─────────────────────────────────────────────────────────────

def test_thin_scope_is_flagged_but_not_dropped():
    """콜이 기준 미만이면 표시만 한다. **계산에서 빼면 안 된다.**"""
    thin = _rows([(1, [-9.0, -9.2, -8.8])], calls=E.MIN_CALLS_SCOPE - 1)
    thick = _rows([(2, [-3.0, -3.2, -2.8])], calls=E.MIN_CALLS_SCOPE)
    g = E.assign_grades(pd.concat([thin, thick], ignore_index=True))

    assert len(g) == 2, "표본 부족 후보가 표에서 빠졌다"
    flag = dict(zip(g["cand_id"], g["sample_ok"]))
    assert flag[1] is False or flag[1] == False   # noqa: E712
    assert flag[2] is True or flag[2] == True     # noqa: E712
    assert "표본 부족" in E.candidate_report(g)


def test_report_shows_the_sample_size_bias():
    """**표본 부족이 개선율 상위에 몰린다**는 사실을 보고가 말해야 한다.

    "12곳뿐"으로 읽히면 편향이 숨는다. 전체 비율과 상위권 비율을 나란히 낸다.
    """
    thin = _rows([(1, [-9.0, -9.4, -8.6])], calls=E.MIN_CALLS_SCOPE - 1)
    thick = _rows([(i, [-3.0 - i * 0.1, -3.2 - i * 0.1, -2.8 - i * 0.1])
                   for i in range(2, 6)], calls=E.MIN_CALLS_SCOPE * 3)
    g = E.assign_grades(pd.concat([thin, thick], ignore_index=True))
    txt = E.candidate_report(g)
    assert "개선율 순위는 표본 크기의 함수다" in txt
    assert "vs 정상" in txt


# ─────────────────────────────────────────────────────────────
# 수혜 규모 병기
# ─────────────────────────────────────────────────────────────

def test_total_minutes_is_delta_times_calls():
    """총 절감 분은 임의 가중 복합 지표가 아니라 분/콜 × 콜수 다."""
    g = E.assign_grades(_rows([(1, [-5.0, -5.1, -4.9])], calls=3000))
    assert np.allclose(g["total_delta_min"], g["delta"] * 3000)


def test_narrow_scope_wins_on_rate_but_loses_on_scale():
    """**개선율 1위와 총절감 1위가 갈리는 것을 표가 드러내야 한다.**

    좁은 범위(콜 800)의 큰 개선율 vs 넓은 범위(콜 8000)의 작은 개선율.
    개선율로는 앞이, 총절감으로는 뒤가 이긴다 — 그게 병기하는 이유다.
    """
    narrow = _rows([(1, [-9.0, -9.2, -8.8])], calls=800)
    wide = _rows([(2, [-2.0, -2.2, -1.8])], calls=8000)
    g = E.assign_grades(pd.concat([narrow, wide], ignore_index=True))

    assert g.iloc[0]["cand_id"] == 1, "정렬은 개선율 기준이어야 한다"
    best_total = g.nsmallest(1, "total_delta_min")["cand_id"].iloc[0]
    assert best_total == 2, "총절감으로는 넓은 범위가 앞서야 한다"
    assert dict(zip(g["cand_id"], g["rank_rate"]))[1] == 1
    assert dict(zip(g["cand_id"], g["rank_total"]))[2] == 1


def test_report_shows_scale_beside_rate():
    """후보 한 줄에 **네 가지가 다** 나와야 한다 — 따로 찾게 두면 안 본다.

    개선율 ± 대비 구간 │ 콜 · 동 · 총절감 │ 겹침 수 │ 표본 부족.
    """
    g = E.assign_grades(_rows([(1, [-5.0, -5.1, -4.9]), (2, [-1.0, -1.2, -0.8])],
                              calls=3000))
    txt = E.candidate_report(g)
    assert "콜  3,000" in txt or "콜 3,000" in txt
    assert "±" in txt and "동 20" in txt
    assert "겹침" in txt
    assert "총 절감 분 상위" in txt
    assert "반대 방향으로 오독된다" in txt


# ─────────────────────────────────────────────────────────────
# 범위
# ─────────────────────────────────────────────────────────────

def test_scope_members_uses_radius():
    dong = pd.DataFrame({"gu": ["A", "A", "B"], "dong_canon": ["가", "나", "다"],
                         "lat": [37.5, 37.52, 37.9], "lon": [127.0, 127.0, 127.0]})
    got = E.scope_members(37.5, 127.0, dong, radius_km=3.0)
    assert got == {("A", "가"), ("A", "나")}          # 다동은 44km 밖
    assert E.scope_members(37.5, 127.0, dong, radius_km=0.1) == {("A", "가")}


def test_scope_pickup_is_call_weighted():
    """콜 가중 — 동 균등이 아니다. 큰 동이 더 무겁다."""
    log = pd.DataFrame({
        "origin_gu": ["A"] * 10 + ["A"] * 2 + ["B"] * 5,
        "origin_dong_canon": ["가"] * 10 + ["나"] * 2 + ["다"] * 5,
        "assigned_at": pd.Timestamp("2025-08-20 09:00"),
        "boarded_at": ([pd.Timestamp("2025-08-20 09:10")] * 10
                       + [pd.Timestamp("2025-08-20 09:40")] * 2
                       + [pd.Timestamp("2025-08-20 09:20")] * 5),
    })
    mean, n = E.scope_pickup(log, {("A", "가"), ("A", "나")})
    assert n == 12
    assert mean == pytest.approx((10 * 10 + 2 * 40) / 12)
