# -*- coding: utf-8 -*-
"""민감도 점검 — **대상 선정 규칙이 규칙대로 도는가**만 본다.

여기서 막는 것은 사후 선택이다. 9곳이 결과를 보고 고른 것이 아니라 규칙에서
나왔다는 사실은 `select_candidates` 가 표본 부족을 빼고 상·중·하를 정해진 자리에서
뽑는지로만 보장된다. 값 자체(개선율이 얼마인가)는 여기서 판정하지 않는다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import sensitivity as SN  # noqa: E402


def _graded(n_ok=20, n_weak=4):
    """집계표 흉내 — 개선율이 −10 부터 고르게 깔린 후보 표."""
    n = n_ok + n_weak
    return pd.DataFrame({
        "cand_id": np.arange(1, n + 1),
        "cand_name": [f"c{i}" for i in range(n)],
        "rate_pct": np.linspace(-10.0, -1.0, n),
        # 표본 부족을 **개선율 상위 쪽에 박아 둔다** — 실제로 그렇게 몰려 있고
        # (12곳 중 6곳이 상위 10), 안 빼면 상위 밴드가 통째로 그쪽이 된다
        "sample_ok": [False] * n_weak + [True] * n_ok,
    })


def test_selection_excludes_weak_samples():
    """표본 부족은 상위에 몰려 있어도 뽑히지 않는다."""
    g = _graded()
    sel = SN.select_candidates(g.assign(**{"lat": 0.0, "lon": 0.0}), n=3)
    assert len(sel) == 9
    assert sel["sample_ok"].all(), "표본 부족이 대상에 섞였다"


def test_selection_takes_top_middle_bottom():
    """상 3 · 중 3 · 하 3 이 정렬 위치에서 나온다 — 값을 보고 고르지 않는다."""
    g = _graded(n_ok=232, n_weak=12)
    sel = SN.select_candidates(g.assign(**{"lat": 0.0, "lon": 0.0}), n=3)
    ok = g[g["sample_ok"]].sort_values("rate_pct").reset_index(drop=True)

    assert sel["band"].tolist() == ["상"] * 3 + ["중"] * 3 + ["하"] * 3
    assert sel.loc[:2, "cand_id"].tolist() == ok.loc[:2, "cand_id"].tolist()
    assert sel.loc[6:, "cand_id"].tolist() == ok.loc[229:, "cand_id"].tolist()
    # 중앙 3곳은 중앙 색인을 가운데 둔다
    mid = len(ok) // 2
    assert sel.loc[3:5, "cand_id"].tolist() == \
        ok.loc[mid - 1:mid + 1, "cand_id"].tolist()


def test_selection_refuses_when_too_few():
    """뽑을 수 없으면 조용히 줄이지 않고 튕긴다."""
    with pytest.raises(ValueError):
        SN.select_candidates(_graded(n_ok=5, n_weak=2), n=3)


def test_config_weights_keep_the_15_35_split():
    """대기쪽 안의 15:35 비율은 설정을 바꿔도 유지된다(원문 구조 보존)."""
    for key in ("base", "near_first", "wait_first"):
        kw = SN.CONFIG_BY_KEY[key][2]
        w = kw.get("score_w", __import__("simulator").SCORE_W)
        assert w["order"] / (w["order"] + w["wait"]) == pytest.approx(0.30)


def test_config_axis_is_wait_vs_near():
    """세 설정이 `대기 : 근접` 축 위의 50:40 / 40:50 / 60:30 이어야 한다."""
    import simulator as S
    got = {}
    for key in ("base", "near_first", "wait_first"):
        w = SN.CONFIG_BY_KEY[key][2].get("score_w", S.SCORE_W)
        got[key] = (w["order"] + w["wait"], w["dist"])
    assert got == {"base": (50.0, 40.0), "near_first": (40.0, 50),
                   "wait_first": (60.0, 30)}


def test_theta_configs_build_idle_break():
    """θ 설정은 `IdleBreak` 로 감싸져 `run_placement` 인자가 된다."""
    gaps = np.concatenate([np.full(5000, 5.0), np.linspace(60, 300, 5000)])
    assert SN._kwargs({}, gaps) == {}
    for key, theta in (("theta120", 120.0), ("theta60", 60.0)):
        kw = SN._kwargs(SN.CONFIG_BY_KEY[key][2], gaps)
        assert "theta" not in kw, "θ 는 IdleBreak 로 바뀌어 나가야 한다"
        assert kw["idle_break"].theta == theta
