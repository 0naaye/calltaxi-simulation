# 같은 30일 · 같은 3km 범위의 실측값 — 시뮬 before 가 수준을 재현하는지
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
# 저장소 안에서 자기 위치로 루트를 잡는다 — 절대경로를 박아 두면
# clone 한 사람이 못 돌린다.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

import load
import metrics as M
import simulator as S
import evaluate as E

calls = load.load_calls()
sub = S.slice_period(calls, start=S.REPRESENTATIVE_START, days=S.REPRESENTATIVE_DAYS)
cands = E.candidate_table()
dong_xy = E.dong_coords()

g = pd.read_csv(ROOT / "outputs/placement_grades.csv", encoding="utf-8-sig")
PICKS = {1: None, 122: None, 244: None}


def stats(d: pd.DataFrame) -> dict:
    w = d["wait_min"]
    kind = pd.Series(M.cancel_kind(d), index=d.index)
    return {
        "n_calls": len(d),
        "n_boarded": int(w.notna().sum()),
        "wait_mean": float(w.mean()),
        "wait_p90": float(w.quantile(.90)),
        "n_abandoned": int((kind == "abandoned").sum()),
        "abandoned_pct": float((kind == "abandoned").sum()) / len(d) * 100,
        "n_over60": int((w > M.LONG_WAIT_MIN).sum()),
        "over60_pct": float((w > M.LONG_WAIT_MIN).mean() * 100),
    }


print("=" * 76)
print(f"실측 · 대표 기간 {S.REPRESENTATIVE_START} +{S.REPRESENTATIVE_DAYS}일 "
      f"(콜 {len(sub):,}건)")
print("=" * 76)
print("\n[서울 전체]")
a = stats(sub)
for k, v in a.items():
    print(f"   {k:<16} {v:,.2f}" if isinstance(v, float) else f"   {k:<16} {v:,}")

# 시뮬 before 의 시드 평균(앞 실행 결과)
SIM_BEFORE = {
    411: {"wait_mean": 18.67, "n_abandoned": 45.0, "n_over60": 66.3,
          "n_calls": 1961, "n_boarded": 1786.0},
    240: {"wait_mean": 18.11, "n_abandoned": 50.3, "n_over60": 28.0,
          "n_calls": 3426, "n_boarded": 3144.3},
    598: {"wait_mean": 15.29, "n_abandoned": 66.3, "n_over60": 33.7,
          "n_calls": 6148, "n_boarded": 5636.3},
}

for rank in PICKS:
    r = g[g["rank_rate"] == rank].iloc[0]
    cid = int(r.cand_id)
    c = cands[cands["cand_id"] == cid].iloc[0]
    members = E.scope_members(c.lat, c.lon, dong_xy)
    key = pd.MultiIndex.from_arrays([sub["origin_gu"].astype(str),
                                     sub["origin_dong_canon"].astype(str)])
    d = sub[key.isin(members)]
    s = stats(d)
    sim = SIM_BEFORE[cid]
    print(f"\n[rank {rank} · cand {cid} · {r.cand_name} ({r.gu})] 3km 동 {len(members)}")
    print(f"   {'지표':<18} {'실측':>10} {'시뮬 before':>12} {'시뮬/실측':>10}")
    for key_, label in [("n_calls", "범위 접수(건)"), ("n_boarded", "승차(건)"),
                        ("wait_mean", "총 대기 평균(분)"),
                        ("n_abandoned", "대기 중 포기(건)"),
                        ("n_over60", "60분 초과(건)")]:
        av, sv = s[key_], sim[key_]
        ratio = sv / av if av else float("nan")
        print(f"   {label:<18} {av:>10,.1f} {sv:>12,.1f} {ratio:>9.2f}배")
    print(f"   {'포기 비율(%)':<18} {s['abandoned_pct']:>10.2f} "
          f"{sim['n_abandoned']/sim['n_calls']*100:>12.2f}")
    print(f"   {'60분 초과 비율(%)':<18} {s['over60_pct']:>10.2f} "
          f"{sim['n_over60']/sim['n_boarded']*100:>12.2f}")
