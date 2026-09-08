# 3페이지 before/after 후보 지표 — 총 대기 · 대기 중 포기 · 60분 초과
# 대표 후보 3곳(개선율 rank 1 / 중앙 / 최하) × 시드 42·43·44 · 3km 영향권
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
# 저장소 안에서 자기 위치로 루트를 잡는다 — 절대경로를 박아 두면
# clone 한 사람이 못 돌린다.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

import load
import metrics as M
import simulator as S
import travel_time as T
import evaluate as E

SEEDS = E.STAGE2_SEEDS

# ── 후보 선정: 개선율 순위로만 고른다(사전 규칙, 표본 필터 없음) ──
g = pd.read_csv(ROOT / "outputs/placement_grades.csv",
                encoding="utf-8-sig").sort_values("rank_rate")
PICKS = [1, (len(g) + 1) // 2, len(g)]
sel = pd.concat([g[g["rank_rate"] == r] for r in PICKS])
print("=" * 78)
print(f"대표 후보 3곳 — 개선율 rank {PICKS} / 전체 {len(g)}곳")
print("=" * 78)
for _, r in sel.iterrows():
    print(f"  rank {int(r.rank_rate):3d}  id={int(r.cand_id):4d}  {r.cand_name[:28]:<28} "
          f"{r.gu}  픽업 {r.rate_pct:+.2f}% (sd {r.rate_sd:.2f})  "
          f"3km 동 {int(r.n_dong_scope)} · 콜 {int(r.n_calls_scope):,}  "
          f"표본충분={bool(r.sample_ok)}")

print("\n입력 로딩...")
calls = load.load_calls()
depots = load.load_depots()
reservation = load.load_reservation_occupancy()
mx = S.TravelMatrix(T.TravelTime.build(),
                    set(calls["origin_dong_canon"].astype(str))
                    | set(calls["dest_dong_canon"].astype(str)))
sub = S.slice_period(calls, start=S.REPRESENTATIVE_START,
                     days=S.REPRESENTATIVE_DAYS)
print(f"대표 기간 {S.REPRESENTATIVE_START} +{S.REPRESENTATIVE_DAYS}일 · 콜 {len(sub):,}건")

cands = E.candidate_table()
dong_xy = E.dong_coords()


def scope_stats(log: pd.DataFrame, members: set) -> dict:
    """3km 영향권 안 콜의 지표. 분모는 그 범위의 접수 콜 전체다."""
    key = pd.MultiIndex.from_arrays([log["origin_gu"].astype(str),
                                     log["origin_dong_canon"].astype(str)])
    d = log[key.isin(members)]
    w = d["wait_min"]
    kind = pd.Series(M.cancel_kind(d), index=d.index)
    pick = M.pickup_min(d)
    return {
        "n_calls": len(d),
        "n_boarded": int(w.notna().sum()),
        "wait_mean": float(w.mean()),
        "wait_p50": float(w.median()),
        "wait_p90": float(w.quantile(.90)),
        "n_abandoned": int((kind == "abandoned").sum()),
        "n_immediate": int((kind == "immediate").sum()),
        "n_over60": int((w > M.LONG_WAIT_MIN).sum()),
        "max_wait": float(w.max()),
        "pickup_mean": float(pick.mean()),
    }


rows = []
baseline = {}
for seed in SEEDS:
    print(f"\n[seed {seed}] before 재생...")
    baseline[seed] = S.run_placement(None, sub, depots, mx,
                                     reservation=reservation, seed=seed)
    for _, r in sel.iterrows():
        cid = int(r.cand_id)
        c = cands[cands["cand_id"] == cid].iloc[0]
        members = E.scope_members(c.lat, c.lon, dong_xy)
        log = S.run_placement(cid, sub, depots, mx,
                              reservation=reservation, seed=seed)
        b = scope_stats(baseline[seed], members)
        a = scope_stats(log, members)
        print(f"  cand {cid} {r.cand_name[:16]:<16} "
              f"대기 {b['wait_mean']:.2f}→{a['wait_mean']:.2f}  "
              f"포기 {b['n_abandoned']}→{a['n_abandoned']}  "
              f"60분초과 {b['n_over60']}→{a['n_over60']}")
        for k, v in b.items():
            rows.append({"cand_id": cid, "rank": int(r.rank_rate),
                         "name": r.cand_name, "seed": seed,
                         "when": "before", "metric": k, "value": v})
        for k, v in a.items():
            rows.append({"cand_id": cid, "rank": int(r.rank_rate),
                         "name": r.cand_name, "seed": seed,
                         "when": "after", "metric": k, "value": v})

df = pd.DataFrame(rows)
out = (Path(sys.argv[1]) if len(sys.argv) > 1
       else ROOT / "outputs" / "page3_extra_raw.csv")
df.to_csv(out, index=False, encoding="utf-8-sig")

# ── 집계 ──────────────────────────────────────────────
print("\n" + "=" * 78)
print("시드별 원값 · 시드 평균 ± 표준편차")
print("=" * 78)

METRICS = [("wait_mean", "⑴ 총 대기 평균(분)", 2),
           ("n_abandoned", "⑵ 대기 중 포기(건)", 1),
           ("n_over60", "⑶ 60분 초과(건)", 1),
           ("pickup_mean", "(대조) 픽업 평균(분)", 2),
           ("n_calls", "(참고) 범위 콜(건)", 0),
           ("n_boarded", "(참고) 승차(건)", 1),
           ("wait_p90", "(참고) 총 대기 p90(분)", 2),
           ("max_wait", "(참고) 최대 대기(분)", 1)]

for cid in sel["cand_id"].astype(int):
    r = sel[sel["cand_id"] == cid].iloc[0]
    print(f"\n■ rank {int(r.rank_rate)} · cand {cid} · {r.cand_name} ({r.gu})")
    print(f"   3km 동 {int(r.n_dong_scope)} · 표본충분={bool(r.sample_ok)}")
    d = df[df["cand_id"] == cid]
    for key, label, nd in METRICS:
        line = []
        for when in ("before", "after"):
            v = d[(d["metric"] == key) & (d["when"] == when)].set_index("seed")["value"]
            per = " / ".join(f"{v[s]:.{nd}f}" for s in SEEDS)
            line.append((v.mean(), v.std(ddof=1), per))
        (bm, bs, bp), (am, asd, ap) = line
        delta = am - bm
        # 대응표본 — 시드마다 before/after 가 짝이므로 차이의 sd 를 따로 낸다
        vb = d[(d["metric"] == key) & (d["when"] == "before")].set_index("seed")["value"]
        va = d[(d["metric"] == key) & (d["when"] == "after")].set_index("seed")["value"]
        dif = (va - vb)
        pct = delta / bm * 100 if bm else np.nan
        print(f"   {label:<22} before {bm:>8.{nd}f} ±{bs:>6.{nd}f}  [{bp}]")
        print(f"   {'':<22} after  {am:>8.{nd}f} ±{asd:>6.{nd}f}  [{ap}]")
        print(f"   {'':<22} Δ      {delta:>+8.{nd}f} ±{dif.std(ddof=1):>6.{nd}f}"
              f"  ({pct:+.1f}%)  시드별 Δ "
              f"[{' / '.join(f'{dif[s]:+.{nd}f}' for s in SEEDS)}]")

print(f"\n원자료: {out}")
