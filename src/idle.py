# -*- coding: utf-8 -*-
"""idle.py — 차량 유휴 구간 산출

실측 운행기록에서 차량별 유휴 구간을 계산한다.

정의
유휴 구간 = 하차일시 ~ 같은 차량의 다음 배차일시

```
배차 이후는 다음 콜에 배정된 상태이므로 유휴에서 제외한다.
이 값은 '유휴 시간'을 의미하며 실제 배차 가능한 가용 시간과는 다를 수 있다.
```

입력
calls_2025_replay.csv (특장차)
- 승차·하차·배차 시각과 차량번호가 있는 완료 운행
- 운행시간 길이에 따른 추가 필터는 적용하지 않음

주의
차량번호가 없는 일부 정산 미기록 건은 차량 시퀀스에 포함할 수 없다.
시간대별 동시 유휴는 자정을 넘지 않는 구간만 사용한다.
긴 유휴 구간에는 식사·교대·충전 등이 포함될 수 있어 CAP_MINUTES로 민감도를 확인한다.

산출
- 유휴 구간 길이 분포
- 평일/주말 시간대별 동시 유휴 차량 수
- outputs/idle_gaps.csv
- outputs/idle_by_hour.csv

실행
.venv/Scripts/python.exe src/idle.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parent.parent          # simulation/
DATA_DIR = (Path(os.environ.get("DATA_DIR", "").strip()).expanduser().resolve()
            if os.environ.get("DATA_DIR", "").strip() else REPO.parent / "data")
OUT_DIR = REPO / "outputs"

SRC = DATA_DIR / "calls_2025_replay.csv"        # load.CALLS_FILE 과 같은 파일
USECOLS = ["차량번호", "배차일시", "승차일시", "하차일시"]

MINUTES_PER_DAY = 1440

# 민감도용 상한(분). 이보다 긴 당일 구간은 식사·교대·근무외가 섞여 있을 가능성이
# 커서, 잘랐을 때 동시 유휴가 얼마나 줄어드는지를 함께 낸다. **분포 산출에서
# 실제로 자르지는 않는다** — 임계 θ 는 A-01 잔여 조치로 아직 정해지지 않았다.
CAP_MINUTES = 120

# 동시 유휴 대수의 분모. 차고지 정원(691대)이나 연간 누적 차량번호(723대)가 아니라
# **그 날 실제로 나온 대수**여야 한다 — metrics.daily_active_vehicles() 의 요일별
# 중앙값이다(평일 574 / 주말 374, 접수일 기준).
FLEET_WEEKDAY, FLEET_WEEKEND = 574, 374
DEPOT_CAPACITY = 691


def load_trips(path: Path = SRC) -> pd.DataFrame:
    """운행 기록 로딩·필터. 단계별 잔존 건수는 attrs['funnel'] 에 담는다."""
    df = pd.read_csv(path, encoding="utf-8-sig", usecols=USECOLS, low_memory=False)
    funnel = {"원본": len(df)}

    assign = pd.to_datetime(df["배차일시"], format="ISO8601", errors="coerce")
    board = pd.to_datetime(df["승차일시"], format="ISO8601", errors="coerce")
    alight = pd.to_datetime(df["하차일시"], format="ISO8601", errors="coerce")

    ok = board.notna() & alight.notna()
    assign, board, alight, df = assign[ok], board[ok], alight[ok], df[ok]
    funnel["승차·하차 기록"] = len(df)

    ok = assign.notna()
    assign, board, alight, df = assign[ok], board[ok], alight[ok], df[ok]
    funnel["배차 기록"] = len(df)

    vehicle = pd.to_numeric(df["차량번호"], errors="coerce")
    ok = vehicle.notna()
    funnel["차량번호 있음"] = int(ok.sum())

    out = pd.DataFrame({
        "vehicle_id": vehicle[ok].astype("int64").to_numpy(),
        "assigned_at": assign[ok].to_numpy(),
        "alighted_at": alight[ok].to_numpy(),
    })
    out.attrs["funnel"] = funnel
    return out.reset_index(drop=True)


def build_gaps(trips: pd.DataFrame) -> pd.DataFrame:
    """차량별 시간순 정렬 후 (하차 → 다음 배차) 구간을 만든다.

    마지막 운행은 다음 배차가 없어 구간이 만들어지지 않는다 — 차량 수만큼 줄어든다.
    반환: vehicle_id, start(하차), end(다음 배차), idle_min, is_same_day
    """
    t = trips.sort_values(["vehicle_id", "assigned_at"], kind="stable")
    nxt = t.groupby("vehicle_id", sort=False)["assigned_at"].shift(-1)

    gaps = pd.DataFrame({
        "vehicle_id": t["vehicle_id"].to_numpy(),
        "start": t["alighted_at"].to_numpy(),
        "end": nxt.to_numpy(),
    }).dropna(subset=["end"]).reset_index(drop=True)

    gaps["idle_min"] = (gaps["end"] - gaps["start"]).dt.total_seconds() / 60
    gaps["is_same_day"] = gaps["start"].dt.normalize() == gaps["end"].dt.normalize()
    return gaps


def hourly_concurrency(gaps: pd.DataFrame, n_days: int) -> pd.DataFrame:
    """시간대(0~23)별 평균 동시 유휴 차량 수.

    구간 [start, end) 를 하루 24개 시간 칸에 잘라 넣고, 칸별 총 유휴 분을
    (60분 × 일수) 로 나눈다. 그 시간대에 평균 몇 대가 동시에 비어 있었나가 된다.

        f(t, h) = ⌊t/1440⌋ × 60 + clip(t mod 1440 − 60h, 0, 60)
        구간의 h 칸 점유 = f(end, h) − f(start, h)

    누적분포 f 의 차분이라 자정을 넘는 구간도 자동으로 갈라진다.
    """
    epoch = gaps["start"].min().normalize()
    s = (gaps["start"] - epoch).dt.total_seconds().to_numpy() / 60
    e = (gaps["end"] - epoch).dt.total_seconds().to_numpy() / 60

    def occupied(t, h):
        whole, rem = np.divmod(t, MINUTES_PER_DAY)
        return whole * 60 + np.clip(rem - 60 * h, 0, 60)

    rows = []
    for h in range(24):
        minutes = float((occupied(e, h) - occupied(s, h)).sum())
        rows.append({"hour": h, "idle_min_total": minutes,
                     "mean_idle_vehicles": minutes / (60 * n_days)})
    return pd.DataFrame(rows)


def main():
    if not SRC.exists():
        print(f"원본이 없다: {SRC}")
        print("→ 폴더 구조가 다르면 환경변수 DATA_DIR 로 지정한다.")
        raise SystemExit(1)

    print(f"원본 로딩 중: {SRC.name}")
    trips = load_trips()
    for k, v in trips.attrs["funnel"].items():
        print(f"  {k:<16} {v:>10,}")

    gaps = build_gaps(trips)
    n_vehicles = trips["vehicle_id"].nunique()
    print(f"\n유휴 구간 {len(gaps):>10,}건  "
          f"(운행 {len(trips):,} − 차량 {n_vehicles} = 차량별 마지막 운행은 다음 배차가 없다)")

    neg = gaps["idle_min"] < 0
    print(f"  음수 구간 {int(neg.sum()):,}건 — 다음 배차가 직전 하차보다 앞선 기록"
          f"(중복 배차). 분포 계산에서 뺀다.")

    valid = gaps[~neg]
    same = valid[valid["is_same_day"]]
    cross = valid[~valid["is_same_day"]]

    def dist(g, label):
        q = g["idle_min"].quantile([0.5, 0.9]).to_numpy()
        print(f"  {label:<14} {len(g):>10,}건  중앙 {q[0]:7.1f}분  p90 {q[1]:8.1f}분  "
              f"평균 {g['idle_min'].mean():8.1f}분  최대 {g['idle_min'].max():9.1f}분")

    print("\n[구간 길이 분포]")
    dist(valid, "전체")
    dist(same, "당일 내")
    dist(cross, "자정 넘김")
    print("  자정을 넘긴 구간은 근무 종료~다음 근무 시작(비운행 시간)이 섞여 있다.")

    # 일 경계 처리 확인 — 당일 내 구간은 끝점이 같은 날이므로 그 차량이 그 날 다시
    # 일했다는 뜻이고, 따라서 '그 날 마지막 하차 이후'는 전부 자정 넘김으로 빠진다.
    day = same["end"].dt.normalize()
    assert (same["start"].dt.normalize() == day).all()
    print("\n[일 경계] 당일 내 구간은 끝점이 같은 날 = 그 날 다시 운행한 차량이다."
          " 근무 종료 후 구간은 전부 자정 넘김으로 빠진다 → 저녁 과대 없음(A-01 ⑵)")

    n_days = int(same["start"].dt.normalize().nunique())
    by_hour = hourly_concurrency(same, n_days)
    print(f"\n[시간대별 평균 동시 유휴 차량 수] 당일 내 구간 {len(same):,}건 · {n_days}일 기준")
    for _, r in by_hour.iterrows():
        bar = "█" * int(round(r["mean_idle_vehicles"] / 2))
        print(f"  {int(r['hour']):>2}시  {r['mean_idle_vehicles']:6.1f}대  {bar}")

    peak = by_hour.loc[by_hour["mean_idle_vehicles"].idxmax()]
    print(f"  최대 {int(peak['hour'])}시 {peak['mean_idle_vehicles']:.1f}대")

    # 요일별 — 평일과 주말은 가동 대수가 1.5배 갈려 섞으면 둘 다 아닌 값이 된다.
    print("\n[요일별] 분모는 그 날 실제로 나온 대수다(차고지 정원 691대가 아니다)")
    parts = {}
    for label, mask, fleet in (("평일", same["start"].dt.weekday < 5, FLEET_WEEKDAY),
                               ("주말", same["start"].dt.weekday >= 5, FLEET_WEEKEND)):
        g = same[mask]
        nd = int(g["start"].dt.normalize().nunique())
        h = hourly_concurrency(g, nd)
        pk = h.loc[h["mean_idle_vehicles"].idxmax()]
        parts[label] = (pk, fleet, nd, h)
        print(f"  {label} {nd:>3}일  최대 {int(pk['hour'])}시 "
              f"{pk['mean_idle_vehicles']:6.1f}대 / 가동 {fleet}대 대비 "
              f"{pk['mean_idle_vehicles'] / fleet:5.1%}"
              f"   (정원 {DEPOT_CAPACITY}대 대비 {pk['mean_idle_vehicles'] / DEPOT_CAPACITY:.1%})")

    # 긴 당일 구간 민감도 — 자르지는 않고 얼마나 줄어드는지만 본다(A-01 θ 미정)
    long = same["idle_min"] > CAP_MINUTES
    share = same.loc[long, "idle_min"].sum() / same["idle_min"].sum()
    capped = hourly_concurrency(same[~long], n_days)
    cpk = capped.loc[capped["mean_idle_vehicles"].idxmax()]
    ev = by_hour.loc[by_hour["hour"].between(17, 21), "mean_idle_vehicles"].mean()
    cev = capped.loc[capped["hour"].between(17, 21), "mean_idle_vehicles"].mean()
    print(f"\n[민감도] 당일 내 {CAP_MINUTES}분 초과 {int(long.sum()):,}건({long.mean():.1%})이"
          f" 총 유휴 분의 {share:.1%}를 차지한다")
    print(f"  잘라내면 최대 {peak['mean_idle_vehicles']:.1f} → {cpk['mean_idle_vehicles']:.1f}대"
          f" · 저녁(17~21시) 평균 {ev:.1f} → {cev:.1f}대")
    print("  → 식사·교대·충전이 섞인 구간이다(A-01: 유휴 ≠ 가용). 임계 θ 가 정해지기"
          " 전까지 위 대수를 공급량으로 환산하면 안 된다.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([{
        "n_trips": len(trips),
        "n_vehicles": n_vehicles,
        "n_gaps": len(gaps),
        "n_gaps_negative": int(neg.sum()),
        "n_gaps_same_day": len(same),
        "n_gaps_cross_day": len(cross),
        "idle_min_median": float(valid["idle_min"].median()),
        "idle_min_p90": float(valid["idle_min"].quantile(0.9)),
        "idle_min_median_same_day": float(same["idle_min"].median()),
        "idle_min_p90_same_day": float(same["idle_min"].quantile(0.9)),
        "n_days": n_days,
        "peak_hour": int(peak["hour"]),
        "peak_mean_idle_vehicles": float(peak["mean_idle_vehicles"]),
        "peak_weekday": float(parts["평일"][0]["mean_idle_vehicles"]),
        "peak_weekend": float(parts["주말"][0]["mean_idle_vehicles"]),
        "fleet_weekday": FLEET_WEEKDAY,
        "fleet_weekend": FLEET_WEEKEND,
        "peak_share_weekday": float(parts["평일"][0]["mean_idle_vehicles"] / FLEET_WEEKDAY),
        "peak_share_weekend": float(parts["주말"][0]["mean_idle_vehicles"] / FLEET_WEEKEND),
        "cap_minutes": CAP_MINUTES,
        "long_gap_share_of_minutes": float(share),
        "peak_capped": float(cpk["mean_idle_vehicles"]),
    }])
    summary.to_csv(OUT_DIR / "idle_gaps.csv", index=False, encoding="utf-8-sig")
    by_hour.to_csv(OUT_DIR / "idle_by_hour.csv", index=False, encoding="utf-8-sig")
    print(f"\n저장: {OUT_DIR / 'idle_gaps.csv'} / {OUT_DIR / 'idle_by_hour.csv'}")


if __name__ == "__main__":
    main()
