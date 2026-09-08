# -*- coding: utf-8 -*-

"""patience.py — 콜 인내심 분포 추정

실측 콜 기록에서 대기열 이탈 시점을 추정해 시뮬레이션의 포기 로직에 사용한다.

Kaplan-Meier 생존분석을 적용하며,
배차 전 취소는 이벤트로, 배차된 콜은 우측절단으로 처리한다.
즉시 취소와 취소 시각이 불명확한 건은 인내심 추정에서 제외한다.

정의
모집단: 즉시콜 + 서울 출발 콜
이벤트: 배차 전 취소
절단: 배차 완료
duration: 접수 시점부터 취소 또는 배차까지

산출
outputs/patience_km.csv
outputs/patience_summary.csv

실행
python src/patience.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

import load  # noqa: E402
import metrics as M  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "outputs"

# 인내심으로 보지 않는 취소의 경계(분). metrics.CANCEL_IMMEDIATE_MIN 과 같은 값이며,
# 어긋나면 취소 4구간과 인내심 모집단이 갈라진다.
IMMEDIATE_MIN = M.CANCEL_IMMEDIATE_MIN

# 관측 범위(분). 600분을 넘는 대기는 기록 오류로 보고 뺀다.
MAX_DURATION_MIN = 600.0

# 문서·검증에 쓰는 확인 시점(분)
CHECKPOINTS = [5, 10, 20, 30, 45, 60, 90]


def build_observations(calls: pd.DataFrame) -> pd.DataFrame:
    """콜 로그 → (duration, event) 관측표. 단계별 건수는 attrs['funnel'] 에 담는다.

    event=True 가 포기(이벤트), False 가 절단이다.
    """
    funnel = {"모집단(즉시콜·서울)": len(calls)}

    kind = pd.Series(M.cancel_kind(calls), index=calls.index).astype(object)
    canceled_at = pd.to_datetime(calls["canceled_at"], errors="coerce")
    cancel_min = (canceled_at - calls["received_at"]).dt.total_seconds() / 60
    assign_min = (calls["assigned_at"] - calls["received_at"]).dt.total_seconds() / 60

    is_immediate = kind == "immediate"
    is_abandoned = kind == "abandoned"
    is_other = kind == "other"
    is_assigned = calls["assigned_at"].notna()

    funnel["즉시 취소 제외"] = -int(is_immediate.sum())
    funnel["취소시각 결측 제외"] = -int(is_other.sum())

    duration = pd.Series(np.nan, index=calls.index, dtype="float64")
    event = pd.Series(False, index=calls.index)

    duration[is_abandoned] = cancel_min[is_abandoned]
    event[is_abandoned] = True
    duration[is_assigned] = assign_min[is_assigned]      # 절단

    keep = duration.notna() & ~is_immediate & ~is_other
    funnel["duration 산출됨"] = int(keep.sum())

    in_range = keep & duration.between(0, MAX_DURATION_MIN)
    funnel[f"0~{MAX_DURATION_MIN:.0f}분 범위"] = int(in_range.sum())

    out = pd.DataFrame({
        "duration": duration[in_range].to_numpy(dtype="float64"),
        "event": event[in_range].to_numpy(dtype=bool),
    })
    funnel["  포기(이벤트)"] = int(out["event"].sum())
    funnel["  절단"] = int((~out["event"]).sum())
    out.attrs["funnel"] = funnel
    out.attrs["n_immediate"] = int(is_immediate.sum())
    out.attrs["n_population"] = len(calls)
    return out


def kaplan_meier(duration: np.ndarray, event: np.ndarray) -> pd.DataFrame:
    """우측절단 Kaplan-Meier 추정.

        S(t) = ∏         (1 − d_i / n_i)
               t_i ≤ t

    d_i 는 t_i 의 이벤트 수, n_i 는 t_i 직전의 위험집합 크기(= duration ≥ t_i 인 수).
    같은 시점의 절단은 위험집합에 남긴다(관례) — n_i 를 'duration ≥ t_i' 로 세면
    자동으로 그렇게 된다.

    생존율은 이벤트 시점에서만 꺾이므로 **이벤트 시점만** 돌려준다. 그 사이는
    직전 값을 유지하는 계단함수다.

    반환: timeline, n_at_risk, n_events, survival, cum_abandon
    """
    order = np.argsort(duration, kind="stable")
    d = duration[order]
    e = event[order].astype(np.int64)

    times, first = np.unique(d, return_index=True)
    n_at_risk = len(d) - first                      # duration ≥ t 인 건수
    n_events = np.add.reduceat(e, first)

    hit = n_events > 0                              # 이벤트가 있는 시점만 꺾인다
    times, n_at_risk, n_events = times[hit], n_at_risk[hit], n_events[hit]

    survival = np.cumprod(1.0 - n_events / n_at_risk)
    return pd.DataFrame({
        "timeline": times,
        "n_at_risk": n_at_risk,
        "n_events": n_events,
        "survival": survival,
        "cum_abandon": 1.0 - survival,
    })


def survival_at(km: pd.DataFrame, t: float) -> float:
    """시점 t 의 생존율. 계단함수라 t 이하의 마지막 이벤트 시점 값을 쓴다."""
    i = np.searchsorted(km["timeline"].to_numpy(), t, side="right") - 1
    return 1.0 if i < 0 else float(km["survival"].to_numpy()[i])


def at_risk_at(km: pd.DataFrame, t: float) -> int:
    """시점 t 의 위험집합 크기. 곡선을 얼마나 믿을 수 있는지의 척도다."""
    i = np.searchsorted(km["timeline"].to_numpy(), t, side="right") - 1
    return int(km["n_at_risk"].to_numpy()[0]) if i < 0 \
        else int(km["n_at_risk"].to_numpy()[i])


def km_grid(km: pd.DataFrame, step: float = 0.5,
            tmax: float = MAX_DURATION_MIN) -> pd.DataFrame:
    """계단함수를 일정 간격 격자에서 평가한다 — 시뮬이 역CDF로 쓸 형태.

    격자점에서의 값은 계단함수를 그대로 읽은 것이라 **정확하다**(격자 사이만
    보간되지 않을 뿐). 이벤트 시점 8만 개를 그대로 두면 5MB가 넘어 저장소에
    넣기 어렵고, 시뮬의 시간 단위가 분이라 0.5분 해상도면 충분하다.
    """
    grid = np.arange(0.0, tmax + step / 2, step)
    t = km["timeline"].to_numpy()
    idx = np.searchsorted(t, grid, side="right") - 1
    surv = np.where(idx < 0, 1.0, km["survival"].to_numpy()[np.clip(idx, 0, None)])
    risk = np.where(idx < 0, km["n_at_risk"].to_numpy()[0],
                    km["n_at_risk"].to_numpy()[np.clip(idx, 0, None)])
    return pd.DataFrame({
        "timeline": grid,
        "survival": surv,
        "cum_abandon": 1.0 - surv,
        "n_at_risk": risk.astype(np.int64),
    })


def median_survival(km: pd.DataFrame) -> float:
    """생존율이 0.5 이하로 처음 떨어지는 시점. 안 떨어지면 nan."""
    s = km["survival"].to_numpy()
    idx = np.nonzero(s <= 0.5)[0]
    return float(km["timeline"].to_numpy()[idx[0]]) if len(idx) else float("nan")


def _km_bruteforce(duration: np.ndarray, event: np.ndarray) -> np.ndarray:
    """검산용 순진 구현. 벡터화 결과와 맞는지 보려고 둔다(작은 표본에만)."""
    times = np.unique(duration[event])
    s, out = 1.0, []
    for t in times:
        n = int((duration >= t).sum())
        d = int(((duration == t) & event).sum())
        s *= (1 - d / n)
        out.append(s)
    return np.array(out)


def main():
    print("콜 로딩 중 (load.load_calls — 캐시가 있으면 재사용)")
    calls = load.load_calls()

    obs = build_observations(calls)
    print("\n[모집단 구성]")
    for k, v in obs.attrs["funnel"].items():
        print(f"  {k:<22} {v:>12,}")

    n_pop = obs.attrs["n_population"]
    n_imm = obs.attrs["n_immediate"]
    print(f"\n  즉시 취소 {n_imm:,}건 = 모집단의 {n_imm / n_pop:.2%} "
          f"— 인내심과 무관한 오조작·변심이라 접수 시점의 외생 확률로 처리한다.")

    dur = obs["duration"].to_numpy()
    ev = obs["event"].to_numpy()

    # 벡터화 KM 검산 — 앞쪽 표본으로 순진 구현과 대조
    sub = slice(0, 20_000)
    a = kaplan_meier(dur[sub], ev[sub])["survival"].to_numpy()
    b = _km_bruteforce(dur[sub], ev[sub])
    print(f"\n  KM 구현 검산(2만 건 부분표본): 최대 절대차 {np.abs(a - b).max():.3e}")

    km = kaplan_meier(dur, ev)
    print(f"\n[KM 생존곡선] 꺾이는 시점 {len(km):,}개 "
          f"(이벤트가 있는 시점에서만 꺾인다)")

    print("\n[누적 포기율]")
    for t in CHECKPOINTS:
        print(f"  {t:>3}분   {1 - survival_at(km, t):6.2%}")

    km_med = median_survival(km)
    naive_med = float(np.median(dur[ev]))
    print(f"\n[생존편향]")
    print(f"  KM 중앙        {km_med:>8.1f}분   (절단 포함 — 인내심 성향)")
    print(f"  순진 중앙      {naive_med:>8.1f}분   (취소자만 — 성질 급한 표본)")
    print(f"  배율           {km_med / naive_med:>8.1f}배")
    print("  순진 분포를 시뮬에 쓰면 취소가 폭발한다.")

    # 꼬리가 얼마나 얇은지 — 곡선을 어디까지 믿을 수 있나
    print(f"\n[위험집합] 곡선의 신뢰 구간을 가늠하는 값이다")
    for t in [30, 60, 90, 120, 180, km_med]:
        print(f"  {t:>6.0f}분   위험집합 {at_risk_at(km, t):>9,}  "
              f"누적 포기 {1 - survival_at(km, t):6.2%}")
    n_late = int(km[km["timeline"] > 90]["n_events"].sum())
    print(f"  90분 이후 이벤트 {n_late:,}건 = 전체 포기의 {n_late / int(ev.sum()):.1%}")
    print("  → 중앙 이후 구간은 관측이 몇백 건뿐이다. 곡선 앞부분(~60분)과 같은 "
          "무게로 읽으면 안 된다.")

    n_aband = int(ev.sum())
    aband_ratio = n_aband / n_pop
    print(f"\n[검증 기준]")
    print(f"  실측 대기 중 포기 {n_aband:,}건 / 콜 {n_pop:,}건 = {aband_ratio:.2%}")
    print("  시뮬이 이 비율을 재현하는지로 인내심 곡선과 배차 로직을 함께 판정한다.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grid = km_grid(km)
    grid.to_csv(OUT_DIR / "patience_km.csv", index=False, encoding="utf-8-sig")

    summary = {
        "n_population": n_pop,
        "n_immediate": n_imm,
        "immediate_ratio": n_imm / n_pop,
        "n_observed": len(obs),
        "n_events": n_aband,
        "n_censored": int((~ev).sum()),
        "abandon_ratio": aband_ratio,
        "km_median_min": km_med,
        "naive_median_min": naive_med,
        "bias_ratio": km_med / naive_med,
        "last_event_min": float(km["timeline"].iloc[-1]),
        "n_at_risk_60min": at_risk_at(km, 60),
        "n_at_risk_median": at_risk_at(km, km_med),
        "n_events_after_90min": n_late,
        "max_duration_min": MAX_DURATION_MIN,
        "immediate_min": IMMEDIATE_MIN,
        "grid_step_min": 0.5,
    }
    for t in CHECKPOINTS:
        summary[f"cum_abandon_{t}min"] = 1 - survival_at(km, t)
    pd.DataFrame([summary]).to_csv(OUT_DIR / "patience_summary.csv",
                                   index=False, encoding="utf-8-sig")
    print(f"\n저장: {OUT_DIR / 'patience_km.csv'} / {OUT_DIR / 'patience_summary.csv'}")

    print("\n[시뮬에서 쓰는 법]")
    print("  u ~ U(0,1) 을 뽑아 survival 이 u 이하로 처음 떨어지는 timeline 이 인내심이다.")
    print(f"  곡선은 {km['timeline'].iloc[-1]:.1f}분에서 생존율 0 이 된다 — 마지막 관측이 "
          "이벤트라 그렇다.")
    print("  꼬리가 얇으니 그 구간을 절대값으로 읽지 말 것(위 위험집합 참조).")
    print("  즉시 취소는 곡선과 별개다. 접수 시점에 "
          f"{n_imm / n_pop:.4f} 확률로 따로 뽑는다(A-14).")


if __name__ == "__main__":
    main()
