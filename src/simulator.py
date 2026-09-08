"""
simulator.py — 장애인콜택시 이산사건 시뮬레이션 엔진

SimPy로 차량 출차부터 콜 접수, 배차, 공차 이동, 승차, 운행, 하차,
유휴 및 근무 종료까지의 하루 운행을 재현한다.

배치안(거점 위치·배속)을 입력받아 before/after 성과를 계산하며,
출력은 실측 콜과 동일한 형태로 구성해 `metrics.py`의 지표 계산을 재사용한다.

주요 가정
- 유휴 차량은 하차지 인근에서 대기
- 조별 시차 출차 적용
- 신규 거점은 기존 배속을 유지한 채 증차
- 예약콜은 실측 점유량만큼 가용 차량에서 차감
- 인내심은 Kaplan-Meier 기반 분포 사용
- 특장차 콜만 재생

차량 위치와 배차 계산은 행정동 단위로 처리하며,
동 간 이동시간은 `travel_time.py`에서 조회한다.

차량 공급은 연간 고유 차량 수가 아니라
평일·주말의 실제 일별 가동 대수를 기준으로 구성한다.

실행
python src/simulator.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import simpy

import load
import travel_time as T

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "outputs"

# ─────────────────────────────────────────────────────────────
# 상수
# ─────────────────────────────────────────────────────────────

# 난수 시드(재현성 — 변경 시 기록). 단일 시드 · PCG64 단일 스트림이며,
# 샘플링 스트림 분리·공통난수(CRN)는 후보 비교 단계에서 정한다(model_flow.md).
SEED = 42

# 자동배차 탐색 반경(km) — 서울시설공단 장애인콜택시 이용 안내
# https://www.sisul.or.kr/open_content/calltaxi/introduce/receipt.jsp
#
# 원문은 '평일 주간 7km / 주말 8km / 19~24시 12km' 세 갈래다. **평일 심야
# (00:00~06:30)를 따로 적지 않아** 야간 반경을 그 구간에도 적용했다 — 심야에
# 반경이 좁을 이유가 없고, 좁게 두면 새벽 콜이 구조적으로 미배차된다.
DISPATCH_RADIUS_DAY = 7.0        # 평일 06:30~19:00
DISPATCH_RADIUS_WEEKEND = 8.0    # 주말 06:30~19:00
DISPATCH_RADIUS_NIGHT = 12.0     # 그 외(19:00~익일 06:30)
DAY_START_MIN, DAY_END_MIN = 6 * 60 + 30, 19 * 60

# **평일 심야(00:00~06:30) 반경만 따로 뗀 손잡이**(A-17). 기본값은 야간과 같은
# 12km 라 `_radius_km` 의 동작이 바뀌지 않는다 — 민감도에서 7km 로 갈아끼우려고
# 이름만 갈라 둔 것이다. 원문이 이 구간을 적지 않아 어느 쪽도 근거가 없고,
# 주간(7km)으로 읽으면 근무 차량 6~8대뿐인 새벽에 반경이 가장 좁아진다.
DISPATCH_RADIUS_PREDAWN = DISPATCH_RADIUS_NIGHT

# 배차 점수 가중치(합 90) — 위 이용 안내 원문 그대로다.
#   order   접수순서   먼저 접수된 콜이 높다
#   wait    경과대기   오래 기다린 콜이 높다
#   dist    거리       가까운(이동시간이 짧은) 콜이 높다
#
# **order 와 wait 은 한 시점에서 순위가 같다** — 먼저 접수된 콜이 곧 오래 기다린
# 콜이다. 두 항을 따로 두는 것은 원문을 그대로 옮기기 위해서이고, 실질은
# 50 : 40 으로 '대기 우선 vs 근접 우선'을 가르는 셈이다. 검증에서 어긋나면
# 여기를 민감도로 확인한다.
SCORE_W = {"order": 15.0, "wait": 35.0, "dist": 40.0}

# 조 편성(A-09) — 첫 운행 시각 실측 분포(특장차 한정, 차량-일 16.9만 건).
# docs/provenance.md 8절과 같은 값이다. 합이 1이 되도록 정규화해 쓴다.
SHIFT_TABLE = {7: 0.284, 8: 0.243, 12: 0.238, 13: 0.060, 15: 0.033,
               9: 0.031, 10: 0.020, 19: 0.019, 16: 0.017, 14: 0.016, 0: 0.014}

# 조별 종료 시각(실측 중앙). 표에 없는 출차 시각은 WORK_HOURS 를 더해 잡는다.
SHIFT_END = {7: 14, 8: 15, 12: 18, 13: 19, 15: 22}
WORK_HOURS = 6.72                # 근무 길이 중앙(사분위 5.39~7.71)
SHIFT_JITTER_MIN = 15.0          # 출차·종료 시각의 분 단위 흔들림(±)

# 일별 가동 차량(접수일 기준 중앙). metrics.SIM_FLEET 과 같은 값이며,
# **요일별로 갈아끼운다** — 연간 중앙 561 하나로 접으면 평일 공급이 과소해진다.
FLEET_WEEKDAY, FLEET_WEEKEND = 574, 374

# 신규 거점 증차 대수(A-10). 기존 44곳의 배속은 건드리지 않는다.
NEW_DEPOT_VEHICLES = 10

# 대표 기간(A-03) — 30일 창 365개 중 **연간 평균 대기(40.7553분)에 가장 가까운**
# 구간이다. 결과를 보기 전에 고정했다(사후 선택 차단). 그 구간 평균 40.7543분으로
# 차이가 0.001분이고 콜 108,223건이다.
#
# **시뮬은 연간 검증 기준이 아니라 이 구간의 실측값과 대조한다** — 특히 지니는
# 기간이 짧으면 동별 평균의 표본오차가 얹혀 위로 편향된다(연간 0.092 vs 대표월
# 0.1062). 구간별 실측 7개는 docs/provenance.md 13절.
REPRESENTATIVE_START, REPRESENTATIVE_DAYS = "2025-08-20", 30

# 즉시 취소 외생 확률(A-14) — 인내심 곡선과 별개로 접수 시점에 뽑는다.
IMMEDIATE_CANCEL_P = 0.0316
IMMEDIATE_CANCEL_MAX_MIN = 1.0   # metrics.CANCEL_IMMEDIATE_MIN 과 같은 경계

# 배차 후 취소 외생 확률 — 실측 3.74%(배차는 됐는데 승차 기록이 없는 건).
# 차가 오다 어긋난 것이라 인내심과 무관하다. 배차 시점에 한 번 뽑는다.
AFTER_ASSIGN_CANCEL_P = 0.0374

# 휴게 임계 θ(분) — **유휴 ≠ 가용**(A-01). 이 시간 동안 배차가 없으면 그 차량이
# 식사·교대·충전으로 자리를 비운 것으로 보고 배차 대상에서 뺀다. 휴게 길이는 자유
# 파라미터로 두지 않고 실측 유휴 구간의 초과분에서 뽑는다(`load.load_idle_gaps`).
#
# **기본값은 None — 끈다.** 30~180분을 훑어 실측과 대조한 결과(2026.08.09),
# θ 는 대기를 끌어올리기는 하지만 **독립적인 두 검사에서 반대 방향으로 움직인다.**
#   동시 유휴 최대(실측 164.6대)  θ 없음 163.5 → θ=20 148.6  (멀어진다)
#   지역 지니(실측 0.10)          θ=120 0.11 → θ=20 0.14      (멀어진다)
# 게다가 유휴 분의 55%를 걷어내는 θ=15 에서도 매칭 대기가 19.4분으로 실측
# 24.8분에 못 미친다. **θ 를 더 조이면 대기만 맞고 나머지가 어긋난다** — 그것은
# θ 가 원인이 아니라 대기를 맞추는 손잡이가 됐다는 뜻이다.
#
# 진짜 원인은 매칭 지연 쪽이다. 실측은 **1분 안에 배차된 콜이 0.33%뿐**이고
# p1 이 1.27분이라 바닥이 있는데, 이 엔진은 빈 차가 있으면 즉시 붙인다(콜의 78%가
# 접수 즉시 배차). 그 바닥과 꼬리를 만드는 절차(기사 수락·자동수락 타이머)가
# 모델에 없다. 기구는 남겨두되 근거가 생길 때까지 켜지 않는다.
IDLE_BREAK_THETA_MIN = None

# 배차 후보 필터(A-20) — 반경 안 유휴 차량이 전부 실제로 배차 가능하지는 않다.
#
# 진단(2026.08.09): 접수 즉시 배차되는 콜의 반경 안 후보가 중앙 30대인데 그중
# **최솟값**을 고르면 이동시간 8.6분, 후보 **중앙값** 차량은 22.3분이다. 실측 픽업
# 중앙 17.8분은 최솟값보다 중앙값 쪽에 가깝다 — 실제 배차는 가장 가까운 차를
# 고르지 못한다. 기사 거절·사전 배정·차량 상태로 후보가 걸러지는 것으로 보인다.
#
# **θ 와 다르다.** θ 는 차량을 몇 시간씩 빼 공급 자체를 줄이므로 개입 경로 밖이었지만,
# 이 필터는 **그 배차 순간에만** 적용돼 총 처리량을 거의 건드리지 않고 '누가 오느냐'만
# 바꾼다. 걸러진 뒤에도 남은 후보 중 가장 가까운 차를 고르므로 **거점이 가까우면
# 픽업이 줄어드는 반응은 살아 있다** — 그래서 개입 경로 안이다.
#
# 두 가지 방식을 훑어 골랐다(모드별 근거는 `_filter_candidates`).
#   random     각 후보가 keep_p 확률로 살아남는다 — 기사 개별 거절
#   drop_near  가까운 순으로 (1−keep_p) 를 버린다 — 근거리가 체계적으로 막힘
CANDIDATE_KEEP_P = 1.0            # 1.0 이면 필터 없음(현행 기본)
CANDIDATE_FILTER_MODE = "random"

# 난수 스트림 — 콜 단위로 고정된 값을 뽑기 위한 식별자.
# **배치안이 달라도 같은 콜은 같은 인내심·같은 취소 판정을 받는다.** 그래야
# 후보 간 차이가 위치에서만 나온다(공통난수). 이벤트 순서에 기대지 않는다.
STREAM_IMMEDIATE, STREAM_PATIENCE = 0, 1
STREAM_IMMEDIATE_AT, STREAM_AFTER_ASSIGN = 2, 3
# 차량 슬롯 스트림 — 조 편성도 ID 기반이라야 증차가 기존 편성을 흔들지 않는다
STREAM_SHIFT_HOUR, STREAM_SHIFT_JITTER, STREAM_SHIFT_END_JITTER = 4, 5, 6

# 하루치 차량 슬롯 ID 폭. 가동 대수(최대 626)보다 넉넉해야 날짜 간 ID 가 겹치지 않는다.
SLOTS_PER_DAY = 1024

# 재생하지 않고 남겨둔 것 — 검증에서 어긋날 때 먼저 의심할 목록이다.
KNOWN_GAPS = (
    "배차 후 취소(실측 3.74%) 미구현 — 배차되면 반드시 승차한다. 시뮬 취소는 "
    "즉시 취소와 대기 중 포기 둘뿐이라 취소율이 그만큼 낮게 나온다",
    "조별 종료는 실측 중앙값 고정이다 — 분포를 주지 않아 저녁 공급의 분산이 없다",
    "유휴 ≠ 가용(A-01)을 반영하지 않는다 — 식사·교대·충전이 없어 유휴 차량이 "
    "전부 배차 가능하다. 공급이 그만큼 낙관적이다",
    "광역콜은 운행 시간만 반영하고 경기 체류·복귀를 모델링하지 않는다",
    "차량 고장·결행 없음",
    "재생 구간 끝에 접수된 콜은 처리 시간이 모자라 미해결로 남는다(warmdown 참조)",
)


# ─────────────────────────────────────────────────────────────
# 이동시간 행렬 — 조회를 배열 인덱싱으로 바꾼다
# ─────────────────────────────────────────────────────────────

class TravelMatrix:
    """동쌍 × (시간대 5 × 평일/주말) 이동시간 행렬 + 동쌍 직선거리 행렬.

    `travel_time.get_travel_time` 을 콜마다 부르면 파이썬 호출이 수백만 번이라
    엔진이 조회에 묶인다. 425개 동은 전수 조합이 180,625개뿐이라 **10개 프로필을
    통째로 미리 채워두는 편이 싸다**(빌드 ~10초, 메모리 7MB). 채우는 값은
    `lookup_many` 가 내는 것과 같아 3단 폴백 사슬도 동일하다.

    좌표를 못 찾는 동(서울 밖 목적지 등)은 행렬에 없다. 그 경우 시간대·요일
    프로필의 운행시간 중앙값으로 떨어진다 — `lookup` 의 'default' 순위와 같다.

    직선거리 행렬은 **탐색 반경 판정 전용**이다. 반경은 원문이 km 로 주므로
    직선거리로 자르고, 점수의 거리항은 이동시간(분)을 쓴다 — 실제 배차 판단에
    맞는 것은 좌표 거리가 아니라 걸리는 시간이다.

    **행렬의 동 목록은 경계 파일 425개로 끝나지 않는다.** 콜 원본의 동 구분이
    더 잘아서(공릉13동·답십리3동·신당1~6동 등 19개) 표준형이 경계 파일에 없고,
    `travel_time` 은 그런 이름을 기본명 좌표로 흡수한다. 그 이름들을 행렬에
    넣지 않으면 해당 동의 콜이 통째로 빠지므로 `extra_dongs` 로 받아 함께 채운다.
    """

    def __init__(self, tt: T.TravelTime, extra_dongs=None):
        cen = tt.centroids[tt.centroids["key_kind"] == "canon"].reset_index(drop=True)
        dongs = cen["dong"].tolist()
        known = set(dongs)
        for d in sorted(set(extra_dongs or ())):
            if d not in known and tt.coord_of(d) is not None:
                dongs.append(d)
                known.add(d)

        self.dongs = dongs
        self.index = {d: i for i, d in enumerate(dongs)}
        n = len(dongs)

        xy = np.array([tt.coord_of(d) for d in dongs], dtype=float)
        self.lat, self.lon = xy[:, 0], xy[:, 1]
        self.km = T.haversine_km(self.lat[:, None], self.lon[:, None],
                                 self.lat[None, :], self.lon[None, :]).astype("float32")

        self.periods = list(T.PERIODS)
        # 각 시간대를 대표하는 시각 하나(그 시간대면 어느 시각이든 값이 같다)
        rep_hour = {}
        for h in range(24):
            rep_hour.setdefault(T.period_of(h), h)

        o = np.repeat(dongs, n)
        d = np.tile(dongs, n)
        self.minutes = np.empty((len(self.periods), 2, n, n), dtype="float32")
        for pi, period in enumerate(self.periods):
            for wi, wk in enumerate((False, True)):
                got = tt.lookup_many(o, d, rep_hour[period], wk)["minutes"]
                self.minutes[pi, wi] = got.to_numpy(dtype="float32").reshape(n, n)

        prof = tt.profile.set_index(["period", "is_weekend"])["minutes"]
        self.default_minutes = np.array(
            [[float(prof.get((p, wk), T.FALLBACK_MINUTES)) for wk in (False, True)]
             for p in self.periods], dtype="float32")
        self._period_of_hour = np.array(
            [self.periods.index(T.period_of(h)) for h in range(24)], dtype="int8")

    def period_index(self, hour: int) -> int:
        return int(self._period_of_hour[hour])

    def idx(self, dong: str):
        """표준형 동명 → 행렬 색인. 없으면 None(좌표 없는 동)."""
        return self.index.get(dong)

    def nearest_index(self, lat: float, lon: float) -> int:
        """좌표 → 가장 가까운 동 중심점 색인.

        차고지·후보의 '출발 동'을 잡는 데 쓴다. 경계 공간조인이 아니라 중심점
        최근접이라 경계 근처의 거점은 이웃 동으로 잡힐 수 있다 — 차량의 하루
        첫 위치에만 영향을 주고, 첫 배차 이후에는 하차지로 갈아탄다(A-01).
        """
        return int(np.argmin(T.haversine_km(lat, lon, self.lat, self.lon)))


# ─────────────────────────────────────────────────────────────
# 인내심 (A-14)
# ─────────────────────────────────────────────────────────────

class Patience:
    """KM 생존곡선에서 인내심을 뽑는다(역CDF). 입력은 outputs/patience_km.csv.

    `u ~ U(0,1)` 을 뽑아 생존율이 u 이하로 처음 떨어지는 시점이 그 콜의 인내심이다.
    곡선이 0.5분 격자라 인내심도 0.5분 단위로 떨어진다 — 대기 분포의 해상도에
    비하면 무시할 수 있다.

    **꼬리가 얇다.** 90분 이후 이벤트가 전체 포기의 1.6%뿐이라 그 구간은 관측
    몇백 건에 기댄다. 지금은 곡선을 그대로 쓰고, 검증에서 포기율이 어긋나면
    그때 조정한다.
    """

    def __init__(self, path: Path = None):
        path = Path(path) if path else OUT_DIR / "patience_km.csv"
        km = pd.read_csv(path, encoding="utf-8-sig")
        # 생존율은 단조 감소다. searchsorted 를 쓰려고 오름차순으로 뒤집어 둔다.
        self._surv_asc = km["survival"].to_numpy(dtype="float64")[::-1].copy()
        self._time_asc = km["timeline"].to_numpy(dtype="float64")[::-1].copy()
        self.max_min = float(km["timeline"].iloc[-1])

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """인내심 n개(분). 생존율이 u 이하로 처음 떨어지는 timeline."""
        return self.from_uniform(rng.random(n))

    def from_uniform(self, u: np.ndarray) -> np.ndarray:
        """이미 뽑은 균등난수로 역CDF를 친다 — 콜 단위 스트림이 쓰는 진입점."""
        j = np.searchsorted(self._surv_asc, np.asarray(u), side="left")
        return self._time_asc[np.clip(j, 0, len(self._time_asc) - 1)]


class IdleBreak:
    """휴게 길이 표본기 — **θ 만 파라미터이고 길이는 실측에서 나온다**(A-01).

    실측 당일 내 유휴 구간 중 `gap > θ` 인 것들의 **초과분**(gap − θ)이 곧 그
    차량이 자리를 비웠던 시간이다. 그 분포를 그대로 역CDF로 뽑는다. 휴게 길이를
    따로 정하면 θ 와 함께 두 개의 손잡이가 되어 무엇이든 맞출 수 있게 된다.

    θ 가 커지면 표본이 얇아진다 — 남는 구간이 몇 건 안 되면 만들지 않고 튕긴다.
    """

    MIN_TAIL = 100

    def __init__(self, gaps: np.ndarray, theta_min: float):
        self.theta = float(theta_min)
        tail = np.asarray(gaps, dtype="float64")
        tail = tail[tail > self.theta] - self.theta
        if len(tail) < self.MIN_TAIL:
            raise ValueError(f"θ={theta_min}분을 넘는 유휴 구간이 {len(tail)}건뿐이라 "
                             f"휴게 길이 분포를 만들 수 없다")
        self._tail = np.sort(tail)
        self.share_of_idle_minutes = float(
            tail.sum() / max(np.asarray(gaps, dtype="float64").sum(), 1e-9))

    def sample(self, rng: np.random.Generator) -> float:
        return float(self._tail[int(rng.integers(len(self._tail)))])


def uniform_by_id(ids: np.ndarray, seed: int, stream: int) -> np.ndarray:
    """콜 ID → [0,1) 균등난수. **같은 (id, seed, stream)이면 항상 같은 값이다.**

    배치안·기간·이벤트 순서가 무엇이든 콜 하나의 인내심이 고정돼야 후보 간
    차이가 순수하게 위치에서 나온다(공통난수). 순서에 기대는 생성기로는 그
    성질이 안 나온다 — 배차가 달라지면 뽑는 순서가 달라지기 때문이다.

    splitmix64 를 벡터로 돌린다. 카운터 기반이라 건너뛰기·부분집합에 안전하다.
    """
    with np.errstate(over="ignore"):
        x = (np.asarray(ids, dtype=np.uint64)
             + np.uint64(0x9E3779B97F4A7C15) * np.uint64(stream + 1)
             + np.uint64(0xBF58476D1CE4E5B9) * np.uint64(seed))
        z = x + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)


# ─────────────────────────────────────────────────────────────
# 차량 배치 · 조 편성
# ─────────────────────────────────────────────────────────────

def allocate_fleet(depots: pd.DataFrame, n_total: int) -> np.ndarray:
    """배속 정원 비율로 가동 대수를 나눈다. 반환은 거점별 대수(합 = n_total).

    **세 숫자를 구분해야 한다.**
      배속 정원 합 691대   거점별 `capacity`. 여기서는 **배분 비율**로만 쓴다
      일별 가동 574/374대  그 날 실제로 나온 대수. 시뮬에 넣는 값
      연간 누적 723대      1년간 등장한 고유 차량번호. 쓰지 않는다

    최대잉여법으로 반올림해 합을 정확히 맞춘다. 단순 반올림은 44곳에서 합이
    몇 대씩 어긋나고, 그 오차가 거점별 차이로 오해된다.

    정원이 있는 거점은 **최소 1대를 보장한다** — 비율만 따르면 작은 거점이
    0대가 되어 그 지역 공급이 통째로 사라진다.
    """
    cap = depots["capacity"].to_numpy(dtype=float)
    if cap.sum() <= 0:
        raise ValueError("거점 정원 합이 0이다")
    if n_total < int((cap > 0).sum()):
        raise ValueError(f"가동 대수 {n_total} 가 거점 수보다 적어 최소 1대를 "
                         f"보장할 수 없다({int((cap > 0).sum())}곳)")

    exact = cap / cap.sum() * n_total
    base = np.where(cap > 0, np.maximum(np.floor(exact), 1), 0).astype(int)
    frac = exact - np.floor(exact)

    short = n_total - int(base.sum())
    if short > 0:                      # 남은 대수는 소수부가 큰 순으로
        for i in np.argsort(-frac)[:short]:
            base[i] += 1
    else:
        # 최소 1대 보장으로 넘친 만큼 되돌린다. **한 바퀴로는 모자랄 수 있다** —
        # 가동 대수가 거점 수에 가까우면 큰 거점이 2대 이상을 내놔야 해서,
        # 소수부가 작은 순으로 여러 바퀴 돈다(1대 밑으로는 내리지 않는다).
        order = np.argsort(frac)
        while short < 0:
            moved = False
            for i in order:
                if short == 0:
                    break
                if base[i] > 1:
                    base[i] -= 1
                    short += 1
                    moved = True
            if not moved:              # 전부 1대 — 위 가드에서 걸러졌어야 한다
                raise AssertionError("배분이 수렴하지 않는다")
    return base


def build_shifts(slot_ids: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """차량 슬롯의 (출차 시각, 근무 종료 시각) — 자정 기준 분.

    출차 시각은 실측 분포(SHIFT_TABLE)에서 뽑는다. 종료는 실측 중앙(SHIFT_END)이
    있으면 그 값, 없으면 출차 + WORK_HOURS.

    **분 단위 흔들림을 준다**(±15분 균등) — 실측이 시(hour) 단위 집계라 그대로
    쓰면 같은 조가 정확히 같은 분에 나오고, 출차 직후 배차가 한 틱에 몰려 대기
    분포에 계단이 생긴다.

    **슬롯 ID 로 뽑는다 — 순서 기반 생성기가 아니다.** 신규 거점이 붙으면 가동
    대수가 574 → 582 로 늘어나는데(A-10), 순서 기반이면 그 순간 **전 차량의 조
    편성이 통째로 달라져** before/after 차이에 난수가 섞인다. 실측한 크기가
    작지 않다 — 시드만 바꿔도 동별 픽업이 표준편차 **0.465분**(최대 ±1.4분)
    움직이는데, 후보 하나의 효과는 −0.1~−0.2분이다. **잡음이 신호보다 크다.**
    슬롯 ID 로 고정하면 앞 574 슬롯의 조 편성이 그대로 유지되고 늘어난 8대만
    새로 붙어, 차이가 순수하게 배치에서 나온다(공통난수).
    """
    ids = np.asarray(slot_ids)
    hours = np.array(list(SHIFT_TABLE), dtype=int)
    p = np.array([SHIFT_TABLE[h] for h in hours], dtype=float)
    cum = np.cumsum(p / p.sum())
    start_h = hours[np.searchsorted(cum, uniform_by_id(ids, seed, STREAM_SHIFT_HOUR),
                                    side="right").clip(0, len(hours) - 1)]

    j = SHIFT_JITTER_MIN
    start = start_h * 60.0 + (uniform_by_id(ids, seed, STREAM_SHIFT_JITTER) * 2 - 1) * j
    end = np.array([SHIFT_END[int(h)] * 60.0 if int(h) in SHIFT_END
                    else h * 60.0 + WORK_HOURS * 60.0 for h in start_h])
    end = end + (uniform_by_id(ids, seed, STREAM_SHIFT_END_JITTER) * 2 - 1) * j
    return start, np.maximum(end, start + 60.0)


# ─────────────────────────────────────────────────────────────
# 배치안
# ─────────────────────────────────────────────────────────────

def resolve_placement(placement, depots: pd.DataFrame) -> pd.DataFrame:
    """배치안을 거점 표로 편다. 반환은 `depots` 와 같은 스키마 + 신규 행.

    placement 로 받는 것:
      None                    현행 배치(44곳). before 쪽
      int                     `outputs/dong_candidates.csv` 의 `cand_id`
      dict(name, lat, lon)    좌표를 직접 준다
      DataFrame               위 dict 여러 건

    **신규 거점의 배속은 언제나 10대다**(A-10 · `NEW_DEPOT_VEHICLES`). 규모를
    인자로 받지 않는다 — 배정 단위 10대에는 문서 근거가 있고(A-08), 규모를
    바꿀 수 있게 두면 **증차량이 상수라는 전제가 깨져 후보별 차이가 위치에서만
    나온다는 해석이 무너진다.**

    **기존 44곳의 배속은 건드리지 않는다** — 차감 방식(최근접/여유/비례)이 전부
    임의 기준을 도입하고 신설 공문에도 차감 사례가 없다.
    """
    if placement is None:
        return depots.copy()

    if isinstance(placement, (int, np.integer)):
        path = OUT_DIR / "dong_candidates.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} 가 없다 — src/candidates.py 를 먼저 돌릴 것")
        c = pd.read_csv(path, encoding="utf-8-sig")
        hit = c[c["cand_id"] == placement]
        if hit.empty:
            raise ValueError(f"cand_id {placement} 를 후보 배정 표에서 찾지 못했다")
        r = hit.iloc[0]
        placement = {"name": r["cand_name"], "lat": r["cand_lat"], "lon": r["cand_lon"]}

    new = pd.DataFrame([placement] if isinstance(placement, dict) else placement)
    missing = {"name", "lat", "lon"} - set(new.columns)
    if missing:
        raise ValueError(f"배치안에 없는 컬럼: {sorted(missing)}")
    # 배속은 고정이다. 넘겨받은 값이 있어도 쓰지 않는다(A-10).
    new["capacity"] = NEW_DEPOT_VEHICLES

    out = pd.concat([depots, new], ignore_index=True)
    out["depot_id"] = np.arange(1, len(out) + 1)
    out["capacity"] = out["capacity"].astype(int)
    return out


# ─────────────────────────────────────────────────────────────
# 엔진
# ─────────────────────────────────────────────────────────────

@dataclass
class _Vehicle:
    vid: int
    depot: int                    # 차고지 동 색인
    dong: int                     # 현재 위치(동 색인)
    shift_start: float
    shift_end: float
    job: int = -1                 # 배차된 콜 행번호. -1 이면 없음
    wake: object = None           # 배차 알림 이벤트
    n_trips: int = 0


@dataclass
class _Call:
    row: int                      # calls 프레임의 행번호
    origin: int                   # 동 색인
    dest: int                     # 동 색인(-1 이면 좌표 없음)
    received: float               # 분
    deadline: float               # 접수 + 인내심


class CallTaxiSim:
    """시뮬레이션 본체. 한 배치안 · 한 기간을 재생한다.

    시간 단위는 **재생 시작 자정으로부터의 분**이다. 콜·차량 모두 이 축을 쓰고
    로그로 낼 때만 datetime 으로 되돌린다.

    프로세스는 둘이다.
      `_call_source`  접수 시각 순서로 콜을 큐에 넣는다
      `_vehicle`      차량 하나의 하루(출차 → 배차 대기 → 운행 반복 → 복귀)

    배차는 **이벤트마다 국소적으로** 푼다 — 콜이 오면 그 콜에 맞는 차를, 차가
    비면 그 차에 맞는 콜을 찾는다. 실제 자동배차가 그렇게 돌고, 전역 재매칭을
    매번 돌리면 유휴 차량 × 대기 콜이라 엔진이 그쪽에 묶인다.
    """

    def __init__(self, calls: pd.DataFrame, depots: pd.DataFrame,
                 matrix: TravelMatrix, *, patience: Patience = None,
                 reservation: pd.DataFrame = None, seed: int = SEED,
                 warmdown_min: float = 360.0, idle_break: "IdleBreak" = None,
                 after_assign_cancel_p: float = AFTER_ASSIGN_CANCEL_P,
                 candidate_keep_p: float = CANDIDATE_KEEP_P,
                 candidate_filter_mode: str = CANDIDATE_FILTER_MODE,
                 score_w: dict = None,
                 radius_predawn: float = DISPATCH_RADIUS_PREDAWN,
                 fleet_scale: float = 1.0):
        self.env = simpy.Environment()
        # 이 생성기는 **차량 쪽 표본만** 쓴다(조 편성·휴게 길이). 콜 쪽은 콜 ID
        # 기반 스트림이라 배치안이 달라도 같은 값이 나온다 — uniform_by_id 참조.
        self.rng = np.random.default_rng(seed)
        self.seed = int(seed)
        self.mx = matrix
        self.patience = patience or Patience()
        self.warmdown_min = warmdown_min
        self.idle_break = idle_break
        self.after_assign_cancel_p = float(after_assign_cancel_p)
        self.candidate_keep_p = float(candidate_keep_p)
        self.candidate_filter_mode = candidate_filter_mode
        # 배차 점수 가중치·평일 심야 반경은 민감도 대상이라 인스턴스 값으로 둔다
        # (A-18 · A-17). 넘기지 않으면 모듈 기본값 그대로다.
        self.score_w = dict(score_w or SCORE_W)
        self.radius_predawn = float(radius_predawn)
        # 신규 거점은 **증차**다(A-10) — 기존 배속을 나눠 갖는 것이 아니다.
        # 배속 정원이 691 → 701 로 늘면 그 날 나오는 대수도 같은 비율로 는다.
        self.fleet_scale = float(fleet_scale)

        self.calls = calls.reset_index(drop=True)
        self.t0 = (self.calls["received_at"].min().normalize() if len(self.calls)
                   else pd.Timestamp("2025-01-01"))

        # 예약콜 차감(A-11) — 시간대 × 요일 대수
        self._reserved = (np.zeros((24, 2)) if reservation is None
                          else reservation[["weekday", "weekend"]].to_numpy(float))

        self._depots = depots.reset_index(drop=True)
        self._depot_dong = np.array(
            [self.mx.nearest_index(r.lat, r.lon) for r in self._depots.itertuples()],
            dtype=int)

        self._queue: list[_Call] = []          # 대기 콜
        self._idle: set[int] = set()           # 유휴 차량 vid
        self._vehicles: list[_Vehicle] = []

        n = len(self.calls)
        self.assigned_at = np.full(n, np.nan)
        self.boarded_at = np.full(n, np.nan)
        self.alighted_at = np.full(n, np.nan)
        self.canceled_at = np.full(n, np.nan)
        self.served_by = np.full(n, -1, dtype=np.int32)
        self.events: list[tuple] = []          # (분, 종류, 콜행, 차량) — 추적용
        # 배차 진단 — 반경 안 후보 대수 / 고른 차의 이동시간 / 후보 중앙
        self.pool_sizes: list[int] = []
        self.pool_chosen: list[float] = []
        self.pool_median: list[float] = []
        self.pool_taken: list[float] = []
        self.stats = {"immediate_cancel": 0, "abandoned": 0, "after_assign_cancel": 0,
                      "served": 0, "unresolved": 0,
                      "dispatch_on_call": 0, "dispatch_on_idle": 0, "breaks": 0,
                      "no_acceptor": 0}

    # ── 시간 변환 ─────────────────────────────────────────

    def _to_min(self, ts: pd.Series) -> np.ndarray:
        return (ts - self.t0).dt.total_seconds().to_numpy() / 60.0

    def _stamp(self, minutes: np.ndarray) -> pd.Series:
        return self.t0 + pd.to_timedelta(minutes, unit="m")

    # ── 준비 ──────────────────────────────────────────────

    def _prepare_calls(self):
        c = self.calls
        self._recv = self._to_min(c["received_at"])
        self._origin = np.array(
            [self.mx.index.get(str(d), -1) for d in c["origin_dong_canon"]], dtype=int)
        self._dest = np.array(
            [self.mx.index.get(str(d), -1) for d in c["dest_dong_canon"]], dtype=int)

        # 콜 단위 난수 — call_id 로 고정한다. 배치안·기간·이벤트 순서와 무관하게
        # 같은 콜은 같은 인내심·같은 취소 판정을 받는다(공통난수).
        cid = c["call_id"].to_numpy()
        n = len(c)
        # 즉시 취소(A-14) — 접수 시점의 외생 확률. 인내심 곡선과 별개다.
        self._immediate = uniform_by_id(cid, self.seed, STREAM_IMMEDIATE) < IMMEDIATE_CANCEL_P
        self._patience_min = self.patience.from_uniform(
            uniform_by_id(cid, self.seed, STREAM_PATIENCE))
        # 즉시 취소 시각은 1분 이내 균등 — metrics.cancel_kind 가 'immediate' 로 읽는다
        self._immediate_at = 0.05 + uniform_by_id(cid, self.seed, STREAM_IMMEDIATE_AT) \
            * (IMMEDIATE_CANCEL_MAX_MIN - 0.05)
        # 배차 후 취소 — 배차 시점에 이 판정을 본다
        self._after_assign_cancel = (
            uniform_by_id(cid, self.seed, STREAM_AFTER_ASSIGN) < self.after_assign_cancel_p)

        # 출발 동 좌표가 없으면 배차 위치를 잡을 수 없다. load 가 이미 걸러내므로
        # 여기 걸리면 입력이 잘못된 것이다(조용히 넘기면 그 동이 통째로 사라진다).
        if n and (self._origin < 0).any():
            bad = sorted(set(c.loc[self._origin < 0, "origin_dong_canon"].astype(str)))
            raise ValueError(f"이동시간 행렬에 없는 출발 동: {bad[:5]}")

    def _build_vehicles(self):
        """재생 기간의 날짜마다 차량을 새로 편성한다(요일별 대수 · 조별 출차)."""
        if len(self.calls) == 0:
            return
        days = pd.date_range(self.calls["received_at"].min().normalize(),
                             self.calls["received_at"].max().normalize(), freq="D")
        vid = 0
        for day in days:
            base = FLEET_WEEKEND if day.weekday() >= 5 else FLEET_WEEKDAY
            n_fleet = int(round(base * self.fleet_scale))
            per_depot = allocate_fleet(self._depots, n_fleet)
            # 슬롯 ID = 날짜 × SLOTS_PER_DAY + 슬롯 번호. 거점 순서대로 채우므로
            # 기존 거점의 배속이 유지되면 앞쪽 슬롯의 조 편성도 그대로 유지된다.
            day_idx = (day - self.t0).days
            slot_ids = day_idx * SLOTS_PER_DAY + np.arange(n_fleet)
            start, end = build_shifts(slot_ids, self.seed)

            day_off = (day - self.t0).total_seconds() / 60.0
            k = 0
            for di, cnt in enumerate(per_depot):
                dong = int(self._depot_dong[di])
                for _ in range(int(cnt)):
                    self._vehicles.append(_Vehicle(
                        vid=vid, depot=dong, dong=dong,
                        shift_start=day_off + start[k], shift_end=day_off + end[k]))
                    vid += 1
                    k += 1

    # ── 시각 → 프로필·반경 ────────────────────────────────

    def _clock(self, minutes: float) -> tuple[int, int]:
        """(시, 주말여부). t0 가 자정이라 분에서 바로 뽑는다."""
        day, tod = divmod(minutes, 1440.0)
        return int(tod // 60), int((self.t0.weekday() + int(day)) % 7 >= 5)

    def _radius_km(self, hour: int, weekend: int, minutes: float) -> float:
        tod = minutes % 1440.0
        if DAY_START_MIN <= tod < DAY_END_MIN:
            return DISPATCH_RADIUS_WEEKEND if weekend else DISPATCH_RADIUS_DAY
        # 평일 심야(00:00~06:30)만 따로 뗀다 — 원문에 없는 구간이라 12km 로 본
        # 것이 A-17 이고, 그 가정 자체가 민감도 대상이다. 주말 심야와 19시 이후는
        # 원문에 있으므로 건드리지 않는다.
        if not weekend and tod < DAY_START_MIN:
            return self.radius_predawn
        return DISPATCH_RADIUS_NIGHT

    def _travel(self, o: int, d: int, pi: int, wi: int) -> float:
        """동 o → 동 d 이동시간(분). 좌표가 없는 동이면 시간대 중앙값."""
        if o < 0 or d < 0:
            return float(self.mx.default_minutes[pi, wi])
        return float(self.mx.minutes[pi, wi, o, d])

    # ── 배차 ──────────────────────────────────────────────

    def _prune(self, now: float):
        """인내심을 넘긴 콜을 큐에서 뺀다. 취소 시각은 인내심 만료 시점이다."""
        if not self._queue:
            return
        alive = []
        for call in self._queue:
            if call.deadline <= now:
                self.canceled_at[call.row] = call.deadline
                self.stats["abandoned"] += 1
                self.events.append((call.deadline, "abandon", call.row, -1))
            else:
                alive.append(call)
        self._queue = alive

    def _assign(self, veh: _Vehicle, call: _Call, now: float):
        self._queue.remove(call)
        self._idle.discard(veh.vid)
        veh.job = call.row
        self.assigned_at[call.row] = now
        self.served_by[call.row] = veh.vid
        self.events.append((now, "assign", call.row, veh.vid))
        if veh.wake is not None and not veh.wake.triggered:
            veh.wake.succeed()

    def _match_for_vehicle(self, veh: _Vehicle) -> bool:
        """비어 있는 차량 하나에 대기 콜 중 점수 최고를 붙인다.

        점수는 세 항의 가중합이고 각 항은 **후보 집합 안에서 정규화**한다. 절대
        기준으로 바꾸려면 상한(몇 분을 1점으로 볼지)을 새로 정해야 하는데 그
        값의 근거가 없다. 정규화 방식이 바뀌면 결과가 움직이므로 민감도 대상이다.
        """
        now = self.env.now
        self._prune(now)
        if not self._queue:
            return False
        # 예약콜에 묶여 있어야 할 대수만큼은 유휴로 남긴다(A-11)
        hour, wknd = self._clock(now)
        if len(self._idle) <= self._reserved[hour, wknd]:
            return False

        origins = np.fromiter((c.origin for c in self._queue), dtype=int,
                              count=len(self._queue))
        within = self.mx.km[veh.dong, origins] <= self._radius_km(hour, wknd, now)
        if not within.any():
            return False

        idx = np.flatnonzero(within)
        pi = self.mx.period_index(hour)
        minutes = self.mx.minutes[pi, wknd, veh.dong, origins[idx]]
        recv = np.array([self._queue[i].received for i in idx])

        w = self.score_w
        score = (w["order"] * _rank_score(-recv)
                 + w["wait"] * _minmax(now - recv)
                 + w["dist"] * (1.0 - _minmax(minutes)))
        self._assign(veh, self._queue[int(idx[int(np.argmax(score))])], now)
        self.stats["dispatch_on_idle"] += 1
        return True

    def _match_for_call(self, call: _Call) -> bool:
        """새로 접수된 콜에 유휴 차량 중 이동시간이 가장 짧은 차를 붙인다.

        콜 하나만 놓고 보면 접수순서·경과대기 항이 상수라 **거리항만 순위를
        가른다.** 점수식을 그대로 돌려도 결과가 같아 이동시간 최소로 줄였다.

        **그래서 `score_w` 를 바꿔도 이쪽은 변하지 않는다**(A-18 민감도). 가중치
        손잡이가 무는 곳은 차량 쪽 매칭(`_match_for_vehicle`)뿐이다 — 후보 집합이
        콜 여럿일 때만 세 항이 함께 순위를 만든다.
        """
        now = self.env.now
        hour, wknd = self._clock(now)
        if not self._idle or len(self._idle) <= self._reserved[hour, wknd]:
            return False

        vids = np.fromiter(self._idle, dtype=int, count=len(self._idle))
        dongs = np.array([self._vehicles[v].dong for v in vids])
        within = self.mx.km[dongs, call.origin] <= self._radius_km(hour, wknd, now)
        if not within.any():
            return False

        idx = np.flatnonzero(within)
        pi = self.mx.period_index(hour)
        minutes = self.mx.minutes[pi, wknd, dongs[idx], call.origin]

        # 진단용 — 반경 안에 후보가 몇 대였고 필터 전 최솟값·중앙이 얼마였나
        self.pool_sizes.append(len(idx))
        self.pool_chosen.append(float(minutes.min()))
        self.pool_median.append(float(np.median(minutes)))

        keep = self._filter_candidates(minutes)
        if keep is None:                      # 아무도 받지 않았다 — 큐에 남는다
            self.stats["no_acceptor"] += 1
            return False
        idx, minutes = idx[keep], minutes[keep]

        veh = self._vehicles[int(vids[idx[int(np.argmin(minutes))]])]
        self.pool_taken.append(float(minutes.min()))
        self._assign(veh, call, now)
        self.stats["dispatch_on_call"] += 1
        return True

    def _filter_candidates(self, minutes: np.ndarray):
        """배차 후보를 걸러 남길 색인. 필터가 꺼져 있으면 전부(A-20).

        **random** — 각 후보가 `keep_p` 확률로 살아남는다. 기사 개별 거절에 대응하며,
        가끔은 가장 가까운 차가 살아남아 **분포의 왼쪽 꼬리가 남는다.**

        **drop_near** — 가까운 순으로 `(1−keep_p)` 를 버리고 남은 것 중 최솟값을 쓴다.
        결국 후보 분포의 `(1−keep_p)` 분위를 고르는 것이라 **왼쪽 꼬리가 잘린다.**
        근거리 차량이 체계적으로 막혀 있다는 해석(사전 배정·차량 상태)에 대응한다.

        둘은 같은 중앙값을 내도 **분산이 다르다** — p90 으로 갈린다.

        아무도 안 남으면 None. 그 콜은 큐에 남아 다음 이벤트에서 다시 시도된다.
        """
        n = len(minutes)
        if self.candidate_keep_p >= 1.0 or n <= 1:
            return np.arange(n)
        if self.candidate_filter_mode == "random":
            keep = np.flatnonzero(self.rng.random(n) < self.candidate_keep_p)
            return keep if len(keep) else None
        if self.candidate_filter_mode == "drop_near":
            n_drop = min(int(round((1.0 - self.candidate_keep_p) * n)), n - 1)
            return np.argsort(minutes, kind="stable")[n_drop:]
        raise ValueError(f"모르는 필터 모드: {self.candidate_filter_mode}")

    # ── 프로세스 ──────────────────────────────────────────

    def _call_source(self):
        for row in np.argsort(self._recv, kind="stable"):
            row = int(row)
            wait = self._recv[row] - self.env.now
            if wait > 0:
                yield self.env.timeout(wait)
            now = self.env.now

            if self._immediate[row]:
                self.canceled_at[row] = now + self._immediate_at[row]
                self.stats["immediate_cancel"] += 1
                self.events.append((now, "immediate_cancel", row, -1))
                continue

            call = _Call(row=row, origin=int(self._origin[row]),
                         dest=int(self._dest[row]), received=now,
                         deadline=now + float(self._patience_min[row]))
            self._queue.append(call)
            self.events.append((now, "arrive", row, -1))
            self._match_for_call(call)

    def _vehicle(self, veh: _Vehicle):
        if veh.shift_start > self.env.now:
            yield self.env.timeout(veh.shift_start - self.env.now)
        self.events.append((self.env.now, "shift_start", -1, veh.vid))

        while self.env.now < veh.shift_end:
            if veh.job < 0:
                veh.wake = self.env.event()
                self._idle.add(veh.vid)
                self._match_for_vehicle(veh)
            if veh.job < 0:
                left = veh.shift_end - self.env.now
                if left <= 0:
                    break
                # θ 동안 배차가 없으면 자리를 비운 것으로 본다(A-01: 유휴 ≠ 가용).
                # θ 가 없으면 근무 끝까지 그냥 기다린다.
                span = left if self.idle_break is None else min(self.idle_break.theta, left)
                yield veh.wake | self.env.timeout(span)
                if veh.job < 0 and self.idle_break is not None \
                        and self.env.now < veh.shift_end:
                    self._idle.discard(veh.vid)
                    rest = min(self.idle_break.sample(self.rng),
                               veh.shift_end - self.env.now)
                    self.stats["breaks"] += 1
                    self.events.append((self.env.now, "break", -1, veh.vid))
                    yield self.env.timeout(rest)
                    self.events.append((self.env.now, "resume", -1, veh.vid))
                    continue
            if veh.job < 0:                       # 배차 없이 근무 종료
                break

            row, veh.job = veh.job, -1
            self._idle.discard(veh.vid)
            hour, wknd = self._clock(self.env.now)
            pi = self.mx.period_index(hour)

            # 공차 이동 — 대기 위치 동 → 승차지 동
            origin = int(self._origin[row])
            yield self.env.timeout(self._travel(veh.dong, origin, pi, wknd))

            # 배차 후 취소 — 차가 도착했는데 승차가 없는 건. 차량은 그 자리에
            # 남는다(공차로 갔으므로 위치는 승차지다).
            if self._after_assign_cancel[row]:
                self.canceled_at[row] = self.env.now
                self.stats["after_assign_cancel"] += 1
                self.events.append((self.env.now, "after_assign_cancel", row, veh.vid))
                veh.dong = origin
                continue

            self.boarded_at[row] = self.env.now
            self.events.append((self.env.now, "board", row, veh.vid))

            # 운행 — 승차지 동 → 목적지 동
            hour, wknd = self._clock(self.env.now)
            pi = self.mx.period_index(hour)
            dest = int(self._dest[row])
            yield self.env.timeout(self._travel(origin, dest, pi, wknd))
            self.alighted_at[row] = self.env.now
            # A-01 — 하차 지점 인근에서 제자리 대기한다. 차고지로 돌아가지 않는다.
            veh.dong = dest if dest >= 0 else origin
            veh.n_trips += 1
            self.stats["served"] += 1
            self.events.append((self.env.now, "alight", row, veh.vid))

        self._idle.discard(veh.vid)
        # 근무 종료 — 차고지 복귀(A-01). 복귀 구간은 콜을 만들지 않아 이동시간을
        # 흘리지 않고 위치만 되돌린다(이 차량은 이후 배차 대상이 아니다).
        veh.dong = veh.depot
        self.events.append((self.env.now, "shift_end", -1, veh.vid))

    # ── 실행 ──────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        """재생하고 **실측 콜과 같은 컬럼 구조**의 로그를 돌려준다."""
        self._prepare_calls()
        self._build_vehicles()

        if len(self.calls):
            self.env.process(self._call_source())
            for veh in self._vehicles:
                self.env.process(self._vehicle(veh))
            self.env.run(until=float(self._recv.max()) + self.warmdown_min)
            self._prune(self.env.now)

        return self._build_log()

    def _build_log(self) -> pd.DataFrame:
        """입력 콜의 결과 컬럼만 시뮬 값으로 갈아끼운다.

        나머지 컬럼(출발·목적·좌표·이용목적·시간대…)은 콜의 속성이지 배치의
        결과가 아니라 그대로 둔다. 이래야 `metrics` 가 실측과 시뮬을 **같은
        함수로** 통과시킬 수 있다.
        """
        log = self.calls.copy()
        if len(log) == 0:
            log.attrs["stats"] = dict(self.stats)
            log.attrs["n_vehicles"] = 0
            return log

        # 승차했는데 하차가 없는 건 = 재생 구간 끝에서 잘린 운행. 미해결로 센다.
        cut = ~np.isnan(self.boarded_at) & np.isnan(self.alighted_at)
        boarded = np.where(cut, np.nan, self.boarded_at)
        never = np.isnan(boarded) & np.isnan(self.canceled_at)
        self.stats["unresolved"] = int((cut | never).sum())

        log["assigned_at"] = self._stamp(self.assigned_at)
        log["boarded_at"] = self._stamp(boarded)
        log["alighted_at"] = self._stamp(self.alighted_at)
        log["canceled_at"] = self._stamp(self.canceled_at)
        log["vehicle_id"] = self.served_by

        log["wait_min"] = (boarded - self._recv).astype("float32")
        log["assign_min"] = (self.assigned_at - self._recv).astype("float32")
        log["ride_min"] = (self.alighted_at - boarded).astype("float32")
        # 승차거리는 콜의 속성(어디서 어디로 가는가)이라 그대로 둔다 — 배치가
        # 바꾸는 것은 언제·누가 태우느냐이지 거리가 아니다. 운행이 없으면 뺀다.
        no_ride = np.isnan(boarded)
        log.loc[no_ride, ["ride_distance_m", "fare"]] = np.nan

        # 시뮬에는 '정산 미기록'이 없다 — 승차 시각을 우리가 만들기 때문이다.
        # 실측에서 그 건들이 취소로 새지 않게 갈라둔 플래그이므로 여기서는 전부 False.
        log["is_unsettled"] = False
        log["is_canceled"] = log["boarded_at"].isna()
        log.attrs["stats"] = dict(self.stats)
        log.attrs["n_vehicles"] = len(self._vehicles)
        return log


def _minmax(v: np.ndarray) -> np.ndarray:
    """후보 집합 안에서 0~1 정규화. 폭이 0이면 전부 1(변별력 없음)."""
    lo, hi = v.min(), v.max()
    return np.ones_like(v) if hi <= lo else (v - lo) / (hi - lo)


def _rank_score(v: np.ndarray) -> np.ndarray:
    """순위 점수 — 값이 큰 쪽이 1. 접수순서처럼 간격이 아니라 순서만 쓰는 항."""
    if len(v) == 1:
        return np.ones(1)
    return np.argsort(np.argsort(v)) / (len(v) - 1)


# ─────────────────────────────────────────────────────────────
# 진입점
# ─────────────────────────────────────────────────────────────

def run_placement(placement, calls: pd.DataFrame, depots: pd.DataFrame,
                  travel_time, seed: int = SEED, *,
                  reservation: pd.DataFrame = None,
                  patience: Patience = None,
                  idle_break: IdleBreak = None,
                  after_assign_cancel_p: float = AFTER_ASSIGN_CANCEL_P,
                  candidate_keep_p: float = CANDIDATE_KEEP_P,
                  candidate_filter_mode: str = CANDIDATE_FILTER_MODE,
                  score_w: dict = None,
                  radius_predawn: float = DISPATCH_RADIUS_PREDAWN) -> pd.DataFrame:
    """배치안 하나를 평가. 반환은 `metrics` 에 그대로 넣을 수 있는 콜 로그다.

    placement 는 `resolve_placement` 가 받는 형식(None / cand_id / dict / DataFrame).
    travel_time 은 `TravelMatrix` 이거나 `travel_time.TravelTime` 이다 — 후자를
    주면 여기서 행렬로 감싼다(빌드 ~10초). 여러 배치안을 돌릴 때는 행렬을 한 번만
    만들어 넘길 것.
    """
    if isinstance(travel_time, TravelMatrix):
        mx = travel_time
    else:
        extra = set(calls["origin_dong_canon"].astype(str)) | \
                set(calls["dest_dong_canon"].astype(str))
        mx = TravelMatrix(travel_time, extra)
    resolved = resolve_placement(placement, depots)
    # A-10 — 증차분만큼 가동 대수도 늘린다. 배속 정원 비율이 그 크기다.
    scale = float(resolved["capacity"].sum()) / float(depots["capacity"].sum())
    sim = CallTaxiSim(calls, resolved, mx,
                      patience=patience, reservation=reservation, seed=seed,
                      idle_break=idle_break,
                      after_assign_cancel_p=after_assign_cancel_p,
                      candidate_keep_p=candidate_keep_p,
                      candidate_filter_mode=candidate_filter_mode,
                      score_w=score_w, radius_predawn=radius_predawn,
                      fleet_scale=scale)
    return sim.run()


def slice_period(calls: pd.DataFrame, start=None, days: int = 7) -> pd.DataFrame:
    """재생 구간 잘라내기. 기본은 접수 첫날부터.

    **대표 기간은 `REPRESENTATIVE_START` · `REPRESENTATIVE_DAYS` 다**(A-03).
    시뮬 결과는 연간 검증 기준이 아니라 **그 구간의 실측값**과 대조한다 —
    자릿수와 이유는 docs/provenance.md 13절.
    """
    first = calls["received_at"].min().normalize() if start is None else pd.Timestamp(start)
    last = first + pd.Timedelta(int(days), unit="D")
    m = (calls["received_at"] >= first) & (calls["received_at"] < last)
    return calls[m].reset_index(drop=True)


if __name__ == "__main__":
    import time

    sys.stdout.reconfigure(encoding="utf-8")

    print("입력 로딩...")
    t0 = time.perf_counter()
    calls = load.load_calls()
    depots = load.load_depots()
    reservation = load.load_reservation_occupancy()
    tt = T.TravelTime.build()
    print(f"  콜 {len(calls):,} · 차고지 {len(depots)}곳(배속 정원 합 "
          f"{depots['capacity'].sum()}대) · {time.perf_counter() - t0:.1f}초")

    t0 = time.perf_counter()
    mx = TravelMatrix(tt, set(calls["origin_dong_canon"].astype(str))
                      | set(calls["dest_dong_canon"].astype(str)))
    print(f"이동시간 행렬 {mx.minutes.shape} · {mx.minutes.nbytes / 1e6:.1f}MB · "
          f"빌드 {time.perf_counter() - t0:.1f}초")

    import metrics as M

    for days in (1, 7, REPRESENTATIVE_DAYS):
        sub = slice_period(calls, start=REPRESENTATIVE_START, days=days)
        t0 = time.perf_counter()
        log = run_placement(None, sub, depots, mx, reservation=reservation)
        elapsed = time.perf_counter() - t0
        st = log.attrs["stats"]
        w = log["wait_min"].dropna()
        print(f"\n[{days}일] 콜 {len(sub):,} · 차량-일 {log.attrs['n_vehicles']:,} · "
              f"실행 {elapsed:.1f}초")
        print(f"  승차 {st['served']:,} / 즉시취소 {st['immediate_cancel']:,} / "
              f"포기 {st['abandoned']:,} / 미해결 {st['unresolved']:,}")
        # 대조는 **같은 구간의 실측**이다 — 연간 기준이 아니다(A-03, provenance 13절).
        # 판정은 픽업·취소·지니만 하고 총 대기·매칭은 참고로 싣는다(A-19).
        tg, ref = M.period_targets(sub)
        print(M.compare_to_target(M.overall_metrics(log), targets=tg, reference=ref)
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
