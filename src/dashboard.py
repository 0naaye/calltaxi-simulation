# -*- coding: utf-8 -*-
"""
dashboard.py — 동별 대기 실태 및 후보 효과 대시보드

행정동별 대기 현황과 거점 후보의 시뮬레이션 결과를 조회하는 Streamlit 대시보드다.

화면 구성과 지표 정의는 `docs/dashboard.md`를 따른다.

기준 기간
- 진단 화면: 2025년 연간
- 시뮬레이션 화면: 대표 기간

실행
streamlit run src/dashboard.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate as E  # noqa: E402
import load  # noqa: E402
import metrics as M  # noqa: E402
import simulator as S  # noqa: E402

OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"

APP_TITLE = "서울 장애인콜택시 대기 진단"
MAP_TITLE = "동별 대기 실태"
BASIS = "2025년 연간"

# ─────────────────────────────────────────────────────────────
# 상수 — 화면 규칙
# ─────────────────────────────────────────────────────────────

# 45분 미만을 한 색으로 접는 경계. 기관 기준이 아니라 시선을 좁히는 장치다.
# 근거 있는 임계가 생기면 여기만 바꾼다(명세 4절).
RED_THRESHOLD_MIN = 45.0

# 경계 단순화(도). 0.0002 ≈ 22m — 0.90MB → 0.40MB, 빈/불량 폴리곤 0 (명세 7절)
SIMPLIFY_TOL = 0.0002

# 타일 없는 배경. 외부 타일에 의존하면 오프라인에서 지도가 빈다(명세 7절).
MAP_STYLE = "white-bg"
MAP_CENTER = {"lat": 37.5665, "lon": 126.9780}
MAP_ZOOM = 10.2

# 색 — 데이터 색만 의미를 갖는다. 채도를 낮춰 형광이 되지 않게(명세 8절)
GREEN = "#5B8C6E"          # 접힌 구간 단일색
RED_LIGHT = "#E2A79E"      # 기준선
RED_DARK = "#8C2F26"       # 최댓값
GREY_LIGHT = "#DAD8D3"     # 회색조 하단
GREY_DARK = "#8B8880"      # 회색조 상단
NO_MATCH = "#EDEBE7"       # 지표 매칭 없음
INK = "#22201D"
BG = "#F4F3F0"

# 점 레이어 두 종 — 색으로 구분한다. 현행 차고지는 짙은 남색(운영 중),
# 후보는 주황(아직 아님). 크기도 9 vs 7 로 갈라 겹쳐도 읽히게 한다.
DEPOT_COLOR = "#2F4858"
CAND_COLOR = "#C8722B"
# 후보 점 클릭을 코로플레스 클릭과 구분하는 태그(customdata 첫 칸)
CAND_TAG = "cand"
# customdata 에서 cand_id 가 앉는 칸. build_map 이 싣고 클릭 처리기가 꺼낸다 —
# 칸이 밀리면 엉뚱한 후보가 열리므로 양쪽이 이 상수를 함께 본다.
CAND_ID_IDX = 6
PANEL_BG = "#FBFAF8"

PERIOD_ORDER = ["심야", "아침", "낮", "저녁"]

# ── 3페이지(시뮬) 전용 ────────────────────────────────────────
# 개선 농도 램프. 개선율이 클수록 짙다. **한 후보의 지도 안에서는 평평하다** —
# 후보별 before/after 는 3km 범위 전체의 집계 하나뿐이고 동별로는 없다.
# 농도는 후보끼리 비교하라고 두는 것이다(build_sim_map 주석 참조).
IMPROVE_LIGHT = "#D5E3D8"
IMPROVE_DARK = "#2E5E43"
OUT_SCOPE = "#E3E1DC"        # 영향권 밖 — 물러나되 사라지지는 않게
ASSIGNED_LINE = "#1F4230"    # 담당 동 테두리

# 시뮬 화면의 기준 기간. 진단 화면(연간)과 다르다 — 섞어 읽으면 안 되므로
# **시뮬 상수에서 끌어온다.** 화면에 직접 적으면 대표 기간이 바뀔 때 어긋난다.
SIM_START = pd.Timestamp(S.REPRESENTATIVE_START)
SIM_DAYS = S.REPRESENTATIVE_DAYS
# `pd.Timedelta(days=N)` 는 이 pandas 에서 DeprecationWarning 이다 — 단위를 명시한다.
SIM_END = SIM_START + pd.Timedelta(SIM_DAYS - 1, unit="D")
SIM_BASIS = f"대표 기간 {SIM_START:%Y-%m-%d}~{SIM_END:%Y-%m-%d}"

# 픽업 재현 수준(A-19·A-20) — 시뮬 픽업이 실측의 이만큼이다.
#
# **이 값을 "절감이 그만큼 축소됐다"로 읽으면 안 된다.** 편의는 양방향이다 —
# 매칭·배차 미재현(A-19·A-20)은 절감을 줄이는 쪽이고, θ 를 두지 않은 것(A-01)은
# 늘리는 쪽이다(θ=60 이면 대표 9곳 평균 개선율이 −18%, 최상위는 −28% 준다 —
# docs/model_flow.md 「민감도」). 두 편의가 상쇄되는 크기를 알 수 없으므로
# **절대값을 인용하지 말고 후보 간 비교로만 읽는다.** 화면 문구는 그 문장이다.
PICKUP_FIDELITY_PCT = 53
BIAS_NOTE = ("절감폭의 절대 크기는 양방향 편의를 안고 있다. 매칭·배차 미재현"
             "(A-19·A-20)은 절감을 줄이는 쪽이고, θ 를 두지 않은 것(A-01)은 늘리는 "
             "쪽이다. 두 편의가 상쇄되는 크기를 알 수 없으므로 절대값을 인용하지 "
             "말고 후보 간 비교로만 읽는다.")

# 총 대기(접수 → 승차) — **분 단위를 낸다**(08.12 개정). 예전에는 개선율만 냈다.
# 분을 보이면 진단 화면의 실측 대기와 견주게 된다는 이유였는데, **그 대비가 바로
# 이 블록의 목적이다** — 시뮬 총 대기 before 가 실측의 0.30~0.49배라는 사실이
# 매칭 미재현(A-19)의 증거이고, 그것이 채점 지표를 픽업으로 옮긴 근거다.
# 숨기면 근거가 화면에서 사라지고 "왜 픽업으로 재나"가 문서에만 남는다.
#
# **그래서 실측을 반드시 같은 자리에 낸다**(`obs_total_min`). 분만 내고 실측을
# 빼면 옛 걱정이 그대로 살아난다 — 값이 서울 평균 42.7분과 곧장 견줘진다.
# 실측이 없는 행은 분을 내지 않고 개선율로 돌아간다(`sim_total_wait`).
#
# 이 문장은 툴팁이 아니라 값 옆에 그대로 적는다 — **툴팁은 부연을 접는 자리이지
# 근거를 옮겨 놓는 자리가 아니다**(명세 3절). 숫자를 인용할 때 반드시 따라가야
# 하는 말이므로 접으면 안 된다.
#
# **바로 위 실측 대비 줄과 겹치지 않게 줄였다**(08.12). "매칭을 재현하지 못해
# 짧다"는 그 줄이 이미 수치와 함께 말한다 — 같은 말을 두 번 하면 아래 문장이
# 읽히지 않고, 그러면 정작 지켜야 할 「인용하지 말라」가 묻힌다.
TOTAL_WAIT_NOTE = "절대값을 인용하지 말고 후보 간 비교로 읽으십시오."
# 물음표로 접는 쪽 — 두 지표가 무엇인지, 그리고 **순위 기준은 픽업 그대로**라는 것.
# 총 대기 개선율이 픽업보다 크게 나오므로(1.3~2.2배) 정렬이 그쪽으로 바뀐 게
# 아니냐는 물음이 생긴다. 기준을 옮기지 않은 이유는 model_flow 에 있다.
TOTAL_WAIT_HELP = ("총 대기는 접수부터 승차까지, 픽업은 배차부터 승차까지입니다. "
                   "후보 목록 정렬과 분포는 픽업 개선율 기준 그대로입니다.")

# ─────────────────────────────────────────────────────────────
# 색칠 지표 — 사이드바에서 고른다.
#
# fold  기준선. 숫자면 절대값, "p75"/"p25" 면 폴리곤 분포의 백분위.
#       아래는 한 색으로 접고 위만 농도를 준다 — 시선을 좁히는 장치(명세 4절).
# worse "high" 면 클수록 나쁨, "low" 면 작을수록 나쁨(공급 지표).
#
# 평균 대기만 절대값(45분)이다. 시뮬 before/after 에서 개선이 보여야 하는데
# 백분위는 무엇을 넣든 항상 상위 25%가 빨강이라 개선이 드러나지 않는다.
# 진단과 시뮬을 같은 기준으로 읽으려면 절대값이어야 한다.
# 나머지는 근거 있는 절대 임계가 없어 상위 25%(공급 지표는 하위 25%)를 쓴다.
#
# ⚠ 45분과 p75 의 일치는 깨졌다(특장차 한정 모집단 · A-16). 구 모집단에서는 45분이
# 마침 p75(45.5분)라 네 지표가 같은 논리로 정렬됐는데, 지금은 p75 = 46.80분이고
# 45분은 p67.3 지점이다. 평균 대기만 빨강이 137개(32.7%)로 다른 셋(105개 · 25.1%)보다
# 넓다. 그래도 절대값을 유지하는 이유는 위 문단 그대로다 — 일치는 편의였지 근거가
# 아니었다. 좁히려면 RED_THRESHOLD_MIN 만 바꾸되 진단과 시뮬의 자가 달라진다.
#
# 이용률은 넣지 않는다. 낮은 이용률이 미충족 수요인지 낮은 수요인지 자료로
# 가릴 수 없어 방향을 정할 수 없다. 방향을 임의로 정하면 도구가 답을 내는 게
# 된다(명세 4절). 패널에는 그대로 표시한다.
# ─────────────────────────────────────────────────────────────

METRICS = {
    "wait_total_mean": dict(label="평균 대기", unit="분", digits=1,
                            fold=RED_THRESHOLD_MIN, worse="high"),
    "long_wait_ratio": dict(label="60분 초과 비율", unit="%", digits=1,
                            fold="p75", worse="high", scale=100),
    "cancel_abandoned_ratio": dict(label="기다리다 포기", unit="%", digits=1,
                                   fold="p75", worse="high", scale=100),
    "vehicles_3km": dict(label="3km 내 차량", unit="대", digits=0,
                         fold="p25", worse="low"),
}
DEFAULT_METRIC = "wait_total_mean"

# 폴리곤 단위로 가중평균해 둘 지표(지도 색칠용)
POLY_COLS = list(METRICS)


# ─────────────────────────────────────────────────────────────
# 로딩 — 전부 캐시한다(명세 7절)
# ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_metrics() -> pd.DataFrame:
    """동별 지표 432행."""
    df = pd.read_csv(OUTPUTS / "dong_metrics.csv")
    df["base"] = df["dong_canon"].map(load._base_dong)
    return df


@st.cache_data(show_spinner=False)
def load_demand() -> pd.DataFrame:
    """동 × 시간대 × 평일/주말 콜 수."""
    return pd.read_parquet(OUTPUTS / "dong_demand_matrix.parquet")


@st.cache_data(show_spinner=False)
def load_boundary():
    """단순화한 서울 동 경계. (geojson dict, 속성표) 로 돌려준다.

    GeoDataFrame 은 캐시 직렬화가 무거워 dict + DataFrame 으로 쪼갠다.

    **표시 전용 경계다**(`load.load_display_boundary`). `data/` 에 원본이 있으면
    원본을 단순화하고, 없으면 저장소 사본(`assets/`)을 쓴다 — clone 만으로 지도가
    뜨게 하려는 것이다. 면적·거리 계산은 이 경계를 쓰면 안 된다(22m 단순화).
    """
    gdf = load.load_display_boundary().copy()
    gdf["base"] = gdf["dong_canon"].map(load._base_dong)
    gj = json.loads(gdf.to_json())
    for feat, cd in zip(gj["features"], gdf["adm_cd2"]):
        feat["id"] = cd
    attrs = pd.DataFrame({
        "adm_cd2": gdf["adm_cd2"].tolist(),
        "adm_nm": gdf["adm_nm"].tolist(),
        "gu": gdf["gu"].tolist(),
        "dong_canon": gdf["dong_canon"].tolist(),
        "base": gdf["base"].tolist(),
    })
    return gj, attrs


@st.cache_data(show_spinner=False)
def load_depots() -> pd.DataFrame:
    return load.load_depots()


@st.cache_data(show_spinner=False)
def load_candidates() -> pd.DataFrame:
    """지도에 올릴 후보 — **동에 실제로 배정된 곳만**.

    옥외 후보는 348곳이지만 시뮬이 쓰는 건 `candidates.assign_dong_candidates` 가
    동에 배정한 244곳이다. 348곳을 다 찍으면 **지도가 시뮬 입력과 다른 것을 보여준다**
    — 실무자는 점 하나하나를 "여기에 놓을 수 있다"로 읽는데, 배정되지 않은 104곳은
    어느 동의 거점도 되지 않는다.

    한 후보가 여러 동에 배정될 수 있어(대체 배정) 배정 동 목록을 함께 붙인다.
    `dong_candidates.csv` 가 없으면 348곳 전체로 물러난다 — 지도가 비는 것보다 낫다.

    **`load.load_candidates()` 의 attrs 를 그대로 물려준다.** `merge` 는 attrs 를
    버린다 — 예전에 사이드바가 `n_pool` 을 읽다가 KeyError 로 죽은 적이 있다.
    그 문구는 08.10 에 걷어냈지만 `attrs.update` 는 남긴다: 출처의 메타데이터를
    로더가 조용히 잃는 것 자체가 버그이고, 다음에 누가 읽어도 있어야 한다.

    반환: load.load_candidates() 컬럼 + assigned_dongs(문자열), n_assigned_dongs
    """
    cand = load.load_candidates()
    path = OUTPUTS / "dong_candidates.csv"
    if not path.exists():
        cand = cand.assign(assigned_dongs="", n_assigned_dongs=0)
        cand.attrs["n_outdoor"] = len(cand)
        cand.attrs["assigned_only"] = False
        return cand

    a = pd.read_csv(path)
    a = a[a["is_assigned"] & a["cand_id"].notna()]
    per = (a.assign(label=a["gu"] + " " + a["dong_canon"])
            .groupby("cand_id")["label"]
            .agg(assigned_dongs=lambda s: " · ".join(sorted(s)),
                 n_assigned_dongs="size"))

    n_outdoor = len(cand)
    out = cand.merge(per, left_on="cand_id", right_index=True, how="inner")
    out.attrs.update(cand.attrs)          # merge 가 버린 원본 attrs 를 되살린다
    out.attrs["n_outdoor"] = n_outdoor
    out.attrs["assigned_only"] = True
    return out.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# 시뮬 결과 — 3페이지
# ─────────────────────────────────────────────────────────────

# 화면에 쓰지 않는 열. 등급은 계산만 남기고 표시하지 않기로 했다 —
# 판단을 대신하는 표시라 도구의 성격과 어긋나고, 244곳에서 [상] 4곳 중 3곳이
# 표본 부족이라 최고 등급이 가장 좁은 지역을 가리켰다.
# 근거는 docs/model_flow.md 「등급은 계산하되 표시하지 않는다」.
REF_PREFIX = "ref_"


@st.cache_data(show_spinner=False)
def load_placement() -> pd.DataFrame:
    """후보별 시뮬 결과 244행. **`ref_*`(등급) 열은 아예 떨어뜨린다.**

    화면에서 안 쓰는 것으로 충분하지 않다 — 열이 남아 있으면 다음 사람이
    집어 쓴다. 여기서 자르면 3페이지 코드가 등급에 손댈 방법이 없다.

    **총 대기 열은 자르지 않는다.** 참고 지표지만 화면이 쓴다 — 등급과 성격이
    다르다. 등급은 판단을 대신하는 표시라 잘라냈고, 이쪽은 「참고」 꼬리표를 달고
    보여주는 값이다(`sim_total_wait`).

    분 단위 3열(`before_total`·`after_total`·`obs_total_min`)도 그대로 둔다 —
    실측과의 대비가 「왜 픽업으로 재는가」의 근거라 화면이 쓴다(08.12 개정).

    반환: cand_id, cand_name, gu, dongs, n_dong_scope, n_calls_scope,
          before, delta, rate_pct, rate_sd, rate_lo, rate_hi, interval,
          sample_ok, total_delta_min, n_overlap, rate_total_pct,
          rate_total_sd, before_total, after_total, obs_total_min,
          sim_obs_ratio (+ after 파생)
    """
    path = OUTPUTS / "placement_grades.csv"
    if not path.exists():
        return pd.DataFrame()
    g = pd.read_csv(path, encoding="utf-8-sig")
    g = g[[c for c in g.columns if not c.startswith(REF_PREFIX)]].copy()
    g["after"] = g["before"] + g["delta"]
    g["half"] = (g["rate_hi"] - g["rate_lo"]) / 2
    return g


@st.cache_data(show_spinner=False)
def load_placement_seeds() -> pd.DataFrame:
    """시드별 원값 732행. 후보 하나의 시드 3회를 그대로 보여줄 때 쓴다.

    평균 하나만 보이면 σ 가 어디서 왔는지 알 수 없다 — 원값 3개를 열어 두면
    "이 후보는 시드에 따라 이만큼 흔들린다"를 직접 확인할 수 있다.
    """
    path = OUTPUTS / "placement_eval.csv"
    if not path.exists():
        return pd.DataFrame()
    ev = pd.read_csv(path, encoding="utf-8-sig")
    return ev.drop_duplicates(["cand_id", "seed"], keep="first")


@st.cache_data(show_spinner=False)
def placement_scope() -> dict:
    """후보 → (3km 영향권 폴리곤, 담당 동 폴리곤). 둘 다 adm_cd2 집합이다.

    **영향권은 저장돼 있지 않아 다시 계산한다.** `placement_eval.csv` 에는
    범위 안 동 **수**(`n_dong_scope`)만 있고 어느 동인지는 없다. `evaluate`
    의 함수를 그대로 불러 같은 반경·같은 좌표로 다시 잡는다 — 규칙을 여기
    베껴 쓰면 반경이 바뀔 때 지도와 숫자가 어긋난다. 244곳 전부에서
    `n_dong_scope` 와 일치하는 것을 확인했다(tests/test_dashboard.py).

    담당 동은 `dong_candidates.csv` 의 배정 결과다. 영향권과 다르다 —
    영향권은 "3km 안에 드는 동", 담당은 "이 후보를 거점으로 배정받은 동"이다.

    **폴리곤 수는 동 수보다 적을 수 있다.** 경계 파일(2023)과 콜 원본의 동
    구분이 달라 ⑴ 대응 경계가 없는 지표 행이 7개 있고 ⑵ 개편으로 여러 행이 한
    경계에 묶인다(신당1~6동 → 신당동 등 4곳). 244곳 중 66곳이 여기 걸린다.
    그래서 `n_dong`(진짜 영향권 동 수)과 `polys`(그릴 수 있는 경계)를 **따로**
    돌려준다 — 지도가 17개를 칠하면서 "18개 동"이라고 적으면 안 된다.
    같은 어긋남을 2페이지는 `orphan_rows` 로 이미 알리고 있다(명세 5·9절).

    반환: {cand_id: {polys, assigned, lat, lon, n_dong, n_unmapped}}
    """
    ac = OUTPUTS / "dong_candidates.csv"
    if not ac.exists():
        return {}
    a = pd.read_csv(ac)
    a = a[a["is_assigned"] & a["cand_id"].notna()]

    # 지표 행 → 폴리곤. resolve_polygons 의 3단 매칭 결과를 거꾸로 뒤집는다.
    poly = resolve_polygons()
    metrics_df = load_metrics()
    key_of = list(zip(metrics_df["gu"], metrics_df["dong_canon"]))
    codes = poly["adm_cd2"].tolist()
    poly_of_key: dict = {}
    for pi, members in enumerate(poly["members"]):
        for j in members:
            poly_of_key[key_of[j]] = codes[pi]
    known = set(codes)

    dong_xy = E.dong_coords()
    out = {}
    for cid, grp in a.groupby("cand_id"):
        r = grp.iloc[0]
        keys = E.scope_members(float(r["cand_lat"]), float(r["cand_lon"]), dong_xy)
        polys = {poly_of_key[k] for k in keys if k in poly_of_key}
        # 담당 동의 adm_cd2 는 int 로 실려 있다 — 경계 쪽은 문자열이다
        assigned = {str(x) for x in grp["adm_cd2"]} & known
        out[int(cid)] = {
            "polys": sorted(polys), "assigned": sorted(assigned),
            "lat": float(r["cand_lat"]), "lon": float(r["cand_lon"]),
            "n_dong": len(keys),
            "n_unmapped": len(keys) - len(polys),
        }
    return out


def placement_row(cand_id: int):
    """후보 하나의 결과 행. 없으면 None — 결과가 없는 후보도 지도에는 있다."""
    g = load_placement()
    if g.empty:
        return None
    hit = g[g["cand_id"] == cand_id]
    return hit.iloc[0] if len(hit) else None


def candidate_options(sort_key: str) -> list:
    """사이드바 후보 목록 — (cand_id, 표시문자열) 을 정렬해 돌려준다.

    **순위 번호를 붙이지 않는다.** 정렬은 훑어보기 위한 것이지 서열이 아니다 —
    개선율 1위 마천동1 은 다른 24곳과 대비 구간이 겹쳐 단독 1위라 할 수 없다.
    번호를 붙이면 그 사실이 지워진다.

    **개선율과 총절감을 라벨에 함께 적는다.** 정렬 키만 보이면 그 축이 유일한
    기준인 것처럼 읽힌다 — 둘은 순위 상관 +0.32 로 다른 것을 잰다.
    """
    g = load_placement()
    if g.empty:
        return []
    col = "total_delta_min" if sort_key == "total" else "rate_pct"
    g = g.sort_values(col)
    out = []
    for r in g.itertuples():
        flag = "" if r.sample_ok else "  ※표본부족"
        out.append((int(r.cand_id),
                    f"{r.cand_name[:22]} ({r.gu}) · {r.rate_pct:+.2f}% · "
                    f"{r.total_delta_min:,.0f}분{flag}"))
    return out


# ─────────────────────────────────────────────────────────────
# 경계 ↔ 지표 매칭 — load.build_dong_lookup 의 3단 규칙을 폴리곤 방향으로
# 적용한다. 규칙과 근거는 명세 5절.
# ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def resolve_polygons() -> pd.DataFrame:
    """폴리곤 1행 = 지도에 칠할 단위.

    반환: adm_cd2, adm_nm, gu, dong_canon, match_type, members(지표 행 인덱스),
          member_names, n_served, is_reliable, has_metrics + POLY_COLS 가중평균
    """
    df = load_metrics()
    _, attrs = load_boundary()

    midx = {(g, d): i for i, (g, d) in enumerate(zip(df["gu"], df["dong_canon"]))}
    assign: dict = {}
    kind: dict = {}
    used: set = set()

    # 1단계 exact
    for pi, (gu, dc) in enumerate(zip(attrs["gu"], attrs["dong_canon"])):
        j = midx.get((gu, dc))
        if j is not None:
            assign[pi] = [j]
            kind[pi] = "exact"
            used.add(j)

    # 2단계 manual — 행정구역 개편으로 이름이 바뀐 동
    poly_at = {(g, d): pi
               for pi, (g, d) in enumerate(zip(attrs["gu"], attrs["dong_canon"]))}
    for (gu, src), target in load.MANUAL_DONG.items():
        j = midx.get((gu, load.canon_dong(src)))
        if j is None or j in used:
            continue
        pi = poly_at.get((gu, load.canon_dong(target)))
        if pi is None:
            continue
        assign.setdefault(pi, []).append(j)
        kind[pi] = "exact+manual" if kind.get(pi) == "exact" else "manual"
        used.add(j)

    # 3단계 approx — 기본명 그룹의 '잔여' 행만. exact 로 소비된 행을 다시 쓰면
    # 없는 데이터를 지어낸다(명세 5절).
    leftover = df[~df.index.isin(used)]
    for pi, (gu, b) in enumerate(zip(attrs["gu"], attrs["base"])):
        if pi in assign:
            continue
        cand = leftover[(leftover["gu"] == gu) & (leftover["base"] == b)]
        if len(cand):
            assign[pi] = list(cand.index)
            kind[pi] = "approx"
            used.update(cand.index)

    rows = []
    for pi in range(len(attrs)):
        a = attrs.iloc[pi]
        js = assign.get(pi, [])
        rec = {
            "adm_cd2": a["adm_cd2"], "adm_nm": a["adm_nm"], "gu": a["gu"],
            "dong_canon": a["dong_canon"],
            "match_type": kind[pi] if js else "none",
            "members": js, "member_names": [],
            "n_served": 0, "is_reliable": True, "has_metrics": bool(js),
        }
        if js:
            sub = df.loc[js]
            w = sub["n_served"].clip(lower=0)
            tot = w.sum()
            rec["member_names"] = sub["dong"].tolist()
            rec["n_served"] = int(tot)
            rec["is_reliable"] = bool((sub["n_served"] >= 100).all())
            for col in POLY_COLS:
                # 승차 완료 가중 평균 — 24건 동과 4,000건 동을 같은 무게로 섞지 않는다
                rec[col] = ((sub[col] * w).sum() / tot) if tot else float("nan")
        else:
            for col in POLY_COLS:
                rec[col] = float("nan")
        rows.append(rec)

    out = pd.DataFrame(rows)
    out["wait"] = out["wait_total_mean"]        # 요약·범례에서 쓰는 별칭
    out.attrs["orphan_rows"] = df[~df.index.isin(used)][
        ["gu", "dong", "n_served", "wait_total_mean"]].to_dict("records")
    return out


@st.cache_data(show_spinner=False)
def seoul_summary() -> dict:
    """상단 요약. 개별 동 값을 읽을 때의 비교 기준이 된다."""
    df = load_metrics()
    poly = resolve_polygons()
    rel = df[df["is_reliable"]]
    return {
        "mean_wait": float(df["wait_total_mean_seoul_mean"].iloc[0]),
        "n_red": int((poly["wait"] >= RED_THRESHOLD_MIN).sum()),
        "n_colored": int(poly["has_metrics"].sum()),
        # 지역 지니 — 승차 100건 이상 동, 동 균등가중(가정 A-12)
        "gini": float(M.gini(rel["wait_total_mean"])),
        "n_reliable": int(len(rel)),
    }


def metric_fold(poly: pd.DataFrame, key: str) -> float:
    """색을 접는 기준선(원 단위). 절대값이거나 폴리곤 분포의 백분위다.

    백분위는 **색칠 대상 폴리곤 419개** 위에서 잡는다. 지표 행 432 기준이
    아니라 실제로 지도에 칠해지는 모집단이라야 "빨강 25%"가 성립한다.
    """
    spec = METRICS[key]
    fold = spec["fold"]
    if isinstance(fold, (int, float)):
        return float(fold)
    q = float(str(fold).lstrip("p")) / 100.0
    vals = poly.loc[poly["has_metrics"], key].dropna()
    return float(vals.quantile(q))


@st.cache_data(show_spinner=False)
def metric_summary() -> dict:
    """지표별 (기준선, 빨강 폴리곤 수). 사이드바 범례에 쓴다."""
    poly = resolve_polygons()
    out = {}
    for key, spec in METRICS.items():
        sc = spec.get("scale", 1)
        fold = metric_fold(poly, key) * sc
        vals = poly.loc[poly["has_metrics"], key].dropna() * sc
        n_red = int((vals >= fold).sum()) if spec["worse"] == "high" \
            else int((vals <= fold).sum())
        out[key] = (fold, n_red)
    return out


# ─────────────────────────────────────────────────────────────
# 색
# ─────────────────────────────────────────────────────────────

def fold_colorscale(vmin: float, vmax: float, fold: float) -> list:
    """기준선 미만 단일 초록, 이상은 옅은→진한 빨강. 하드 스톱(명세 4절)."""
    if vmax <= vmin:
        return [[0.0, GREEN], [1.0, GREEN]]
    t = (fold - vmin) / (vmax - vmin)
    t = min(max(t, 0.0), 1.0)
    if t <= 0:
        return [[0.0, RED_LIGHT], [1.0, RED_DARK]]
    if t >= 1:
        return [[0.0, GREEN], [1.0, GREEN]]
    eps = 1e-6
    return [[0.0, GREEN], [t - eps, GREEN], [t, RED_LIGHT], [1.0, RED_DARK]]


def grey_colorscale() -> list:
    """채도를 버리고 명암만 남긴다 — 물러나되 사라지지는 않게."""
    return [[0.0, GREY_LIGHT], [1.0, GREY_DARK]]


def metric_z(poly: pd.DataFrame, key: str):
    """(z, zmin, zmax, colorscale, 표시용 원값).

    공급 지표(worse='low')는 부호를 뒤집어 같은 스케일 로직을 그대로 쓴다 —
    작을수록 빨강이 된다.
    """
    spec = METRICS[key]
    has = poly[poly["has_metrics"]]
    raw = has[key].astype(float)
    scale = spec.get("scale", 1)
    fold = metric_fold(poly, key) * scale
    vals = raw * scale
    if spec["worse"] == "low":
        z = (-vals).tolist()
        zmin, zmax = float(-vals.max()), float(-vals.min())
        cs = fold_colorscale(zmin, zmax, -fold)
    else:
        z = vals.tolist()
        zmin, zmax = float(vals.min()), float(vals.max())
        cs = fold_colorscale(zmin, zmax, fold)
    return z, zmin, zmax, cs, vals.tolist(), fold


# ─────────────────────────────────────────────────────────────
# 지도
# ─────────────────────────────────────────────────────────────

def build_map(poly: pd.DataFrame, selected, show_depots: bool,
              show_cands: bool, metric_key: str) -> go.Figure:
    """지도 figure.

    geojson·locations 를 rerun 마다 동일하게 유지해 실제로 달라지는 것이
    z(색 배열)뿐이도록 만든다. uirevision 이 붙어 있으면 Plotly 가 이를
    갱신으로 처리해 카메라를 유지하고 transition 을 적용한다(명세 7절).
    """
    gj, _ = load_boundary()
    has = poly[poly["has_metrics"]]
    spec = METRICS[metric_key]
    z, zmin, zmax, cs, shown, _fold = metric_z(poly, metric_key)

    locs = has["adm_cd2"].tolist()
    names = has["adm_nm"].tolist()
    unit, digits = spec["unit"], spec["digits"]
    cdata = [[n, v] for n, v in zip(names, shown)]
    htmpl = (f"%{{customdata[0]}}<br>{spec['label']} "
             f"%{{customdata[1]:,.{digits}f}}{unit}<extra></extra>")

    fig = go.Figure()

    # 매칭 없는 폴리곤 — 빼지 않는다. 구멍이 뚫리면 '값이 좋은 동'으로 읽힌다.
    none = poly[~poly["has_metrics"]]
    if len(none):
        fig.add_trace(go.Choroplethmap(
            geojson=gj, locations=none["adm_cd2"].tolist(),
            z=[0] * len(none), colorscale=[[0, NO_MATCH], [1, NO_MATCH]],
            showscale=False, marker_line_width=0.4, marker_line_color="#C9C6C0",
            hovertext=none["adm_nm"].tolist(),
            hovertemplate="%{hovertext}<br>지표 매칭 없음<extra></extra>",
            name="지표 매칭 없음",
        ))

    if selected is None:
        fig.add_trace(go.Choroplethmap(
            geojson=gj, locations=locs, z=z, zmin=zmin, zmax=zmax, colorscale=cs,
            marker_line_width=0.4, marker_line_color="#FFFFFF",
            marker_opacity=0.88, customdata=cdata, hovertemplate=htmpl,
            showscale=False, name=spec["label"],
        ))
    else:
        # 회색조 바닥 + 선택 동만 컬러.
        # 선택 외 z 를 None 으로 두면 색이 칠해지지 않는다(트레이스는 유지).
        fig.add_trace(go.Choroplethmap(
            geojson=gj, locations=locs, z=z, zmin=zmin, zmax=zmax,
            colorscale=grey_colorscale(), showscale=False,
            marker_line_width=0.4, marker_line_color="#FFFFFF",
            marker_opacity=0.75, customdata=cdata, hovertemplate=htmpl,
            name="그 외 동",
        ))
        sel_z = [v if loc == selected else None for loc, v in zip(locs, z)]
        fig.add_trace(go.Choroplethmap(
            geojson=gj, locations=locs, z=sel_z, zmin=zmin, zmax=zmax,
            colorscale=cs, showscale=False, marker_line_width=0,
            marker_opacity=0.95, customdata=cdata, hovertemplate=htmpl,
            name="선택 동",
        ))

    # 후보를 차고지보다 먼저 그린다 — 겹칠 때 현행 차고지가 위로 와야
    # "이미 있는 곳"과 "놓을 수 있는 곳"이 구별된다.
    if show_cands:
        cand = load_candidates()
        # customdata 첫 칸의 태그로 클릭 이벤트에서 후보 점을 구분한다.
        # 코로플레스 점은 location 을 갖고 이 태그가 없다.
        # 마지막 칸이 cand_id 다 — 클릭하면 이 값으로 3페이지를 연다.
        # 칸을 늘리거나 줄이면 아래 hovertemplate 의 번호와 클릭 처리기의
        # cd[CAND_ID_IDX] 가 함께 밀린다.
        cdata = [[CAND_TAG, n, int(c), s, g, d, int(i)]
                 for n, c, s, g, d, i in zip(
                     cand["name"], cand["capacity"], cand["source"], cand["gu"],
                     cand["assigned_dongs"], cand["cand_id"])]
        fig.add_trace(go.Scattermap(
            lat=cand["lat"], lon=cand["lon"], mode="markers",
            marker=dict(size=7, color=CAND_COLOR),
            customdata=cdata,
            hovertemplate=("%{customdata[1]}<br>%{customdata[2]:,}면 · "
                           "%{customdata[3]} · %{customdata[4]}"
                           "<br>배정: %{customdata[5]}<extra></extra>"),
            name="후보 주차장",
        ))

    if show_depots:
        dep = load_depots()
        fig.add_trace(go.Scattermap(
            lat=dep["lat"], lon=dep["lon"], mode="markers",
            marker=dict(size=9, color=DEPOT_COLOR),
            hovertext=[f"{n} · {c}대" for n, c in zip(dep["name"], dep["capacity"])],
            hovertemplate="%{hovertext}<extra></extra>", name="차고지",
        ))

    # **선택으로 인한 흐려짐을 끈다.** `clickmode="event+select"` 라 어느 점을
    # 클릭하면 Plotly 가 나머지 트레이스를 '비선택'으로 보고 흐리게 만든다 —
    # 차고지 점을 눌렀을 때 동들이 통째로 회색조가 되던 것이 이것이다.
    # 2페이지의 회색조는 우리가 트레이스를 따로 그려서 내는 것이므로 Plotly 의
    # 흐려짐은 순전히 군더더기다. 트레이스마다 비선택 투명도를 제 값으로 못 박아
    # 클릭해도 겉모습이 변하지 않게 한다.
    #
    # 차고지는 여기에 더해 **클릭해도 아무 일도 하지 않는다** — 아래 클릭
    # 처리기가 태그도 location 도 없는 점을 그냥 흘린다. Plotly 에는 트레이스
    # 단위로 클릭을 끄는 스위치가 없고(`hoverinfo="skip"` 은 hover 까지 죽인다),
    # 차고지 hover 는 살려 둬야 해서 이렇게 한다.
    for tr in fig.data:
        op = tr.marker.opacity
        tr.unselected = dict(marker=dict(opacity=1.0 if op is None else op))

    fig.update_layout(
        map=dict(style=MAP_STYLE, center=MAP_CENTER, zoom=MAP_ZOOM),
        margin=dict(l=0, r=0, t=0, b=0),
        height=720,
        uirevision="constant",           # rerun 시 줌·팬 유지 — 가장 크게 기여
        transition=dict(duration=400, easing="cubic-in-out"),
        paper_bgcolor=BG,
        showlegend=False,
        font=dict(color=INK),
        clickmode="event+select",
    )
    return fig


def improve_color(rate: float, lo: float, hi: float) -> str:
    """개선율 → 초록 농도. lo·hi 는 **244곳 전체** 범위다(후보 간 비교용)."""
    t = 0.0 if hi <= lo else (abs(rate) - lo) / (hi - lo)
    t = min(max(t, 0.0), 1.0)

    def mix(a: str, b: str) -> str:
        ca = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
        cb = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
        return "#" + "".join(f"{round(x + (y - x) * t):02X}" for x, y in zip(ca, cb))

    return mix(IMPROVE_LIGHT, IMPROVE_DARK)


def build_sim_map(cand_id: int, row) -> go.Figure:
    """한 후보의 3km 영향권 지도.

    **색은 후보 하나 안에서 평평하다 — 그게 데이터의 한계다.** `placement_eval.csv`
    의 before/after 는 3km 범위 **전체를 콜 가중 평균한 값 하나**이고
    (`evaluate.scope_pickup`), 동별 before/after 는 산출돼 있지 않다. 그래서
    영향권 동들을 서로 다른 농도로 칠하면 **없는 차이를 지어내는 것**이 된다.
    한 색으로 칠하고 그 값을 범례에 적는다 — 지도가 답하는 것은 "얼마나
    줄어드는가"가 아니라 **"어디까지 닿는가"** 다.

    농도는 **후보끼리** 비교하라고 둔다. 244곳의 |개선율| 범위 위에서 잡으므로
    다른 후보로 넘어가면 진하기가 달라진다.

    담당 동은 테두리로 가른다. 영향권(3km 안)과 담당(거점으로 배정된 동)은
    다른 개념이고, 한 후보가 최대 5개 동을 맡는다.

    `uirevision` 은 **후보마다 다르다.** "constant" 로 두면 후보를 바꿔도
    카메라가 그대로라 새 후보가 화면 밖에 있을 수 있다. 같은 후보 안의
    rerun 에서는 값이 같아 줌·팬이 유지된다.
    """
    gj, _ = load_boundary()
    poly = resolve_polygons()
    sc = placement_scope().get(cand_id, {})
    scope_set = set(sc.get("polys", []))
    assigned_set = set(sc.get("assigned", []))
    lat, lon = sc.get("lat"), sc.get("lon")

    g = load_placement()
    lo, hi = g["rate_pct"].abs().min(), g["rate_pct"].abs().max()
    fill = improve_color(float(row["rate_pct"]), lo, hi)

    locs = poly["adm_cd2"].tolist()
    names = poly["adm_nm"].tolist()
    fig = go.Figure()

    # ① 바닥 — 영향권 밖. 빼지 않는다. 구멍이 뚫리면 서울이 아닌 것으로 읽힌다.
    fig.add_trace(go.Choroplethmap(
        geojson=gj, locations=locs, z=[0] * len(locs),
        colorscale=[[0, OUT_SCOPE], [1, OUT_SCOPE]], showscale=False,
        marker_line_width=0.4, marker_line_color="#CFCCC6",
        hovertext=names, hovertemplate="%{hovertext}<br>영향권 밖<extra></extra>",
        name="영향권 밖",
    ))

    # ② 영향권 — 한 색. 값은 범례에 적는다.
    rate = float(row["rate_pct"])
    before, after = float(row["before"]), float(row["after"])
    z_scope = [0 if loc in scope_set else None for loc in locs]
    fig.add_trace(go.Choroplethmap(
        geojson=gj, locations=locs, z=z_scope, zmin=0, zmax=1,
        colorscale=[[0, fill], [1, fill]], showscale=False,
        marker_line_width=0.5, marker_line_color="#FFFFFF",
        marker_opacity=0.92, hovertext=names,
        hovertemplate=(f"%{{hovertext}}<br>3km 영향권 · 픽업 {before:.2f} → "
                       f"{after:.2f}분 ({rate:+.2f}%)<extra></extra>"),
        name="3km 영향권",
    ))

    # ③ 담당 동 — 테두리만 강하게. 채움은 ②와 같아야 "영향권 안"이 유지된다.
    if assigned_set:
        z_as = [0 if loc in assigned_set else None for loc in locs]
        fig.add_trace(go.Choroplethmap(
            geojson=gj, locations=locs, z=z_as, zmin=0, zmax=1,
            colorscale=[[0, fill], [1, fill]], showscale=False,
            marker_line_width=2.0, marker_line_color=ASSIGNED_LINE,
            marker_opacity=1.0, hovertext=names,
            hovertemplate="%{hovertext}<br><b>담당 동</b><extra></extra>",
            name="담당 동",
        ))

    # ④ 후보 위치
    if lat is not None:
        fig.add_trace(go.Scattermap(
            lat=[lat], lon=[lon], mode="markers",
            marker=dict(size=13, color=CAND_COLOR),
            hovertext=[f"{row['cand_name']} · {row['gu']}"],
            hovertemplate="%{hovertext}<extra></extra>", name="후보",
        ))

    center = ({"lat": lat, "lon": lon} if lat is not None else MAP_CENTER)
    fig.update_layout(
        map=dict(style=MAP_STYLE, center=center, zoom=11.4),
        margin=dict(l=0, r=0, t=0, b=0), height=720,
        uirevision=f"sim-{cand_id}",     # 후보가 바뀌면 카메라를 다시 잡는다
        transition=dict(duration=400, easing="cubic-in-out"),
        paper_bgcolor=BG, showlegend=False, font=dict(color=INK),
    )
    return fig


# ─────────────────────────────────────────────────────────────
# 패널 조각
# ─────────────────────────────────────────────────────────────

def fmt(v, unit: str = "", digits: int = 1) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"{v:,.{digits}f}{unit}"


def headline(label: str, value: str, gu: str = None, seoul: str = None) -> str:
    cmp_html = (f'<div class="cmp">구 {gu} · 서울 {seoul}</div>'
                if gu is not None else "")
    return (f'<div class="head-metric"><div class="hlabel">{label}</div>'
            f'<div class="hvalue">{value}</div>{cmp_html}</div>')


def rows_html(items: list) -> str:
    out = []
    for label, value in items:
        out.append(f'<div class="metric"><div class="mlabel">{label}</div>'
                   f'<div class="mvalue">{value}</div></div>')
    return "".join(out)


def note_html(text: str) -> str:
    return f'<div class="note">{text}</div>' if text else ""


def role_html(text: str) -> str:
    """지표 역할 표기 — **본문 색.** 회색 주석과 성격이 다르다(CSS `.role` 참조)."""
    return f'<div class="role">{text}</div>' if text else ""


def cand_spec_rows(cand_id: int) -> None:
    """후보 제원 — 면수·소스·자치구. 3페이지 「후보 제원」에 접어 둔다.

    예전에는 지도에서 후보를 클릭했을 때 오른쪽에 뜨는 패널이었다. 후보 클릭이
    3페이지로 가게 되면서 이리로 옮겼다 — **사이드바 목록으로 들어와도 같은
    것을 봐야 하므로** 클릭 customdata 가 아니라 cand_id 로 다시 찾는다.
    """
    cand = load_candidates()
    hit = cand[cand["cand_id"] == cand_id]
    if not len(hit):
        st.markdown(note_html("후보 제원을 찾을 수 없습니다."), unsafe_allow_html=True)
        return
    r = hit.iloc[0]
    st.markdown(rows_html([
        ("면수", f"{int(r['capacity']):,}면"),
        ("소스", r["source"]),
        ("자치구", r["gu"]),
    ]), unsafe_allow_html=True)


def dong_panel(row: pd.Series, df: pd.DataFrame) -> None:
    """선택 동 세부지표.

    핵심 5개(구·서울 비교가 붙는 것)만 항상 보이고 나머지는 접는다.
    """
    members = row["members"]
    sub = df.loc[members]
    w = sub["n_served"].clip(lower=0)
    tot = float(w.sum())

    def wavg(col):
        if col not in sub.columns or tot == 0:
            return float("nan")
        return (sub[col] * w).sum() / tot

    def has_cmp(col):
        g = f"{col}_gu_mean"
        return g in sub.columns and pd.notna(sub.iloc[0][g])

    def pct(col):
        v = wavg(col)
        return v * 100 if pd.notna(v) else v

    st.markdown(f'<div class="dong-name">{row["adm_nm"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="dong-sub">승차 완료 {int(tot):,}건 · {BASIS}</div>',
                unsafe_allow_html=True)

    # 값이 흔들리는 동은 숨기지 않고 사유를 문장으로 붙인다(명세 6절)
    warns = []
    if not row["is_reliable"]:
        warns.append(f"표본 {int(tot):,}건으로 값이 불안정할 수 있습니다.")
    if row["match_type"] == "approx":
        warns.append("경계 개편으로 근사 매칭된 동입니다 "
                     f"(지표: {' · '.join(row['member_names'])}).")
    elif row["match_type"] in ("manual", "exact+manual") and len(members) > 1:
        warns.append(f"개편 전 {' · '.join(row['member_names'])} 를 합산한 값입니다.")
    for msg in warns:
        st.markdown(f'<div class="warn">{msg}</div>', unsafe_allow_html=True)

    # ── 핵심 5개 — 항상 보인다
    cards = []
    cards.append(headline(
        "총 대기 평균", fmt(wavg("wait_total_mean"), "분"),
        *((fmt(wavg("wait_total_mean_gu_mean"), "분"),
           fmt(wavg("wait_total_mean_seoul_mean"), "분"))
          if has_cmp("wait_total_mean") else (None, None))))
    cards.append(headline(
        "60분 초과 비율", fmt(pct("long_wait_ratio"), "%"),
        *((fmt(pct("long_wait_ratio_gu_mean"), "%"),
           fmt(pct("long_wait_ratio_seoul_mean"), "%"))
          if has_cmp("long_wait_ratio") else (None, None))))
    cards.append(headline(
        "기다리다 포기", fmt(pct("cancel_abandoned_ratio"), "%"),
        *((fmt(pct("cancel_abandoned_ratio_gu_mean"), "%"),
           fmt(pct("cancel_abandoned_ratio_seoul_mean"), "%"))
          if has_cmp("cancel_abandoned_ratio") else (None, None))))
    cards.append(headline(
        "3km 내 차량", fmt(wavg("vehicles_3km"), "대", 0),
        *((fmt(wavg("vehicles_3km_gu_mean"), "대", 0),
           fmt(wavg("vehicles_3km_seoul_mean"), "대", 0))
          if has_cmp("vehicles_3km") else (None, None))))

    ur = wavg("usage_rate")
    cards.append(headline(
        "이용률", fmt(ur, "", 2),
        *((fmt(wavg("usage_rate_gu_mean"), "", 2),
           fmt(wavg("usage_rate_seoul_mean"), "", 2))
          if has_cmp("usage_rate") else (None, None))))
    st.markdown(f'<div class="cards">{"".join(cards)}</div>', unsafe_allow_html=True)

    # 이용률 사유는 카드 바로 아래 한 줄로(명세 6절)
    if pd.isna(ur):
        st.markdown(note_html("이용률 — 등록 장애인 통계에 이 동의 값이 '-' 로 들어와 "
                              "낼 수 없습니다."), unsafe_allow_html=True)
    elif bool(sub.iloc[0].get("usage_is_grouped", False)):
        st.markdown(note_html("이용률 — 인접 동과 합산된 그룹 값입니다."),
                    unsafe_allow_html=True)

    # ── 나머지는 접는다
    with st.expander("대기시간 상세"):
        cells = []
        for rlab, pre in [("총", "wait_total"), ("매칭", "wait_match"),
                          ("픽업", "wait_pickup")]:
            cells.append(
                f'<tr><th>{rlab}</th>'
                f'<td>{fmt(wavg(pre + "_mean"), "")}</td>'
                f'<td>{fmt(wavg(pre + "_p50"), "")}</td>'
                f'<td>{fmt(wavg(pre + "_p90"), "")}</td></tr>')
        st.markdown(
            '<table class="grid"><thead><tr><th></th><th>평균</th><th>중앙</th>'
            f'<th>p90</th></tr></thead><tbody>{"".join(cells)}</tbody></table>'
            + note_html("단위 분. 총 = 승차−접수, 매칭 = 배차−접수, 픽업 = 승차−배차"),
            unsafe_allow_html=True)

    with st.expander("취소·완료 상세"):
        n_calls = float(sub["n_calls"].sum())
        done = (tot / n_calls * 100) if n_calls else float("nan")
        st.markdown(rows_html([
            ("완료율", fmt(done, "%")),
            ("즉시 취소", fmt(pct("cancel_immediate_ratio"), "%")),
            ("기다리다 포기", fmt(pct("cancel_abandoned_ratio"), "%")),
            ("배차 후 미승차", fmt(pct("cancel_after_assign_ratio"), "%")),
            ("판별 불가", fmt(pct("cancel_other_ratio"), "%")),
        ]) + note_html("기다리다 포기 = 배차 전 1분 초과 취소"),
            unsafe_allow_html=True)

    with st.expander("콜 발생"):
        dm = load_demand()
        keys = set(zip(sub["gu"], sub["dong_canon"]))
        d = dm[[k in keys for k in zip(dm["gu"], dm["dong_canon"])]]
        n_calls = float(sub["n_calls"].sum())
        # **값마다 단위를 달고 평일·주말은 칸 안에 이름을 적는다.** 예전에는
        # 표 아래에 "단위 건. 평일/주말은 각각 앞·뒤 두 칸"이라 적어 두었는데,
        # 표를 읽으려고 주석을 먼저 읽어야 하는 표는 잘못 짠 표다.
        # 헤더(심야·아침·낮·저녁)는 시간대 행에만 걸리므로 다른 행은 스스로 말해야 한다.
        cells = [f'<tr><th>전체</th><td colspan="4">{fmt(n_calls, "건", 0)}</td></tr>']
        if len(d):
            by_p = d.groupby("period", observed=True)["n_calls"].sum()
            pv = "".join(
                f"<td>{fmt(by_p.get(p, 0), '건', 0)}</td>" for p in PERIOD_ORDER)
            cells.append(f'<tr><th>시간대</th>{pv}</tr>')
            by_w = d.groupby("is_weekend")["n_calls"].sum()
            cells.append(
                f'<tr><th>평일/주말</th>'
                f'<td colspan="2">평일 {fmt(by_w.get(False, 0), "건", 0)}</td>'
                f'<td colspan="2">주말 {fmt(by_w.get(True, 0), "건", 0)}</td></tr>')
        st.markdown(
            '<table class="grid"><thead><tr><th></th><th>심야</th><th>아침</th>'
            f'<th>낮</th><th>저녁</th></tr></thead><tbody>{"".join(cells)}</tbody>'
            '</table>', unsafe_allow_html=True)

    with st.expander("소요시간·속도·거리"):
        st.markdown(rows_html([
            ("차내 소요시간", fmt(wavg("ride_min_mean"), "분")),
            ("평균 속도", fmt(wavg("ride_kmh_mean"), "km/h")),
            ("평균 거리", fmt(wavg("ride_km_mean"), "km", 2)),
        ]), unsafe_allow_html=True)

    with st.expander("등록 장애인·근접도"):
        n_dis = sub["n_disabled"].sum() if sub["n_disabled"].notna().any() else float("nan")
        st.markdown(rows_html([
            ("등록 장애인", fmt(n_dis, "명", 0)),
            ("3km 내 차고지", fmt(wavg("n_depots_3km"), "곳", 0)),
        ]), unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# CSS — 기본 테마를 쓰지 않는다(명세 8절)
# ─────────────────────────────────────────────────────────────

CSS = f"""
<style>
  .stApp {{ background: {BG}; color: {INK}; }}
  header[data-testid="stHeader"] {{ background: transparent; }}
  .block-container {{ padding-top: 1.4rem; padding-bottom: 1rem; max-width: 100%; }}
  section[data-testid="stSidebar"] {{ background: {PANEL_BG};
                                      border-right: 1px solid #E4E1DB; }}

  /* 제목 위계 — 앱 제목 / 지도 제목 */
  .app-title {{ font-size: 1.5rem; font-weight: 680; letter-spacing: -0.025em;
                color: {INK}; line-height: 1.15; }}
  .app-sub {{ font-size: .76rem; color: #8A867D; margin-bottom: .9rem; }}
  .map-title {{ font-size: .95rem; font-weight: 620; color: {INK};
                margin-bottom: .05rem; }}
  .map-sub {{ font-size: .74rem; color: #8A867D; margin-bottom: .5rem;
              font-variant-numeric: tabular-nums; }}

  /* 상단 요약 — st.metric 을 조용하게 */
  div[data-testid="stMetric"] {{ background: {PANEL_BG}; border: 1px solid #E4E1DB;
                                 border-radius: 8px; padding: .55rem .75rem; }}
  div[data-testid="stMetricLabel"] p {{ font-size: .72rem !important;
                                        color: #8A867D !important; }}
  div[data-testid="stMetricValue"] {{ font-size: 1.35rem !important;
                                      font-variant-numeric: tabular-nums;
                                      color: {INK} !important; }}

  /* 패널 — 좁고 스크롤, 등장은 부드럽게 */
  .panel {{
    background: {PANEL_BG}; border: 1px solid #E4E1DB; border-radius: 10px;
    padding: .9rem 1rem .5rem;
    animation: rise .32s cubic-bezier(.2,.7,.3,1);
  }}
  @keyframes rise {{
    from {{ opacity: 0; transform: translateY(8px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
  }}

  .dong-name {{ font-size: 1.3rem; font-weight: 650; letter-spacing: -0.02em; }}
  .dong-sub  {{ font-size: .74rem; color: #7A766E; margin-bottom: .7rem;
                font-variant-numeric: tabular-nums; }}

  .warn {{ font-size: .74rem; line-height: 1.45; color: #7A4A20;
           background: #FBF2E6; border-left: 2px solid #C9975B;
           padding: .45rem .6rem; border-radius: 3px; margin-bottom: .5rem; }}

  /* 전제 — 경고가 아니다. `.warn` 은 "이 후보를 조심하라"인데 이건 어느 후보에나
     같은 조건이라 색을 가른다. 중성 잉크 톤으로 두어 결과 카드와 지도 사이에
     조용히 끼어들되, 왼쪽 굵은 선으로 "읽고 넘어가라"는 무게는 준다. */
  .premise {{ font-size: .78rem; line-height: 1.6; color: #55524C;
              background: #F4F2ED; border-left: 3px solid #8A867D;
              padding: .55rem .75rem; border-radius: 3px;
              margin: .1rem 0 .8rem; }}
  .premise b {{ color: {INK}; font-weight: 620; }}

  /* 참고 지표 — 결과 카드와 전제 사이 한 줄. 카드로 만들지 않는다: st.metric
     한 장 더가 되면 판정 지표와 같은 무게로 읽히는데, 이 값은 순위 기준이
     아니다(순위는 픽업 그대로). 카드보다 낮고 `.note` 보다는 높은 층에 둔다. */
  .aux {{ font-size: .78rem; line-height: 1.55; color: #55524C;
          margin: .35rem 0 .1rem; }}
  .aux b {{ color: {INK}; font-weight: 620;
            font-variant-numeric: tabular-nums; }}
  .aux .dim {{ color: #949087; font-variant-numeric: tabular-nums; }}
  /* 「참고」 — 값 앞에 붙어 성격을 먼저 말한다. 값을 읽고 나서 만나면 늦다. */
  .tag {{ display: inline-block; font-size: .64rem; font-weight: 600;
          color: #7A766E; background: #EDEBE6; border: 1px solid #E0DCD5;
          border-radius: 3px; padding: .04rem .28rem; margin-right: .35rem;
          vertical-align: .06em; letter-spacing: .02em; }}

  /* 픽업 · 총 대기 대비 — 두 줄을 같은 격자에 세운다. 화살표와 괄호가 세로로
     맞아야 "같은 것을 두 자로 잰 값"으로 읽힌다. 줄이 어긋나면 별개 지표 두 개로
     보이고, 그러면 이 블록의 목적(왜 픽업으로 재는가)이 전달되지 않는다.

     **우측 패널(좁은 폭)에 선다**(08.12). 값 칸을 3개로 줄이고 σ·실측 줄은
     격자 전체를 쓰게 해서, 폭이 모자라면 **아래로** 흐르되 두 값 줄은 끝까지
     같은 칸에 서게 했다. 세 칸은 `nowrap` 이라 값 중간에서 꺾이지 않는다 —
     "18.67 → 16.62분" 이 반으로 갈리면 정렬이 무너진다.

     정렬이 버티는 근거는 `tabular-nums` 다. 두 줄 모두 `NN.NN → NN.NN분` 로
     자릿수가 같아 폭이 같아진다. 개선율은 자릿수가 달라(−8.57 / −10.99) 별도
     칸으로 뽑아야 왼쪽 끝이 맞는다. */
  /* 이 블록이 우측 패널의 머리다 — 「담당 동」의 동 이름·콜 수(`.mvalue` .95rem)
     보다 작으면 딸려 온 글로 읽힌다. 그래서 **라벨을 그 크기(.95rem)에 맞추고
     값은 한 단계 위(1.1rem)** 로 올린다. 값이 라벨보다 큰 것이 위계다.
     개선율은 값과 같은 크기로 두되 굵기를 낮춰 before→after 가 먼저 눈에 들게
     했다. 아래 두 줄(실측 대비 · 인용 경고)은 근거라 본문 크기 그대로다.

     칸은 셋이다. 「참고」 배지를 떼면서 넷째 칸이 사라졌고, 그만큼 값 칸에
     자리가 돌아와 글자를 키울 수 있었다. */
  .wcmp {{ font-size: .95rem; line-height: 1.5; color: #55524C;
           margin: 0 0 .15rem; display: grid;
           grid-template-columns: auto auto auto;
           gap: .1rem .45rem; align-items: baseline;
           justify-content: start; }}
  .wcmp .wlab {{ color: #6E6A62; white-space: nowrap; }}
  .wcmp .wval {{ color: {INK}; font-weight: 640; white-space: nowrap;
                 font-size: 1.1rem; font-variant-numeric: tabular-nums; }}
  .wcmp .wrate {{ font-size: 1.1rem; font-weight: 500; white-space: nowrap;
                  font-variant-numeric: tabular-nums; }}
  /* σ·물음표 — 값 줄에 붙이면 좁은 폭에서 값을 밀어낸다. 아래로 내린다. */
  .wcmp .wmeta {{ grid-column: 1 / -1; color: #949087; font-size: .74rem;
                  font-variant-numeric: tabular-nums; }}
  /* 실측 대비 줄 — 격자 전체를 쓰고 한 칸 들여쓴다. 위 두 줄의 근거이지
     별개 항목이 아니라는 것을 들여쓰기로 말한다. 좁은 폭에서는 접히므로
     들여쓴 만큼 매달아(hanging indent) 이어지는 줄도 왼쪽이 맞게 둔다. */
  .wcmp .wfoot {{ grid-column: 1 / -1; color: #7A766E; font-size: .78rem;
                  line-height: 1.5; margin-top: .1rem;
                  padding-left: .85rem; text-indent: -.85rem; }}
  .wcmp .wfoot b {{ color: #55524C; font-weight: 620;
                    font-variant-numeric: tabular-nums; }}
  /* 창이 좁으면 패널도 좁아진다. 값 칸이 `nowrap` 이라 자리가 모자라면 넘치므로
     **줄을 깨는 대신 글자를 줄인다** — 두 줄이 같은 칸에 서는 것이 이 블록의
     전부라 그것만은 어느 폭에서도 지킨다. 위계(값 > 라벨)는 줄어든 뒤에도
     유지한다. */
  @media (max-width: 1150px) {{
    .wcmp {{ font-size: .85rem; column-gap: .32rem; }}
    .wcmp .wval, .wcmp .wrate {{ font-size: .98rem; }}
    .wcmp .wfoot {{ font-size: .74rem; }}
  }}

  /* 물음표 — st.metric 의 `help` 아이콘과 같은 성격이다. 별도 요소를 만들지
     않으려고 `title` 툴팁을 쓴다(전제 블록을 한 번의 markdown 으로 내야 한다). */
  .qmark {{ display: inline-flex; align-items: center; justify-content: center;
            width: 13px; height: 13px; margin-left: .3rem;
            border: 1px solid #B4AFA6; border-radius: 50%;
            font-size: .58rem; color: #8A867D; cursor: help;
            vertical-align: text-top; }}
  .qmark:hover {{ border-color: {INK}; color: {INK}; }}

  /* 핵심 지표 카드 — 숫자가 주인공 */
  .cards {{ display: grid; grid-template-columns: 1fr 1fr; gap: .45rem;
            margin-bottom: .7rem; }}
  .head-metric {{ background: #FFFFFF; border: 1px solid #EAE7E1;
                  border-radius: 8px; padding: .5rem .6rem; }}
  .head-metric:first-child {{ grid-column: 1 / -1; }}
  .hlabel {{ font-size: .7rem; color: #8A867D; margin-bottom: .1rem; }}
  .hvalue {{ font-size: 1.25rem; font-weight: 650; color: {INK};
             font-variant-numeric: tabular-nums; line-height: 1.1; }}
  .cmp {{ font-size: .65rem; color: #9A968D; font-variant-numeric: tabular-nums;
          margin-top: .12rem; }}

  /* 접힌 블록 안의 한 줄 지표 */
  .metric {{ display: flex; align-items: baseline; justify-content: space-between;
             gap: .6rem; padding: .16rem 0; }}
  .mlabel {{ font-size: .78rem; color: #6E6A62; }}
  .mvalue {{ font-size: .95rem; font-weight: 600; color: {INK};
             font-variant-numeric: tabular-nums; }}

  /* 3×3 표 — 세로 9줄을 3줄로 */
  table.grid {{ width: 100%; border-collapse: collapse;
                font-variant-numeric: tabular-nums; }}
  table.grid th {{ font-size: .68rem; font-weight: 600; color: #8A867D;
                   text-align: right; padding: .18rem .3rem; }}
  table.grid thead th {{ border-bottom: 1px solid #E8E5DF; }}
  table.grid tbody th {{ text-align: left; color: #6E6A62; }}
  table.grid td {{ font-size: .92rem; font-weight: 600; text-align: right;
                   padding: .18rem .3rem; color: {INK}; }}
  /* 참고 열 — 같은 표 안에서 판정 지표와 무게를 가른다(색만 낮춘다) */
  table.grid th.aux-td, table.grid td.aux-td {{ color: #948F86; }}

  /* 사이드바 지표별 기준선 표 */
  table.legend-tbl {{ margin-top: .55rem; }}
  table.legend-tbl th {{ font-size: .64rem; }}
  table.legend-tbl tbody th {{ font-weight: 500; }}
  table.legend-tbl td {{ font-size: .74rem; font-weight: 600; }}
  table.legend-tbl td.dim {{ color: #A5A199; font-weight: 500; }}
  table.legend-tbl tr.cur th, table.legend-tbl tr.cur td {{ color: {INK}; }}
  table.legend-tbl tr.cur {{ background: #F0EEE9; }}

  .note {{ font-size: .69rem; color: #949087; line-height: 1.45; margin-top: .4rem; }}

  /* 지표 역할 — **본문 색이다.** 「개선율은 이것, 총절감은 저것」을 `.note` 회색으로
     두면 보조 주석처럼 읽혀 눈이 지나친다. 그런데 **두 지표를 함께 봐야 한다**는 것이
     이 패널의 요점이라, 역할 표기가 먼저 눈에 들어와야 그 요점이 성립한다.
     뒤따르는 백분위·부연은 `.note` 회색 그대로 둔다 — 전부 짙게 하면 아무것도
     도드라지지 않는다. */
  .role {{ font-size: .76rem; color: {INK}; line-height: 1.55; margin-top: .55rem; }}
  .role b {{ font-weight: 620; }}

  .legend {{ font-size: .72rem; color: #7A766E; line-height: 1.9;
             font-variant-numeric: tabular-nums; }}
  .swatch {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px;
             margin-right: .35rem; vertical-align: middle; }}

  /* 버튼 — 화면 톤(저채도 중성)에 맞춘다. 위계는 둘뿐이다:
     주 동작(결과 보기)은 채워서 앞으로, 보조(돌아가기·해제)는 테두리만.
     화살표는 라벨 안에 두고 두 칸 띄워 글자와 붙지 않게 했다. */
  .stButton button, .stButton button:focus:not(:active) {{
    width: 100%; font-size: .78rem; font-weight: 500;
    border-radius: 7px; padding: .34rem .7rem;
    letter-spacing: -0.01em; box-shadow: none;
    transition: background .16s ease, border-color .16s ease,
                color .16s ease, transform .08s ease;
  }}
  /* 보조 — 바탕에 얹힌 테두리. 평소엔 물러나 있다가 hover 에서만 또렷해진다 */
  .stButton button[kind="secondary"] {{
    background: {PANEL_BG}; border: 1px solid #DFDBD4; color: #6E6A62;
  }}
  .stButton button[kind="secondary"]:hover {{
    background: #FFFFFF; border-color: #C4BFB6; color: {INK};
  }}
  /* 주 — 잉크색으로 채운다. 저채도 화면에서 유일하게 짙은 면이라 눈에 먼저 든다 */
  .stButton button[kind="primary"] {{
    background: {INK}; border: 1px solid {INK}; color: {PANEL_BG};
  }}
  .stButton button[kind="primary"]:hover {{
    background: #3A3733; border-color: #3A3733; color: #FFFFFF;
  }}
  /* 눌리는 순간만 1px 내려앉는다 — 소리 없는 확인 */
  .stButton button:active {{ transform: translateY(1px); }}
  .stButton button:focus-visible {{ outline: 2px solid #9A968D;
                                    outline-offset: 2px; }}

  details summary {{ font-size: .8rem !important; color: #6E6A62 !important; }}
</style>
"""


# ─────────────────────────────────────────────────────────────
# 사이드바 — 조작·범례·기준 표기를 본문에서 걷어낸다
# ─────────────────────────────────────────────────────────────

def candidate_picker(sb) -> None:
    """사이드바 「후보 목록」 — 244곳을 정렬해 훑고 고른다.

    **지도에서 찾아 들어가는 경로만으로는 전체를 훑을 수 없다.** "개선 큰 곳부터
    보고 싶다"에 답할 자리가 여기다. 지도 클릭은 위치를 알 때, 목록은 모를 때 쓴다.

    정렬 축을 둘 다 두는 이유 — 개선율과 총절감은 순위 상관 +0.32 로 다른 것을
    잰다(상위 20 중 4곳만 겹친다). 하나만 두면 그 축이 정답인 것처럼 읽힌다.
    """
    g = load_placement()
    sb.markdown('<div class="map-title">후보 목록</div>', unsafe_allow_html=True)
    if g.empty:
        sb.markdown(note_html(
            "시뮬 결과 파일이 없습니다 — <code>python src/evaluate.py</code> 를 "
            "돌리면 목록이 채워집니다."), unsafe_allow_html=True)
        return

    sb.radio("정렬", ["rate", "total"],
             format_func=lambda k: "개선율 순" if k == "rate" else "총절감 순",
             key="cand_sort", horizontal=True, label_visibility="collapsed")

    opts = candidate_options(st.session_state.cand_sort)
    ids = [c for c, _ in opts]
    label_of = dict(opts)
    cur = st.session_state.get("sim_cand")
    idx = ids.index(cur) if cur in ids else 0

    sb.selectbox("후보", ids, index=idx, key="cand_pick",
                 format_func=lambda c: label_of[c], label_visibility="collapsed")
    if sb.button("이 후보 결과 보기  →", type="primary"):
        st.session_state.sim_cand = st.session_state.cand_pick
        st.session_state.page = 3
        st.rerun()

    # 라벨에 두 숫자가 붙어 있으니 각각이 무엇인지 여기서 한 번 말한다.
    # **역할 두 줄은 본문 색이다** — 회색으로 두면 보조 주석처럼 읽혀 눈이 지나치는데,
    # 두 지표를 함께 봐야 한다는 것이 이 목록의 요점이다. 뒤 두 줄(주의·표본 부족)은
    # 회색 그대로 둔다 — 전부 짙게 하면 아무것도 도드라지지 않는다.
    n_weak = int((~g["sample_ok"]).sum())
    sb.markdown(
        role_html("<b>개선율</b> — 3km 안 대기가 몇 % 줄어드는지<br>"
                  "<b>총절감</b> — 줄어든 시간의 합 (한 달 기준)")
        + note_html(
            "좁은 지역일수록 개선율이 커 보입니다. 둘을 함께 보십시오.<br>"
            f"※표본부족 <b>{n_weak}곳</b> — 3km 콜 {E.MIN_CALLS_SCOPE:,}건 미만이라 "
            f"개선율이 커 보여도 닿는 사람이 적습니다."), unsafe_allow_html=True)


# 사이드바 히스토그램 — 폭이 좁아 **분포 모양과 표식 위치만** 읽히면 된다.
# 축 눈금을 최소로 두고 y축은 통째로 지운다(도수는 세는 값이 아니라 모양이다).
DIST_H = 104
DIST_BINS = 34


def _label_pct(v: float) -> str:
    return f"{v:+.1f}%"


def _label_min(v: float, scale: bool) -> str:
    """총절감(분) 축 라벨. 사이드바 폭이 좁아 네 자리를 그대로 못 쓴다.

    `-2,954` 는 6글자라 축 양 끝에서 잘린다. 천 단위로 줄이면 `-3.0천분` 4~5글자다.
    **양 끝의 단위를 섞지 않는다** — 한쪽만 천 단위면 폭을 눈으로 견줄 수 없다.
    """
    return f"{v / 1000:+.1f}천분" if scale else f"{v:+,.0f}분"


def _dist_fig(values: np.ndarray, cur: float, label) -> go.Figure:
    """분포 하나 + 현재 후보 표식. **순위 번호를 붙이지 않는다.**

    대비 구간이 겹치는 후보끼리는 우열을 가릴 수 없다는 원칙이 여기서도 유지돼야
    한다 — 이 그림이 답하는 것은 "몇 등인가"가 아니라 "분포 어디쯤인가"다.

    **두 지표 모두 음수이고 작을수록 좋다**(개선율 −8.57% · 총절감 −2,954분).
    그래서 축 방향을 따로 뒤집을 일이 없다 — 오름차순이 곧 좋은 쪽 → 나쁜 쪽이고,
    두 그림의 방향이 저절로 맞는다.

    **양 끝 값은 눈금이 아니라 주석으로 찍는다.** 눈금은 값에 **가운데 정렬**돼
    라벨의 절반이 그림 밖으로 나가는데, 좌우 여백이 2px 이라 거기서 잘린다.
    폭이 좁은 라벨(`-8.6`)은 살아남고 넓은 라벨(`-2,954`)은 통째로 사라져 **총절감
    차트만 축이 없었다.** 주석을 안쪽으로 정렬하면(왼쪽 끝은 `left`, 오른쪽 끝은
    `right`) 폭과 무관하게 그림 안에 들어온다.

    **축이 없으면 분포의 폭을 알 수 없다** — 표식이 "어디쯤"만 말하고 "얼마나
    차이 나는지"는 못 말한다. 두 지표를 함께 보는 화면이라 폭이 보여야 한다.
    """
    fig = go.Figure(go.Histogram(
        x=values, nbinsx=DIST_BINS, marker_color="#D8D4CC",
        marker_line_width=0, hoverinfo="skip"))
    fig.add_vline(x=cur, line_color=IMPROVE_DARK, line_width=2)
    fig.add_annotation(x=cur, y=1, yref="paper", text="이 후보", showarrow=False,
                       yanchor="bottom", font=dict(size=9, color=IMPROVE_DARK))

    lo, hi = float(np.min(values)), float(np.max(values))
    for x, anchor in ((lo, "left"), (hi, "right")):
        fig.add_annotation(x=x, y=0, yref="paper", text=label(x), showarrow=False,
                           xanchor=anchor, yanchor="top", yshift=-3,
                           font=dict(size=9, color="#9A968D"))

    fig.update_xaxes(showticklabels=False, ticks="", showgrid=False, zeroline=False)
    fig.update_yaxes(visible=False)
    # 아래 여백은 양 끝 라벨이 앉을 자리다 — 2px 이면 라벨이 잘린다.
    fig.update_layout(height=DIST_H, margin=dict(l=2, r=2, t=14, b=18),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      bargap=0.06, showlegend=False)
    return fig


def _subject(noun: str) -> str:
    """주격 조사를 붙인다 — 받침이 있으면 '이', 없으면 '가'.

    「수혜이」처럼 틀린 조사가 화면에 나가는 것을 막는다. 한글 음절은
    `(코드 − 0xAC00) % 28` 이 0 이 아니면 받침이 있다. 한글이 아니면 '이'로 둔다
    (숫자·영문 지표명이 들어올 일은 없지만 조용히 깨지지는 않게).
    """
    ch = ord(noun[-1]) - 0xAC00
    return noun + ("이" if not 0 <= ch < 11172 or ch % 28 else "가")


def _position_text(values: np.ndarray, cur: float, noun: str) -> str:
    """분포에서 '좋은 쪽으로부터' 어디쯤인가. **등수가 아니다.**

    **양 끝에서는 백분율이 거짓말을 한다.** 개선이 가장 큰 후보는 0% 가 되어
    "아무것도 아닌 건가"로 읽히고, 개선이 가장 작은 후보는 100% 가 되어 **최대치로
    읽힌다 — 뜻이 정반대로 전달된다.** 축이 '좋은 쪽에서부터'라는 것을 모르면
    큰 숫자가 좋아 보이기 때문이다.

    **규칙은 "반올림한 숫자가 거짓말을 하면 말로 바꾼다"** 하나다. 0 이나 100 으로
    떨어지는 구간에서만 「가장 ~ 쪽 끝」으로 적고, 그 사이는 백분율 그대로 낸다.
    문턱을 따로 정하지 않아 표기 정밀도와 어긋날 일이 없다.

    **등수를 말하는 것이 아니다.** 244곳에서 0 으로 떨어지는 것은 한두 곳이라
    그 후보들이 「가장 큰 쪽 끝」을 함께 쓴다 — 끝에 있다는 사실만 말하고 그들
    사이의 순서는 말하지 않는다. 대비 구간이 겹치는 후보끼리 우열을 가릴 수 없다는
    원칙과 같은 방향이다. 히스토그램의 표식이 이미 같은 것을 보여 준다.
    """
    v = np.asarray(values, dtype=float)
    pct = round(float((v < cur).mean()) * 100)
    subj = _subject(noun)
    if pct <= 0:
        return f"{subj} 가장 큰 쪽 끝입니다"
    if pct >= 100:
        return f"{subj} 가장 작은 쪽 끝입니다"
    return f"{subj} 큰 쪽에서 약 <b>{pct}%</b> 지점"


def distribution_panel(sb, row) -> None:
    """개선율·총절감 분포 안에서 이 후보가 어디 있는지 — **후보 목록 아래.**

    **후보 하나만 보이면 "10대 넣었으니 좋아졌겠지"로 읽힌다.** 같은 10대를 넣어도
    위치에 따라 개선율이 몇 배로 갈린다는 사실이 화면에 있어야 이 도구가 재는 것이
    증차 효과가 아니라 **위치 효과**라는 것이 성립한다.

    **두 지표를 나란히 두는 이유** — 총절감만 보면 "콜 많은 지역에 두면 된다"로
    읽힌다. 순위 상관이 +0.32 뿐이라 두 순위는 거의 겹치지 않는다. 어느 하나로
    정렬해도 그 방향으로 오독되므로 역할을 각각 적는다.

    **순위 번호는 붙이지 않는다.** 백분위는 "대략 어디쯤"이지 등수가 아니다.
    """
    g = load_placement()
    if g.empty or len(g) < 2:
        return
    rate = g["rate_pct"].to_numpy(float)
    total = g["total_delta_min"].to_numpy(float)
    cur_r, cur_t = float(row["rate_pct"]), float(row["total_delta_min"])
    rho = float(pd.Series(rate).corr(pd.Series(total), method="spearman"))

    sb.markdown(f'<div class="map-title">{len(g)}곳 분포에서 이 후보</div>',
                unsafe_allow_html=True)

    # 역할 표기는 본문 색, 뒤따르는 백분위는 회색 — 색으로 위계를 가른다.
    sb.markdown(role_html("<b>개선율</b> — 이 지역이 얼마나 좋아지나")
                + note_html(_position_text(rate, cur_r, "개선")),
                unsafe_allow_html=True)
    sb.plotly_chart(_dist_fig(rate, cur_r, _label_pct),
                    width="stretch", config={"displayModeBar": False},
                    key=f"dist-rate-{int(row['cand_id'])}")
    sb.markdown(note_html(
        f"같은 {S.NEW_DEPOT_VEHICLES}대를 넣어도 위치에 따라 "
        f"<b>{rate.min():+.2f}% ~ {rate.max():+.2f}%</b> 로 갈립니다."),
        unsafe_allow_html=True)

    sb.markdown(role_html("<b>총절감</b> — 몇 사람에게 닿나")
                + note_html(_position_text(total, cur_t, "수혜")),
                unsafe_allow_html=True)
    # 천 단위 축약은 값이 네 자리를 넘을 때만 — 세 자리면 그대로가 더 읽기 쉽다
    scale = float(np.abs(total).max()) >= 1000
    sb.plotly_chart(_dist_fig(total, cur_t, lambda v: _label_min(v, scale)),
                    width="stretch", config={"displayModeBar": False},
                    key=f"dist-total-{int(row['cand_id'])}")

    sb.markdown(note_html(
        f"두 순위는 상관 <b>{rho:+.2f}</b> 로 거의 겹치지 않습니다 — 다른 것을 "
        "잽니다. 어느 하나만 보면 그 방향으로 읽히므로 둘을 함께 보십시오."),
        unsafe_allow_html=True)


def sidebar(poly: pd.DataFrame) -> None:
    sb = st.sidebar
    sb.markdown('<div class="map-title">색칠 지표</div>', unsafe_allow_html=True)
    # key= 로 session_state 가 단일 출처. value= 와 대입을 섞으면 한 박자
    # 늦게 반영된다.
    sb.radio("색칠 지표", list(METRICS),
             format_func=lambda k: METRICS[k]["label"],
             key="metric_key", label_visibility="collapsed")

    sb.divider()
    sb.markdown('<div class="map-title">레이어</div>', unsafe_allow_html=True)
    sb.toggle("차고지 (현행 44개)", key="show_depots")
    cand = load_candidates()
    sb.toggle(f"후보 주차장 (배정된 {len(cand)}곳)", key="show_cands")
    sb.markdown(
        f'<div class="legend">'
        f'<span class="swatch" style="background:{DEPOT_COLOR}"></span>현행 차고지<br>'
        f'<span class="swatch" style="background:{CAND_COLOR}"></span>후보 주차장'
        f'</div>', unsafe_allow_html=True)
    sb.divider()
    key = st.session_state.metric_key
    spec = METRICS[key]
    summ = metric_summary()
    fold_v, n_red = summ[key]
    scale = spec.get("scale", 1)
    unit, digits = spec["unit"], spec["digits"]
    side = "미만" if spec["worse"] == "high" else "초과"
    beyond = "이상" if spec["worse"] == "high" else "이하"
    vals = poly[key].dropna() * scale
    worst = vals.max() if spec["worse"] == "high" else vals.min()
    fold_s = fmt(fold_v, unit, digits)

    sb.markdown('<div class="map-title">범례</div>', unsafe_allow_html=True)
    sb.markdown(
        f'<div class="legend">'
        f'<span class="swatch" style="background:{GREEN}"></span>{fold_s} {side}<br>'
        f'<span class="swatch" style="background:{RED_LIGHT}"></span>'
        f'{fold_s} {beyond} — <b>{n_red}개 동</b><br>'
        f'<span class="swatch" style="background:{RED_DARK}"></span>'
        f'{fmt(worst, unit, digits)} (최대)<br>'
        f'<span class="swatch" style="background:{NO_MATCH}"></span>지표 매칭 없음'
        f'</div>', unsafe_allow_html=True)

    # 지표별 기준선·빨강 수 — 지표를 바꾸기 전에 규모를 가늠할 수 있게
    cells = []
    for k, sp in METRICS.items():
        f_v, n = summ[k]
        basis = "절대" if isinstance(sp["fold"], (int, float)) else str(sp["fold"])
        mark = ' class="cur"' if k == key else ""
        cells.append(
            f'<tr{mark}><th>{sp["label"]}</th>'
            f'<td>{fmt(f_v, sp["unit"], sp["digits"])}</td>'
            f'<td class="dim">{basis}</td><td>{n}</td></tr>')
    sb.markdown(
        '<table class="grid legend-tbl"><thead><tr><th></th><th>기준선</th>'
        f'<th>기준</th><th>빨강</th></tr></thead><tbody>{"".join(cells)}</tbody>'
        '</table>', unsafe_allow_html=True)

    # 기준선을 어떻게 잡았는지는 위 표의 「기준」 열(절대/p75/p25)이 이미 말한다.
    # 그 아래 붙던 설명 문단은 걷어냈다 — 근거는 명세 4절.

    sb.divider()
    candidate_picker(sb)

    sb.divider()
    n_poly, n_col = len(poly), int(poly["has_metrics"].sum())
    sb.markdown(note_html(
        f"데이터 기준 <b>{BASIS}</b><br>"
        f"경계 {n_poly}개 중 {n_col}개 표시 · 미매칭 {n_poly - n_col}개"),
        unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# 페이지 — pages/ 분리가 아니라 session_state 조건부 렌더링.
# 선택 상태를 들고 다녀야 한다(명세 7절).
# ─────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    ss = st.session_state
    ss.setdefault("page", 1)
    ss.setdefault("selected", None)
    ss.setdefault("show_depots", True)
    ss.setdefault("show_cands", True)
    ss.setdefault("metric_key", DEFAULT_METRIC)
    ss.setdefault("sim_cand", None)          # 3페이지가 보는 후보 cand_id
    ss.setdefault("cand_sort", "rate")

    df = load_metrics()
    poly = resolve_polygons()
    summ = seoul_summary()

    if ss.page == 3:
        render_sim_page()
        return

    sidebar(poly)

    st.markdown(f'<div class="app-title">{APP_TITLE}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="app-sub">{BASIS} 기준 · 실무 진단용</div>',
                unsafe_allow_html=True)

    # 서울 전체 요약 — 개별 동 값을 읽을 때의 비교 기준
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("서울 평균 대기", f"{summ['mean_wait']:.1f}분")
    m2.metric(f"{RED_THRESHOLD_MIN:.0f}분 이상 동", f"{summ['n_red']}개")
    m3.metric("지역 지니", f"{summ['gini']:.3f}")
    m4.metric("표시 동", f"{summ['n_colored']}개")

    left, right = st.columns([4, 1], gap="medium")

    with left:
        spec = METRICS[ss.metric_key]
        st.markdown(f'<div class="map-title">{MAP_TITLE}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="map-sub">{spec["label"]}으로 색칠 · 동을 클릭하면 '
            f'세부지표가 열립니다</div>', unsafe_allow_html=True)

        fig = build_map(poly, ss.selected, ss.show_depots, ss.show_cands,
                        ss.metric_key)
        ev = st.plotly_chart(fig, width="stretch", on_select="rerun",
                             selection_mode="points", key="map")

        pts = []
        if ev is not None:
            sel = ev.get("selection") if isinstance(ev, dict) else getattr(ev, "selection", None)
            pts = (sel or {}).get("points", []) if sel else []
        for p in pts:
            cd = p.get("customdata")
            # 후보 점 — 동 선택과 섞이면 안 되므로 태그로 먼저 가른다.
            # 후보를 누르면 **3페이지(시뮬 결과)** 로 간다. 예전에는 오른쪽에
            # 작은 제원 패널만 띄웠는데, 그 내용은 3페이지 「후보 제원」으로
            # 옮겼다 — 목록으로 들어와도 같은 것을 볼 수 있어야 한다.
            if (isinstance(cd, (list, tuple)) and len(cd) > CAND_ID_IDX
                    and cd[0] == CAND_TAG):
                cid = int(cd[CAND_ID_IDX])
                if cid != ss.sim_cand or ss.page != 3:
                    ss.sim_cand = cid
                    ss.page = 3
                    st.rerun()
                continue
            # 차고지 점 — 태그도 location 도 없다. 여기서 그냥 흘린다.
            loc = p.get("location")
            if loc and loc != ss.selected:
                ss.selected = loc
                ss.page = 2
                st.rerun()

    with right:
        # 아무 동도 고르지 않았으면 **오른쪽은 비워 둔다.** 안내 문구를 넣으면
        # 빈 사각형이 하나 서 있게 되는데, 지도만 보고 있을 때 그게 제일 거슬린다.
        if ss.selected is not None:
            # 전체보기는 패널 맨 위 — 동 이름 바로 위
            if st.button("←  전체 보기 (선택 해제)", type="secondary"):
                ss.selected = None
                ss.page = 1
                st.rerun()

            # `<div class="panel">` 만 따로 부르면 Streamlit 이 그 호출을 독립
            # 요소로 감싸 빈 사각형을 만든다. dong_panel 은 expander 를 쓰므로
            # 한 문자열로 묶을 수 없어 **패널 테두리를 두르지 않는다.**
            row = poly[poly["adm_cd2"] == ss.selected]
            if len(row) and row.iloc[0]["has_metrics"]:
                dong_panel(row.iloc[0], df)
            else:
                nm = row.iloc[0]["adm_nm"] if len(row) else "?"
                # 왜 비었는지(경계 파일 연도 차이)는 실무자가 할 일이 없는
                # 내부 사정이다 — 사실만 적는다. 내력은 명세 5·9절.
                st.markdown(
                    f'<div class="dong-name">{nm}</div>'
                    '<div class="warn">이 경계에 대응하는 콜 데이터가 없습니다.'
                    '</div>', unsafe_allow_html=True)
            orphans = poly.attrs.get("orphan_rows", [])
            if orphans:
                names = " · ".join(o["dong"] for o in orphans)
                st.markdown(note_html(
                    f"지도에 실리지 않는 지표 {len(orphans)}개: {names}. "
                    "대응 경계가 없어 색칠 대상에서 빠집니다."), unsafe_allow_html=True)


def sim_warnings(row) -> None:
    """이 후보를 읽을 때 걸리는 경고. **절도 제목도 없이 경고만 낸다.**

    「해석 도움」이라는 절을 두었다가 걷어냈다 — 상시 표시할 내용이 빠지고 나니
    제목이 빈 상자를 하나 더 만들 뿐이었다. 경고는 있을 때만 나타나는 편이
    눈에 띈다.

    표본 부족만 후보에 따라 뜨고, 아래 두 줄은 어느 후보에나 붙는다 — 절감폭의
    크기와 기준 기간은 이 화면의 숫자를 인용할 때 반드시 따라가야 하는 말이다.
    """
    if not bool(row["sample_ok"]):
        st.markdown(
            f'<div class="warn">3km 콜 {int(row["n_calls_scope"]):,}건으로 '
            f'수혜 범위가 좁습니다.</div>', unsafe_allow_html=True)
    # **방향을 단정하지 않는다**(08.10 정정). 원래 "절감폭은 실제보다 작게
    # 나옵니다"였는데 한쪽 편의만 센 문장이라 틀렸다 — 근거는 BIAS_NOTE 주석.
    st.markdown(note_html(
        "절감폭의 절대 크기는 양방향 편의를 안고 있습니다. "
        "후보 간 비교로 읽으십시오."), unsafe_allow_html=True)
    st.markdown(note_html(
        f"대표 기간({SIM_START:%Y-%m-%d}~{SIM_END:%m-%d}) 기준입니다. "
        f"진단 화면의 연간 값과 다릅니다."), unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def fleet_effect() -> dict:
    """증차가 실제로 몇 대인지 — **코드에서 계산한다.**

    화면에 574/582/8 을 적어 두면 `NEW_DEPOT_VEHICLES`·`FLEET_WEEKDAY` 나 차고지
    정원이 바뀔 때 조용히 어긋난다. `run_placement` 이 하는 계산을 그대로 돌린다.

    **후보와 무관하다** — 배속은 어느 후보든 10대로 고정이고(A-10),
    `allocate_fleet` 은 정원 벡터만 본다. 그래서 한 번 계산해 캐시한다.
    """
    dep = load.load_depots()
    res = S.resolve_placement({"name": "_", "lat": 37.5, "lon": 127.0}, dep)
    scale = float(res["capacity"].sum()) / float(dep["capacity"].sum())
    out = {"cap_base": int(dep["capacity"].sum()),
           "cap_new": int(res["capacity"].sum()), "scale": scale}
    for tag, base in (("weekday", S.FLEET_WEEKDAY), ("weekend", S.FLEET_WEEKEND)):
        n = int(round(base * scale))
        per, old = S.allocate_fleet(res, n), S.allocate_fleet(dep, base)
        out[tag] = {"base": int(base), "after": n, "new_depot": int(per[-1]),
                    "changed": int((per[:-1] != old).sum())}
    return out


# 물음표 도움말 — 본문에 두면 길어지는 부연을 `title` 로 접는다. 별도 요소를
# 만들지 않아 전제 블록이 한 번의 markdown 으로 나간다(패널 조각내지 말 것).
#
# **주말 단서만 남긴다**(08.10). 「기존 차고지에서 차감하지 않는 이유」 문단을
# 걷어냈다 — 그건 설계 근거라 문서에 있으면 되고(A-10), 화면에서 실무자가 할 일이
# 없다. 남은 문단은 성격이 다르다: 전제 블록이 평일 값을 **그 값인 것처럼** 적으므로
# 이 단서가 없으면 화면이 주말에 대해 틀린 말을 한다.
SCENARIO_HELP = (
    "주말은 {wke_base} → {wke_after}대이고 이 거점이 {wke_new}대입니다. "
    "대표 기간이 평일 중심이라 평일 값을 기본으로 보입니다."
)


def sim_scenario(row) -> None:
    """**이 숫자가 어떤 조건에서 나왔는지.** 후보 카드 바로 아래에 둔다.

    아래쪽 경고들과 같은 층에 두면 전제가 결과 뒤에 오게 되어 늦다 — 실무자가
    개선율을 먼저 읽고 "거점만 옮기면 이렇게 되나"로 굳힌 다음에야 전제를 만난다.

    **두 번째 문장이 이 블록의 핵심이다.** "증차해서 좋아진 것 아니냐"에 답하는
    자리다 — 10대는 모든 후보에 똑같이 들어가므로 후보별 차이는 위치에서만 나온다
    (A-10 · `resolve_placement` 이 넘겨받은 배속을 언제나 10 으로 덮는다).

    **경고가 아니다.** `.warn` 은 "이 후보를 조심하라"는 말인데 이건 어느 후보에나
    같은 조건이라, 색을 달리해 성격을 가른다.
    """
    f = fleet_effect()
    wd, wke = f["weekday"], f["weekend"]
    help_txt = SCENARIO_HELP.format(wke_base=wke["base"], wke_after=wke["after"],
                                    wke_new=wke["new_depot"])
    st.markdown(
        '<div class="premise">'
        f'<b>이 거점에 차량 {S.NEW_DEPOT_VEHICLES}대를 증차했을 때의 결과입니다.</b> '
        f'기존 {len(load.load_depots())}개 차고지 배속은 그대로입니다 '
        f'(평일 {wd["base"]} → {wd["after"]}대, 이 거점이 {wd["new_depot"]}대).'
        f'<span class="qmark" title="{help_txt}">?</span>'
        '<br>'
        f'모든 후보에 같은 {S.NEW_DEPOT_VEHICLES}대를 넣었으므로 '
        '<b>후보별 차이는 위치에서만 나옵니다.</b>'
        '</div>', unsafe_allow_html=True)


def sim_total_wait(row) -> None:
    """픽업 · 총 대기 대비 — **참고.** 우측 패널 맨 위, 지도 옆에 선다.

    **본문에서 패널로 옮겼다**(08.12). 카드 아래 본문에 있을 때는 전체 폭을
    썼지만, 지도가 그 아래로 밀려 결과와 지도를 한눈에 못 봤다. 패널로 오면서
    지도와 나란히 서고 본문은 카드 → 전제 → 지도로 짧아졌다.

    좁은 폭에서 두 줄의 세로 정렬이 무너지면 별개 지표 두 개로 읽혀 대비가
    죽는다. 그래서 값 칸을 셋으로 줄이고 σ·실측 줄을 아래로 내렸다(CSS `.wcmp`).

    **테두리 있는 패널이다** — 「담당 동」과 같은 층이라 같은 모양이어야 패널
    안에서 딸려 온 글로 보이지 않는다. 안에 위젯이 없어(σ 물음표도 `title`
    속성이다) **한 번의 markdown 으로** 낼 수 있다(명세 8절).

    **「참고」 배지는 뗐다**(08.12). 이 블록이 순위 기준이 아니라는 것은 이제
    아래 두 줄(실측 대비 · 인용 경고)과 물음표가 말한다 — 목록 정렬·분포·등급이
    픽업 그대로인 것은 코드에서 그대로다. 배지가 빠지면서 넷째 칸이 사라져
    값 칸에 자리가 돌아왔고, 그만큼 글자를 키울 수 있었다.

    **왜 카드가 아닌가.** `st.metric` 을 한 장 더 두면 판정 지표 셋과 같은 무게로
    읽히는데, 이 값은 순위 기준이 아니다 — 목록 정렬도 분포도 픽업 그대로다.
    성격이 다른 것을 같은 모양으로 두면 화면이 "이것도 기준"이라고 말하게 된다.

    **픽업 before/after 도 이 블록이 담당한다**(08.12). 카드 넷째로 있던 것을
    뺐다 — 같은 숫자가 두 번 뜨면 대비가 약해진다.

    **분 단위를 픽업과 나란히 낸다**(08.12 개정). 예전에는 개선율만 냈다 — 분을
    보이면 진단 화면 실측과 견줘진다는 이유였는데, **그 대비가 이 블록의 목적**이다.
    시뮬 총 대기 before 가 실측의 0.30~0.49배라는 것이 매칭 미재현(A-19)의 증거고,
    그것이 채점 지표를 픽업으로 옮긴 근거다. 두 줄을 세로로 맞춰 세우면 "왜 픽업
    으로 재는가"가 문서가 아니라 화면에서 보인다.

    **실측(`obs_total_min`)이 없으면 분을 내지 않는다.** 분만 있고 실측이 없으면
    옛 걱정이 그대로 살아난다 — 값이 서울 평균 42.7분과 곧장 견줘진다. 두 값은
    짝이라 하나만 내면 안 된다. 실측이 비면 개선율 한 줄로 돌아간다.

    **그런데도 왜 보이는가.** 픽업만 보이면 "그래서 이용자 체감은 얼마나
    줄어드나"에 답이 없다. 총 대기가 이용자가 실제로 겪는 시간이고, 후보 간
    비교라면 이 값도 읽을 수 있다.

    σ 는 시드 3회의 표준편차다. **대비 구간(±1.41σ)을 쓰지 않는다** — 그 폭은
    등급 규칙에서 유도한 것이라 겹침 판정과 짝인데, 총 대기로는 겹침을 계산하지
    않는다. 여기서 말할 수 있는 것은 "시드에 따라 이만큼 흔들린다"뿐이다.
    """
    rate = row.get("rate_total_pct")
    if rate is None or pd.isna(rate):
        return
    sd = row.get("rate_total_sd")
    spread = ("" if sd is None or pd.isna(sd)
              else f'<span class="dim"> · 시드 간 σ {sd:.2f}%p</span>')

    obs = row.get("obs_total_min")
    bt, at = row.get("before_total"), row.get("after_total")
    minutes_ok = all(v is not None and pd.notna(v) for v in (obs, bt, at))

    if not minutes_ok:
        # 실측이나 분이 없으면 옛 형태 — 개선율 한 줄. 한 줄뿐이라 정렬 문제가
        # 없으므로 꼬리표는 원래대로 값 앞이다. 테두리는 같이 두른다 — 패널
        # 안에서 이 블록만 테두리가 없으면 딸려 온 글로 보인다.
        st.markdown(
            '<div class="panel"><div class="aux">'
            f'총 대기 개선율 <b>{rate:+.2f}%</b>{spread}'
            f'<span class="qmark" title="{TOTAL_WAIT_HELP}">?</span>'
            + note_html(TOTAL_WAIT_NOTE)
            + '</div></div>', unsafe_allow_html=True)
        return

    ratio = row.get("sim_obs_ratio")
    ratio_txt = ("." if ratio is None or pd.isna(ratio)
                 else f' — 시뮬은 그 <b>{ratio:.2f}배</b>다.')
    # 제 줄에 서므로 앞의 가운뎃점을 뗀다 — `spread` 는 개선율 한 줄로
    # 돌아갈 때(실측 없음) 값 뒤에 이어 붙는 형태라 그쪽은 점이 필요하다.
    meta = "" if sd is None or pd.isna(sd) else f'시드 간 σ {sd:.2f}%p'
    # **패널은 한 번의 markdown 으로 낸다**(명세 8절). 여는 태그만 따로 부르면
    # Streamlit 이 그 호출을 독립 요소로 감싸 빈 사각형을 만든다. 여기 들어가는
    # 것이 전부 문자열이라(σ 물음표도 위젯이 아니라 `title` 속성이다) 한 덩이로
    # 묶을 수 있다 — 위젯이 끼면 테두리를 두르지 못한다.
    st.markdown(
        '<div class="panel">'
        '<div class="wcmp">'
        # 픽업 — 판정 지표. 먼저 온다.
        '<span class="wlab">픽업</span>'
        f'<span class="wval">{row["before"]:.2f} → {row["after"]:.2f}분</span>'
        f'<span class="wrate">({row["rate_pct"]:+.2f}%)</span>'
        # 총 대기 — 라벨은 이것 하나다. 「참고」 배지를 뗐다(08.12). 성격은
        # 아래 두 줄(실측 대비 · 인용 경고)과 물음표가 말한다. 배지가 빠지면서
        # 넷째 칸이 사라져 값 칸에 자리가 돌아왔다.
        '<span class="wlab">총 대기</span>'
        f'<span class="wval">{bt:.2f} → {at:.2f}분</span>'
        f'<span class="wrate">({rate:+.2f}%)</span>'
        f'<span class="wmeta">{meta}'
        f'<span class="qmark" title="{TOTAL_WAIT_HELP}">?</span></span>'
        # 실측 대비 — 위 두 줄의 근거. 가정 번호(A-19)는 뗐다 — 화면은 실무자가
        # 보는 자리이고, 「매칭을 재현하지 않아 짧다」가 그 자체로 뜻이 통한다.
        # 번호는 문서와 코드에 남아 있으므로 추적은 끊기지 않는다.
        f'<span class="wfoot">└ 같은 범위 실측 <b>{obs:.1f}분</b>{ratio_txt}'
        ' 매칭을 재현하지 않아 짧다</span>'
        '</div>'
        + note_html(TOTAL_WAIT_NOTE)
        + '</div>', unsafe_allow_html=True)


def sim_dong_panel(cand_id: int, row) -> None:
    """담당 동 목록 + 영향권 규모.

    **한 후보가 여러 동을 맡는다(최대 5개).** 어느 동들인지 적지 않으면
    A동에서 클릭한 후보와 B동에서 클릭한 후보가 같을 때 혼란이 생긴다.
    """
    dongs = [d for d in str(row["dongs"]).split(" · ") if d]
    sc = placement_scope().get(cand_id, {})
    n_dong = int(row["n_dong_scope"])
    n_un = int(sc.get("n_unmapped", 0))

    # 지도에 그려지는 경계 수가 동 수보다 적을 수 있다 — 침묵하면 실무자가 세어
    # 보고 버그로 읽는다. 사유(경계 파일 차이)는 한 마디로 줄인다.
    gap = note_html(
        f"지도에는 {len(sc.get('polys', []))}개만 그려집니다(경계 파일 차이). "
        f"계산에는 {n_dong}개가 다 들어갑니다.") if n_un else ""

    # **패널은 한 번의 markdown 으로 낸다.** `<div class="panel">` 만 따로 부르면
    # Streamlit 이 그 호출을 독립 요소로 감싸 **빈 사각형**을 하나 만들고, 닫는
    # 태그도 또 하나를 만든다. 조각내지 말 것.
    st.markdown(
        '<div class="panel">'
        f'<div class="dong-name">담당 동 {len(dongs)}개</div>'
        '<div class="dong-sub">이 후보를 거점으로 배정받은 동</div>'
        + rows_html([(f"{i}", d) for i, d in enumerate(dongs, 1)])
        + rows_html([("3km 영향권", f"{n_dong}개 동"),
                     ("영향권 콜", f"{int(row['n_calls_scope']):,}건")])
        + note_html("담당 동은 이 후보를 거점으로 배정받은 동, 영향권은 3km 안의 "
                    "동 전부입니다.")
        + gap
        + '</div>', unsafe_allow_html=True)


def render_sim_page() -> None:
    ss = st.session_state
    st.markdown(f'<div class="app-title">{APP_TITLE}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="app-sub">배치안 시뮬레이션 · {SIM_BASIS}</div>',
                unsafe_allow_html=True)

    candidate_picker(st.sidebar)

    row = placement_row(ss.sim_cand) if ss.sim_cand is not None else None
    # 분포는 **후보 목록 아래**다. 고른 후보가 있어야 표식을 찍을 수 있다.
    if row is not None:
        st.sidebar.divider()
        distribution_panel(st.sidebar, row)

    st.sidebar.divider()
    if st.sidebar.button("←  지도로 돌아가기", type="secondary"):
        ss.page = 2 if ss.selected else 1
        st.rerun()

    if row is None:
        st.markdown(
            '<div class="panel"><div class="warn"><b>결과가 없습니다</b><br>'
            '이 후보의 시뮬 결과가 <code>outputs/placement_grades.csv</code> 에 '
            '없습니다. <code>python src/evaluate.py</code> 로 244곳을 돌린 뒤 다시 '
            '보십시오.</div></div>', unsafe_allow_html=True)
        if st.button("←  지도로", type="secondary"):
            ss.page = 2 if ss.selected else 1
            st.rerun()
        return

    # ── 후보 정보
    st.markdown(f'<div class="dong-name">{row["cand_name"]} ({row["gu"]})</div>',
                unsafe_allow_html=True)
    st.markdown(
        f'<div class="dong-sub">콜 {int(row["n_calls_scope"]):,}건 · '
        f'동 {int(row["n_dong_scope"])}개 · '
        f'총절감 {row["total_delta_min"]:,.0f}분</div>', unsafe_allow_html=True)

    # **카드는 셋이다**(08.12). 넷째로 「픽업 before → after」 가 있었는데 아래
    # 대비 블록이 같은 숫자를 다시 낸다. 같은 값이 두 번 뜨면 블록의 대비 효과가
    # 약해진다 — before/after 는 두 자를 나란히 놓을 때 의미가 있는 값이라
    # 그쪽에 맡기고, 카드는 판정 지표 셋만 남긴다.
    m1, m2, m3 = st.columns(3)
    m1.metric("픽업 개선율", f"{row['rate_pct']:+.2f}%",
              help="3km 안 동네들의 대기가 평균 몇 % 줄어드는지. "
                   "콜이 많은 동에 더 무게를 둡니다.")
    m2.metric("대비 구간", f"± {row['half']:.2f}%p",
              help="시뮬을 세 번 돌렸을 때 값이 흔들린 폭.")
    m3.metric("총 절감", f"{row['total_delta_min']:,.0f}분",
              help="3km 안에서 줄어든 대기 시간을 전부 더한 값. 한 달 기준입니다.")

    # **전제는 결과 바로 뒤, 지도보다 앞이다.** 아래 경고들과 같은 층에 두면 늦다.
    # 대비 블록(`sim_total_wait`)이 우측 패널로 가면서 이 자리가 카드 바로
    # 다음이 됐다 — 전제는 카드에도 블록에도 걸리므로 둘보다 위에 온다.
    sim_scenario(row)

    # 지도 : 패널 = 4:1 → 3:1 → **2:1**. 대비 블록이 패널로 오면서 한 번,
    # 블록 글자를 「담당 동」 크기로 올리면서 또 한 번 넓혔다.
    #
    # **글자를 키우면 폭이 따라와야 한다.** 값 줄이 "18.67 → 16.62분 (−10.99%)
    # [참고]" 인데 값을 .95rem 으로 올리면 약 300px 이 든다. 3:1 은 패널 안쪽이
    # 약 264px 이라 배지가 아래로 접혔고, 접히면 그 줄만 높아져 두 줄의 정렬이
    # 깨진다 — 정렬이 이 블록의 전부다. 2:1 이면 안쪽이 약 363px 로 여유가 있다.
    # 지도는 단계구분도라 폭이 줄어도 읽는 데 지장이 없다.
    left, right = st.columns([2, 1], gap="medium")

    with left:
        sc = placement_scope().get(int(row["cand_id"]), {})
        n_poly, n_un = len(sc.get("polys", [])), int(sc.get("n_unmapped", 0))
        drawn = (f'{int(row["n_dong_scope"])}개 동' if not n_un
                 else f'{int(row["n_dong_scope"])}개 동(경계 {n_poly}개)')
        st.markdown('<div class="map-title">3km 영향권</div>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<div class="map-sub">영향권 {drawn}을 한 색으로 칠했습니다 · '
            f'진한 테두리가 담당 동 · 주황 점이 후보 위치</div>',
            unsafe_allow_html=True)
        # 영향권 안이 한 색인 이유(동별 before/after 가 없다)는 명세 3절에 적어 뒀다.
        # 화면 문단은 걷어냈다 — 지도 부제의 "한 색으로 칠했습니다"가 사실을 말한다.
        st.plotly_chart(build_sim_map(int(row["cand_id"]), row),
                        width="stretch", key=f"simmap-{int(row['cand_id'])}")

    with right:
        if st.button("←  지도로", type="secondary"):
            ss.page = 2 if ss.selected else 1
            st.rerun()

        # 패널 순서: 대비 블록 → 담당 동 → 경고(있을 때만).
        # **블록이 맨 위인 이유** — 지도와 나란히 서는 값이라 지도를 보며
        # 눈을 옮기는 거리가 짧아야 한다. 담당 동은 지도의 범례에 가깝고,
        # 경고는 있을 때만 뜨므로 맨 아래여야 순서가 흔들리지 않는다.
        sim_total_wait(row)
        sim_dong_panel(int(row["cand_id"]), row)
        sim_warnings(row)

        with st.expander("후보 제원"):
            cand_spec_rows(int(row["cand_id"]))

        with st.expander("시드 3회 원값"):
            ev = load_placement_seeds()
            sub = ev[ev["cand_id"] == row["cand_id"]].sort_values("seed")
            if len(sub):
                # 총 대기는 **개선율 한 칸만** 붙인다. before/after 를 분으로
                # 실으면 실측과 견주게 되는데 그 비교는 성립하지 않는다(A-19).
                # 위 참고 행이 적은 σ 를 여기서 눈으로 확인할 수 있어야 하므로
                # 시드별 값 자체는 낸다.
                has_tw = "rate_total_pct" in sub.columns
                cells = "".join(
                    f'<tr><th>{int(s.seed)}</th><td>{s.before:.2f}</td>'
                    f'<td>{s.after:.2f}</td><td>{s.rate_pct:+.2f}%</td>'
                    + (f'<td class="aux-td">{s.rate_total_pct:+.2f}%</td>'
                       if has_tw and pd.notna(s.rate_total_pct) else
                       ('<td class="aux-td">—</td>' if has_tw else ''))
                    + '</tr>'
                    for s in sub.itertuples())
                head_tw = '<th class="aux-td">총 대기</th>' if has_tw else ''
                st.markdown(
                    '<table class="grid"><thead><tr><th>시드</th><th>before</th>'
                    f'<th>after</th><th>픽업</th>{head_tw}</tr></thead>'
                    f'<tbody>{cells}</tbody></table>'
                    # 흔들림이 무엇인지는 「대비 구간」 물음표가 이미 말한다.
                    + note_html(f"before·after 는 픽업(분), 오른쪽 두 열은 "
                                f"개선율(%). 시드 간 σ {row['rate_sd']:.3f}%p"),
                    unsafe_allow_html=True)
            else:
                st.markdown(note_html("원값 파일이 없습니다."),
                            unsafe_allow_html=True)


if __name__ == "__main__":
    main()
