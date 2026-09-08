"""
candidates.py — 동별 거점 후보 배정

옥외 거점 후보를 행정동 단위 시뮬레이션에 사용할 대표 후보로 배정한다.

배정 규칙
1. 동 내부에 후보가 있으면 총 면수가 가장 큰 1곳 선택
2. 내부 후보가 없으면 중심점 1km 이내 최근접 후보 배정
3. 1.0~1.2km 경계 구간은 허용하되, 그 밖은 미배정 처리

1km 기준은 후보 접근성과 거리 왜곡의 균형을 고려해 설정했다.
1.2km를 초과한 대체 배정은 오류로 처리해 과도하게 먼 후보가 시뮬에
사용되지 않도록 한다.

산지 등으로 기하학적 중심점이 실제 생활권과 크게 어긋나는 일부 동은
`recalc_representative_points()`로 대표점을 보정한다.

미배정 동은 수요 부족이 아니라 후보 풀의 제약으로 적합한 옥외 후보를
찾지 못한 경우이며, 진단 대상에서는 제외하지 않는다.

동 경계는 행정동 기준이며 후보의 법정동 명칭 대신 좌표 공간조인을 사용한다.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import load
import travel_time

# 대체 반경. 근거는 모듈 docstring.
NEAR_RADIUS_M = 1000.0
EDGE_RADIUS_M = 1200.0

# 공항·개발제한으로 구조적 공백인 동. 후보를 못 찾는 게 아니라 놓을 땅이 없다.
# 파이프라인 6절은 법정동 기준으로 오곡·오쇠·과해·항동을 든다. 이 모듈은
# 행정동 격자라 아래로 옮긴다 — 오곡·오쇠·과해는 행정동 '공항동'에 속하고,
# 항동은 그 자체가 행정동이다(2019년 항동지구 개발로 오류2동에서 분리).
AIRPORT_GREENBELT = {
    ("강서구", "공항동"),
    ("구로구", "항동"),
}

# 콜 발생 가중 중심으로 대표점을 다시 잡을 대상(산지형 7동).
# 지정은 법정동 이름으로 받았고(산지 6곳), 이 모듈의 행정동 격자에서는 아래로 펴진다.
# 평창은 행정동 평창동이 이미 내부 배정(비봉(구) 1,137m)이라 재계산 대상에서만
# 이름이 남고, 구기는 행정동이 아니라 법정동이다(행정동 평창동에 속한다).
CENTROID_RECALC_BDONG = ("도봉동", "정릉동", "평창동", "공릉동", "구기동", "홍은동")
CENTROID_RECALC_DONG = {
    ("도봉구", "도봉1동"), ("노원구", "공릉2동"), ("서대문구", "홍은2동"),
    ("성북구", "정릉1동"), ("성북구", "정릉2동"),
    ("성북구", "정릉3동"), ("성북구", "정릉4동"),
}

# 대표점 재계산 파라미터. 근거는 recalc_representative_points docstring.
REP_GRID_M = 100.0            # 폴리곤 탐색 격자
REP_MIN_DEST = 8              # 목적동이 이보다 적으면 추정하지 않는다
REP_MIN_TRIPS_PER_DEST = 5    # 목적동별 최소 완료 건수
REP_MIN_KM = 1.5              # 근거리는 우회계수가 크게 흔들려 제외

RULE_INSIDE = "내부"
RULE_NEAR = "1km대체"
RULE_EDGE = "경계선상"
RULE_AIRPORT = "공항·개발제한"
RULE_PENDING = "보류_대표점"

# 배정된 동(시뮬이 돌릴 수 있는 동)
ASSIGNED_RULES = (RULE_INSIDE, RULE_NEAR, RULE_EDGE)

EARTH_R_M = 6371008.8


def haversine_m(lat1, lon1, lat2, lon2):
    """두 좌표 사이 대권거리(m). numpy 브로드캐스팅으로 행렬도 받는다.

    서울 규모(30km)에서 평면 근사와 1m 이내로 갈리지 않으므로 투영 없이 쓴다.
    """
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_R_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def load_dong_frame() -> pd.DataFrame:
    """배정 대상 동 426개. 중심점 + 경계코드 + 면적.

    면적은 보류 버킷을 읽을 때 필요하다 — "중심점이 대표점으로 부적절"의
    근거가 동이 넓다는 사실이라, 숫자 없이 주장하면 확인할 수가 없다.
    면적은 EPSG:5179(미터)로 투영해 계산한다.

    반환 컬럼: gu, dong_canon, adm_nm, adm_cd2, dong_lat, dong_lon, area_km2
    """
    cen = load.load_dong()                        # 426행, 중심점
    gdf = load.load_dong(with_boundary=True)      # 426 폴리곤, EPSG:4326

    area = gdf.to_crs(5179)
    gdf = gdf.assign(area_km2=area.geometry.area.to_numpy() / 1e6)

    out = cen.merge(gdf[["gu", "dong_canon", "adm_cd2", "area_km2"]],
                    on=["gu", "dong_canon"], how="left", validate="one_to_one")
    if out["adm_cd2"].isna().any():
        bad = out.loc[out["adm_cd2"].isna(), ["gu", "dong_canon"]]
        raise ValueError(f"경계와 매칭되지 않는 동: {bad.to_dict('records')}")

    return out.rename(columns={"lat": "dong_lat", "lon": "dong_lon"})[
        ["gu", "dong_canon", "adm_nm", "adm_cd2", "dong_lat", "dong_lon", "area_km2"]]


def locate_candidates(cands: pd.DataFrame) -> pd.DataFrame:
    """후보 좌표를 행정동 경계에 공간조인해 `adm_cd2` 를 붙인다.

    주소 문자열에서 동을 뽑지 않는 이유는 파이프라인 3-10과 같다 —
    '성수동1가'·'당산동3가' 같은 표기가 별개 동으로 잡힌다.

    경계 밖으로 떨어지는 후보는 `adm_cd2` 가 NaN 으로 남는다(전부 배정되면 0곳).
    """
    import geopandas as gpd   # 경계가 필요할 때만 — load.load_dong 과 같은 이유

    gdf = load.load_dong(with_boundary=True)
    pts = gpd.GeoDataFrame(
        cands, geometry=gpd.points_from_xy(cands["lon"], cands["lat"]), crs=4326)
    joined = gpd.sjoin(pts, gdf[["adm_cd2", "geometry"]], how="left", predicate="within")
    # 경계가 겹치면 한 점이 두 폴리곤에 잡힐 수 있다. 첫 건만 남긴다.
    joined = joined[~joined.index.duplicated(keep="first")]
    return pd.DataFrame(joined.drop(columns=["geometry", "index_right"]))


# ─────────────────────────────────────────────────────────────
# 대표점 재계산 — 산지형 동
# ─────────────────────────────────────────────────────────────

def destination_profile(calls: pd.DataFrame) -> pd.DataFrame:
    """(출발동 → 목적동) 실측 승차거리 중앙값·건수 표. 대표점 역산의 입력.

    완료 건 · 승차거리 > 0 · 목적지 서울 · 목적동 좌표 있음만 쓴다. 거리 0 은
    결측 대체값이라 그대로 두면 역산이 원점으로 끌린다(calibration.md 흡수 사실 ⑴).

    반환 컬럼: origin_gu, origin_dong_canon, dest_gu, dest_dong_canon, n, d, lat, lon
    """
    ok = (calls["boarded_at"].notna() & (calls["ride_distance_m"] > 0)
          & calls["dest_in_seoul"] & calls["dest_lat"].notna())
    r = calls.loc[ok, ["origin_gu", "origin_dong_canon", "dest_gu", "dest_dong_canon",
                       "dest_lat", "dest_lon", "ride_distance_m"]]
    prof = (r.groupby(["origin_gu", "origin_dong_canon", "dest_gu", "dest_dong_canon"],
                      observed=True)
             .agg(n=("ride_distance_m", "size"), d=("ride_distance_m", "median"),
                  lat=("dest_lat", "first"), lon=("dest_lon", "first"))
             .reset_index())
    return prof[(prof["n"] >= REP_MIN_TRIPS_PER_DEST)
                & (prof["d"] >= REP_MIN_KM * 1000)].reset_index(drop=True)


def _grid_points(geom, grid_m: float = REP_GRID_M):
    """폴리곤(EPSG:5179) 내부 격자점을 (lat, lon) 배열로. 비면 대표점 하나."""
    import geopandas as gpd
    from shapely.geometry import Point

    minx, miny, maxx, maxy = geom.bounds
    xx, yy = np.meshgrid(np.arange(minx, maxx + grid_m, grid_m),
                         np.arange(miny, maxy + grid_m, grid_m))
    pts = gpd.GeoSeries([Point(x, y) for x, y in zip(xx.ravel(), yy.ravel())], crs=5179)
    inside = pts[pts.within(geom)]
    if inside.empty:
        inside = gpd.GeoSeries([geom.representative_point()], crs=5179)
    g = inside.to_crs(4326)
    return np.array([p.y for p in g]), np.array([p.x for p in g])


def recalc_representative_points(targets, calls: pd.DataFrame = None, *,
                                 grid_m: float = REP_GRID_M) -> pd.DataFrame:
    """산지형 동의 대표점을 실측 승차거리로 역산한다.

    **왜 필요한가.** 동 중심점은 기하 중심이라 산지가 면적의 대부분을 먹는 동에서는
    사람이 사는 곳에서 크게 벗어난다. 도봉1동 중심점은 도봉산 안이고, 정릉1~4동은
    북한산이다. 그 점에서 후보까지 거리를 재면 실재하는 후보가 1.2km 밖으로 밀려
    "후보 없음"이 된다.

    **왜 콜 좌표를 직접 평균내지 않는가.** 콜 원본에 출발 지점 좌표가 없다 —
    출발동만 있고, `load.load_calls` 가 붙이는 `origin_lat/lon` 은 그 동의 중심점이라
    동 안에서 전부 같은 값이다. 문자 그대로의 '콜 발생 가중 중심'은 산출 불가다.

    **대신 승차거리로 역산한다.** 출발동 A → 목적동 B 완료 건의 실측 승차거리
    중앙 d_B 와 건수 n_B 를 모으면, A 의 대표점 p 는

        min over (p, k)  Σ_B  n_B · (k · haversine(p, c_B) − d_B)²

    를 푸는 점이다(c_B = 목적동 중심점, k = 우회계수). k 는 p 가 주어지면 닫힌
    형태 k* = Σn·h·d / Σn·h² 로 나오므로, 폴리곤 격자를 전수 탐색하면 된다.
    p 는 폴리곤 안으로 제한한다 — 동 밖의 점은 그 동의 대표점일 수 없다.

    **이 추정량이 수렴하는 값이 곧 콜 가중 출발지점의 평균이다.** 목적동 중심점
    오차는 방향이 흩어져 상쇄되고, 출발지가 동 안에 퍼져 있다는 사실은 그대로
    남는다. k 를 함께 추정하므로 우회계수 설정값에 기대지 않는다 — 실제로 추정된
    k 중앙은 1.33 으로 `travel_time.DETOUR`(1.33)와 독립적으로 맞아떨어진다.

    **정확도.** 대표점 문제가 없는 동(면적 최소 40곳)에 같은 방법을 적용하면
    중심점에서 중앙 317m · p90 488m 움직인다. 이것이 방법의 잡음 바닥이다.
    이동량은 면적과 함께 커진다(1km² 미만 394m → 4km² 이상 1,409m) — 큰 동일수록
    기하 중심이 대표성을 잃는다는 예상과 맞는다.

    **전 동에 적용하지 않는다.** 정상 동의 보정폭(394m)이 잡음 바닥(317m)과 같은
    크기라, 전면 적용하면 이득 없이 좌표만 흔들린다. 중심점이 대표점으로 부적절함이
    따로 확인된 동(`CENTROID_RECALC_DONG`)에만 쓴다.

    targets: (gu, dong_canon) 쌍의 집합/리스트
    calls: 없으면 load.load_calls()

    반환 컬럼: gu, dong_canon, lat, lon, detour_k, move_m, n_dest, n_trips, rmse_m
              (추정 불가한 동은 행이 없다)
    """
    if calls is None:
        calls = load.load_calls()
    prof = destination_profile(calls)

    gdf = load.load_dong(with_boundary=True).to_crs(5179)
    cen = load.load_dong().set_index(["gu", "dong_canon"])

    rows = []
    for gu, dong in sorted(targets):
        p = prof[(prof["origin_gu"] == gu) & (prof["origin_dong_canon"] == dong)]
        poly = gdf[(gdf["gu"] == gu) & (gdf["dong_canon"] == dong)]
        if len(p) < REP_MIN_DEST or poly.empty or (gu, dong) not in cen.index:
            continue

        glat, glon = _grid_points(poly.geometry.iloc[0], grid_m)
        h = haversine_m(glat[:, None], glon[:, None],
                        p["lat"].to_numpy()[None, :], p["lon"].to_numpy()[None, :])
        n = p["n"].to_numpy()[None, :]
        d = p["d"].to_numpy()[None, :]
        with np.errstate(invalid="ignore", divide="ignore"):
            k = (n * h * d).sum(axis=1) / (n * h * h).sum(axis=1)
        sse = (n * (k[:, None] * h - d) ** 2).sum(axis=1)
        i = int(np.nanargmin(sse))

        c = cen.loc[(gu, dong)]
        rows.append({
            "gu": gu, "dong_canon": dong,
            "lat": float(glat[i]), "lon": float(glon[i]),
            "detour_k": float(k[i]),
            "move_m": float(haversine_m(c["lat"], c["lon"], glat[i], glon[i])),
            "n_dest": len(p), "n_trips": int(p["n"].sum()),
            "rmse_m": float(np.sqrt(sse[i] / n.sum())),
        })
    return pd.DataFrame(rows, columns=["gu", "dong_canon", "lat", "lon", "detour_k",
                                       "move_m", "n_dest", "n_trips", "rmse_m"])


def assign_dong_candidates(cands: pd.DataFrame = None, *,
                           near_m: float = NEAR_RADIUS_M,
                           edge_m: float = EDGE_RADIUS_M,
                           rep_points: pd.DataFrame = None) -> pd.DataFrame:
    """동 426개에 후보를 배정한다. 규칙은 모듈 docstring 참조.

    cands 를 주지 않으면 `load.load_candidates()`(옥외 348곳)를 쓴다.
    옥내·혼합을 넣어 돌리고 싶으면 그 풀을 직접 만들어 넘기면 된다.

    `nearest_dist_m` 는 배정 여부와 무관하게 항상 채운다 — 보류·제외 동이
    얼마나 멀어서 빠졌는지가 그 판정의 근거라, 비워두면 확인할 수가 없다.
    `dist_m`(배정된 후보까지 거리)은 배정된 동에만 있다.

    `capacity` 는 시뮬 입력용 총 면수다. 배정 순위도 이 값으로만 낸다.

    `rep_points` 를 주면(`recalc_representative_points` 산출) 해당 동의 `dong_lat`
    /`dong_lon` 을 그 값으로 갈아끼운 뒤 거리를 잰다. 어느 동이 갈렸는지는
    `rep_source` 컬럼에 남는다(`centroid` / `call_weighted`). **⑴ 내부 판정에는
    영향이 없다** — 후보가 동 경계 안에 있는지는 대표점과 무관하다.

    반환 컬럼:
      gu, dong_canon, adm_nm, adm_cd2, dong_lat, dong_lon, area_km2, rep_source,
      assign_rule, is_assigned, n_outdoor_in_dong, nearest_dist_m,
      cand_id, cand_name, capacity, source, cand_lat, cand_lon, dist_m, cand_beyond_cap
    """
    if cands is None:
        cands = load.load_candidates()
    if not len(cands):
        raise ValueError("후보 풀이 비어 있다")

    dong = load_dong_frame()
    dong["rep_source"] = "centroid"
    if rep_points is not None and len(rep_points):
        idx = rep_points.set_index(["gu", "dong_canon"])
        key = pd.MultiIndex.from_frame(dong[["gu", "dong_canon"]])
        hit = key.isin(idx.index)
        if not hit.any():
            raise ValueError("rep_points 의 동이 배정 대상 426개와 하나도 맞지 않는다")
        dong.loc[hit, "dong_lat"] = key[hit].map(idx["lat"])
        dong.loc[hit, "dong_lon"] = key[hit].map(idx["lon"])
        dong.loc[hit, "rep_source"] = "call_weighted"

    located = locate_candidates(cands)

    inside = located[located["adm_cd2"].notna()].copy()
    n_outside_boundary = int(located["adm_cd2"].isna().sum())

    # 동 중심점 × 후보 전체 거리행렬 (426 × 348). 이 크기면 통째로 잡는 게
    # 동마다 도는 것보다 빠르고, ⑵⑶ 판정과 nearest_dist_m 이 같은 값을 쓴다.
    dist = haversine_m(dong["dong_lat"].to_numpy()[:, None],
                       dong["dong_lon"].to_numpy()[:, None],
                       cands["lat"].to_numpy()[None, :],
                       cands["lon"].to_numpy()[None, :])
    j_near = dist.argmin(axis=1)
    dong = dong.assign(nearest_dist_m=dist[np.arange(len(dong)), j_near])

    # ⑴ 동 내부 — 총 면수(capacity) 최대 1곳.
    # 동점이면 중심점에 가까운 쪽, 그래도 같으면 cand_id 로 결정해 실행마다
    # 결과가 흔들리지 않게 한다.
    cen = dong.set_index("adm_cd2")[["dong_lat", "dong_lon"]]
    inside["dist_m"] = haversine_m(
        inside["adm_cd2"].map(cen["dong_lat"]).to_numpy(),
        inside["adm_cd2"].map(cen["dong_lon"]).to_numpy(),
        inside["lat"].to_numpy(), inside["lon"].to_numpy())

    n_in_dong = inside.groupby("adm_cd2").size().rename("n_outdoor_in_dong")
    best = (inside.sort_values(["capacity", "dist_m", "cand_id"],
                               ascending=[False, True, True])
                  .groupby("adm_cd2", as_index=False).head(1)
                  .set_index("adm_cd2"))

    # ⑵⑶ 나머지 동 — 중심점에서 가장 가까운 후보(동 안팎을 가리지 않는다)
    is_rest = ~dong["adm_cd2"].isin(best.index)
    nearest = cands.iloc[j_near[is_rest.to_numpy()]].copy()
    nearest.index = dong.loc[is_rest, "adm_cd2"].to_numpy()
    nearest["dist_m"] = dong.loc[is_rest, "nearest_dist_m"].to_numpy()

    keep = ["cand_id", "name", "capacity", "source", "lat", "lon", "dist_m"]
    picked = pd.concat([best[keep], nearest[keep]])

    out = dong.merge(
        picked.rename(columns={"name": "cand_name",
                               "lat": "cand_lat", "lon": "cand_lon"}),
        left_on="adm_cd2", right_index=True, how="left", validate="one_to_one")
    out = out.merge(n_in_dong, left_on="adm_cd2", right_index=True, how="left")
    out["n_outdoor_in_dong"] = out["n_outdoor_in_dong"].fillna(0).astype(int)

    # 규칙 라벨. 거리 판정을 먼저 하고 성격은 그 뒤에 본다 — 1.2km 안에
    # 실제 후보가 있으면 공항동이든 산지든 그 후보를 쓰는 게 맞다.
    is_inside = out["adm_cd2"].isin(best.index)
    key = list(zip(out["gu"], out["dong_canon"]))
    rule = np.where(
        is_inside, RULE_INSIDE,
        np.where(out["nearest_dist_m"] <= near_m, RULE_NEAR,
                 np.where(out["nearest_dist_m"] <= edge_m, RULE_EDGE,
                          np.where([k in AIRPORT_GREENBELT for k in key],
                                   RULE_AIRPORT, RULE_PENDING))))
    out["assign_rule"] = rule
    out["is_assigned"] = out["assign_rule"].isin(ASSIGNED_RULES)

    # 배정 안 된 동은 후보 칸을 비운다. 값이 남아 있으면 "가장 가까운 후보"가
    # "배정된 후보"로 잘못 읽힌다.
    blank = ["cand_id", "cand_name", "capacity",
             "source", "cand_lat", "cand_lon", "dist_m"]
    out.loc[~out["is_assigned"], blank] = np.nan
    out["cand_id"] = out["cand_id"].astype("Int64")

    # 대체 배정에 상한 초과가 남아 있으면 시뮬이 엉뚱한 자리를 거점으로 쓴다.
    # 컬럼으로 표시만 하는 게 아니라 여기서 끊는다 — 위 라벨링이 어긋나면 터진다.
    substitute = out["is_assigned"] & out["assign_rule"].ne(RULE_INSIDE)
    over = out.loc[substitute & (out["dist_m"] > edge_m)]
    if len(over):
        raise AssertionError(
            f"대체 배정 상한({edge_m:.0f}m) 초과가 남았다: "
            f"{over[['gu', 'dong_canon', 'dist_m']].to_dict('records')}")

    # ⑴ 내부는 상한을 걸지 않는다 — 후보가 그 동 **안에** 있어서 '엉뚱한 자리'가
    # 아니고, 상한으로 자르면 실재하는 후보를 두고 미배정이 된다. 다만 중심점이
    # 멀리 있다는 사실 자체는 대표점 문제라 표시해 둔다(대상 11곳).
    out["cand_beyond_cap"] = out["is_assigned"] & (out["dist_m"] > edge_m)

    out.attrs["n_candidates"] = len(cands)
    out.attrs["n_outside_boundary"] = n_outside_boundary
    out.attrs["near_m"], out.attrs["edge_m"] = near_m, edge_m

    cols = ["gu", "dong_canon", "adm_nm", "adm_cd2", "dong_lat", "dong_lon",
            "area_km2", "rep_source", "assign_rule", "is_assigned", "n_outdoor_in_dong",
            "nearest_dist_m",
            "cand_id", "cand_name", "capacity", "source", "cand_lat", "cand_lon", "dist_m", "cand_beyond_cap"]
    return out[cols].sort_values(["gu", "dong_canon"]).reset_index(drop=True)


def assignment_summary(assign: pd.DataFrame) -> pd.DataFrame:
    """규칙별 동 수·중앙 대체거리·서로 다른 후보 수.

    후보 수를 따로 세는 이유는 대체 배정이 겹치기 때문이다 — 여러 동이 같은
    후보 한 곳을 가리킬 수 있고, 그 경우 실제 배치 지점은 동 수보다 적다.
    """
    order = [RULE_INSIDE, RULE_NEAR, RULE_EDGE, RULE_AIRPORT, RULE_PENDING]
    rows = []
    for rule in order:
        g = assign[assign["assign_rule"] == rule]
        d = g["dist_m"].dropna()
        rows.append({
            "assign_rule": rule,
            "n_dong": len(g),
            "n_cand": int(g["cand_id"].nunique()),
            "dist_median_m": d.median() if len(d) else np.nan,
            "dist_max_m": d.max() if len(d) else np.nan,
            "capacity_sum": g["capacity"].sum(),
        })
    return pd.DataFrame(rows)


OUTPUT_PATH = Path(__file__).resolve().parent.parent / "outputs" / "dong_candidates.csv"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    cands = load.load_candidates()
    enc = cands.attrs["n_by_enclosure"]
    print(f"후보 풀 {cands.attrs['n_pool']}곳 → 옥외 {len(cands)}곳 "
          f"{cands['capacity'].sum():,}면 "
          f"(옥내 {enc.get('옥내', 0)} · 혼합 {enc.get('혼합', 0)} 제외 — A-15)")
    print("시뮬 입력 용량 = 총 면수(전 후보 동일)")

    print("\n대표점 재계산 중 (산지형 7동 — 콜 로딩 포함)...")
    rep = recalc_representative_points(CENTROID_RECALC_DONG)
    before = assign_dong_candidates(cands)
    assign = assign_dong_candidates(cands, rep_points=rep)

    print(f"\n[대표점 재계산] {len(rep)}/{len(CENTROID_RECALC_DONG)}개 동 추정 성공")
    rep_view = rep.merge(
        before[["gu", "dong_canon", "nearest_dist_m"]].rename(
            columns={"nearest_dist_m": "구_최근접m"}), on=["gu", "dong_canon"])
    rep_view = rep_view.merge(
        assign[["gu", "dong_canon", "nearest_dist_m", "assign_rule"]].rename(
            columns={"nearest_dist_m": "신_최근접m"}), on=["gu", "dong_canon"])
    print(rep_view[["gu", "dong_canon", "n_trips", "move_m", "구_최근접m",
                    "신_최근접m", "assign_rule", "detour_k", "rmse_m"]]
          .to_string(index=False, formatters={
              "move_m": "{:,.0f}".format, "구_최근접m": "{:,.0f}".format,
              "신_최근접m": "{:,.0f}".format, "rmse_m": "{:,.0f}".format,
              "detour_k": "{:.3f}".format}))
    n_saved = int((before.set_index(["gu", "dong_canon"])
                   .loc[list(zip(rep["gu"], rep["dong_canon"])), "is_assigned"]
                   .to_numpy() == False).sum()
                  - (~assign.set_index(["gu", "dong_canon"])
                     .loc[list(zip(rep["gu"], rep["dong_canon"])), "is_assigned"]
                     .to_numpy()).sum())
    print(f"  구제 {n_saved}개 동 · 추정 우회계수 중앙 {rep['detour_k'].median():.3f}"
          f" (travel_time.DETOUR = {travel_time.DETOUR})")

    print(f"\n동 {len(assign)}개 · 경계 밖 후보 {assign.attrs['n_outside_boundary']}곳\n")

    summ = assignment_summary(assign)
    print(summ.to_string(index=False,
                         formatters={"dist_median_m": lambda v: f"{v:,.0f}" if pd.notna(v) else "—",
                                     "dist_max_m": lambda v: f"{v:,.0f}" if pd.notna(v) else "—",
                                     "capacity_sum": lambda v: f"{v:,.0f}"}))
    n_ok = int(assign["is_assigned"].sum())
    print(f"\n최종 배정 {n_ok}개 동 / {len(assign)} ({n_ok / len(assign):.1%})"
          f" · 서로 다른 후보 {assign['cand_id'].nunique()}곳")
    print(f"대체 배정 1.2km 초과 {int((assign['is_assigned'] & assign['assign_rule'].ne(RULE_INSIDE) & (assign['dist_m'] > EDGE_RADIUS_M)).sum())}곳"
          f" · 내부 배정 중 중심점 1.2km 초과 {int(assign['cand_beyond_cap'].sum())}곳(표시만)")

    un = assign[~assign["is_assigned"]]
    print(f"\n미배정 {len(un)}개 동 — 후보 없음으로 확정")
    metrics_path = OUTPUT_PATH.parent / "dong_metrics.csv"
    if metrics_path.exists():
        m = pd.read_csv(metrics_path)[["gu", "dong_canon", "n_calls", "wait_total_mean"]]
        un = un.merge(m, on=["gu", "dong_canon"], how="left")
        big = un[un["n_calls"] >= 100]
        print(f"  콜 100건 이상 {len(big)}개 동 (합 {big['n_calls'].sum():,.0f}건)"
              " — 수요가 없어서 빠지는 게 아니다")
        print(big.nlargest(10, "n_calls")[
            ["gu", "dong_canon", "n_calls", "wait_total_mean", "nearest_dist_m"]]
            .to_string(index=False, float_format=lambda v: f"{v:,.1f}"))

    pending = assign[assign["assign_rule"] == RULE_PENDING]
    print(f"\n보류(대표점 재검토) {len(pending)}개 동 — 콜 정제 이후 재산출")
    print(pending[["gu", "dong_canon", "area_km2", "nearest_dist_m"]]
          .sort_values("area_km2", ascending=False)
          .to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

    named = pending[[(g, d) in CENTROID_RECALC_DONG
                     for g, d in zip(pending["gu"], pending["dong_canon"])]]
    print(f"\n  그 중 재계산 지정 {len(named)}개 동 "
          f"(법정동 {' · '.join(CENTROID_RECALC_BDONG)}) — 재계산 후에도 남은 것:")
    print("  " + (" · ".join(f"{g} {d}" for g, d in
                             zip(named["gu"], named["dong_canon"])) or "없음"))

    air = assign[assign["assign_rule"] == RULE_AIRPORT]
    print(f"\n공항·개발제한 {len(air)}개 동 — 시뮬 대상 제외")
    print(air[["gu", "dong_canon", "area_km2", "nearest_dist_m"]]
          .to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    assign.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n저장: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
