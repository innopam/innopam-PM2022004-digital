#!/usr/bin/env python3
"""
DXF → SHP 통합 변환 모듈

기능:
  1. DXF 로드 + 'B'로 시작하는 레이어만 추출 (건물 레이어)
  2. 색상별 분류:
       - 마젠타 (ACI 6) → '수정도화'
       - 검정/흰색 (ACI 7) → '기성건물'
       - 기타 색 → 'other'
     (BYLAYER 256 시 레이어 색을 따름)
  3. LineString 추출 (LINE / LWPOLYLINE / POLYLINE)
  4. 끝점 KDTree 매칭으로 열린 라인 체인 합치기 (merge_open_lines_v2 알고리즘)
  5. 닫힌 라인 → Polygon 변환 (옵션)
  6. Polygon 전처리: force_2d + make_valid + GeometryCollection part 추출 + Polygon만 유지
     (merge_open_lines_preprocessing 알고리즘)
  7. 카테고리별 SHP 저장 (수정도화 / 기성건물 / other)

사용 예:
  python scripts/dxf_to_shp.py --input my.dxf --out-dir out/
  python scripts/dxf_to_shp.py --input my.dxf --out-dir out/ --layer-prefix B --epsg 5186
  python scripts/dxf_to_shp.py --input my.dxf --out-dir out/ --no-polygon     # 라인만 출력

요구 패키지:
  ezdxf, geopandas, shapely, scipy, numpy, pyogrio
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np
import ezdxf
import geopandas as gpd
import shapely
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union
from scipy.spatial import cKDTree


# ─── 설정 ─────────────────────────────────────────────────────────
DEFAULT_XY_TOLERANCE = 0.01   # 끝점 매칭 XY 허용오차 (m, EPSG:5186 기준)
DEFAULT_Z_TOLERANCE = 1.0     # Z 허용오차 (m)
DEFAULT_EPSG = 5186           # 한국 중부원점

LAYER_PREFIX = 'B'            # 건물 레이어 prefix

# DXF AutoCAD Color Index (ACI)
ACI_MAGENTA = 6
ACI_BLACK_OR_WHITE = 7  # 7은 도면 배경에 따라 검정 또는 흰색
ACI_BYLAYER = 256

CATEGORY_MAGENTA = '수정도화'
CATEGORY_BLACK = '기성건물'
CATEGORY_OTHER = 'other'


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ─── DXF 색상 → 카테고리 ─────────────────────────────────────────
def resolve_aci(entity, doc) -> int:
    """entity의 실제 ACI 값을 반환. BYLAYER이면 레이어 색을 따름."""
    aci = entity.dxf.color
    if aci == ACI_BYLAYER:
        layer = doc.layers.get(entity.dxf.layer)
        if layer is not None:
            aci = layer.color
    return int(aci)


def classify_color(entity, doc) -> str:
    aci = resolve_aci(entity, doc)
    if aci == ACI_MAGENTA:
        return CATEGORY_MAGENTA
    if aci == ACI_BLACK_OR_WHITE:
        return CATEGORY_BLACK
    return CATEGORY_OTHER


# ─── DXF entity → 좌표 시퀀스 ───────────────────────────────────
def entity_to_coords(entity) -> list[tuple[float, float, float]] | None:
    """LINE / LWPOLYLINE / POLYLINE → [(x, y, z), ...]"""
    t = entity.dxftype()
    if t == 'LINE':
        s, e = entity.dxf.start, entity.dxf.end
        return [(s.x, s.y, s.z), (e.x, e.y, e.z)]
    if t == 'LWPOLYLINE':
        z = float(entity.dxf.elevation)
        pts = [(float(p[0]), float(p[1]), z) for p in entity.get_points('xy')]
        if entity.closed and pts and pts[0] != pts[-1]:
            pts.append(pts[0])
        return pts
    if t == 'POLYLINE':
        pts = []
        for v in entity.vertices:
            loc = v.dxf.location
            pts.append((float(loc.x), float(loc.y), float(loc.z)))
        if entity.is_closed and pts and pts[0] != pts[-1]:
            pts.append(pts[0])
        return pts
    return None  # 기타 entity 타입(TEXT 등)은 건너뜀


# ─── DXF 로드 + 필터 + 분류 ─────────────────────────────────────
def extract_features(dxf_path: str, layer_prefix: str = LAYER_PREFIX) -> dict[str, list[LineString]]:
    log(f"DXF 로드: {dxf_path}")
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    by_cat: dict[str, list[LineString]] = defaultdict(list)
    total = 0
    skipped_layer = 0
    skipped_unsupported = 0
    prefix_upper = layer_prefix.upper()

    for ent in msp:
        layer_name = ent.dxf.layer or ''
        if not layer_name.upper().startswith(prefix_upper):
            skipped_layer += 1
            continue
        coords = entity_to_coords(ent)
        if not coords or len(coords) < 2:
            skipped_unsupported += 1
            continue
        cat = classify_color(ent, doc)
        # 3D LineString (Z 보존) — Polygon 변환 시 X/Y만 사용
        ls = LineString([(c[0], c[1], c[2]) for c in coords])
        by_cat[cat].append(ls)
        total += 1

    log(f"  추출 entity {total}건 (skip layer={skipped_layer}, unsupported={skipped_unsupported})")
    for c, items in by_cat.items():
        log(f"    {c}: {len(items)}건")
    return by_cat


# ─── 열린 라인 체인 합치기 (merge_open_lines_v2 알고리즘) ──────
def merge_open_lines(
    linestrings: list[LineString],
    xy_tol: float = DEFAULT_XY_TOLERANCE,
    z_tol: float = DEFAULT_Z_TOLERANCE,
) -> list[LineString]:
    if not linestrings:
        return []

    open_idxs: list[int] = []
    closed_idxs: list[int] = []
    for i, ls in enumerate(linestrings):
        coords = list(ls.coords)
        s = np.array(coords[0][:2])
        e = np.array(coords[-1][:2])
        if np.linalg.norm(s - e) > 0.001:
            open_idxs.append(i)
        else:
            closed_idxs.append(i)
    log(f"    open={len(open_idxs)}, closed={len(closed_idxs)}")

    if not open_idxs:
        return list(linestrings)

    # 끝점 배열 (start=0, end=1)
    n_open = len(open_idxs)
    ep_feat_idx = np.zeros(n_open * 2, dtype=int)
    ep_end_type = np.zeros(n_open * 2, dtype=int)
    ep_xy = np.zeros((n_open * 2, 2))
    ep_z = np.zeros(n_open * 2)
    for i, idx in enumerate(open_idxs):
        coords = list(linestrings[idx].coords)
        s, e = coords[0], coords[-1]
        ep_feat_idx[i * 2] = idx
        ep_end_type[i * 2] = 0
        ep_xy[i * 2] = [s[0], s[1]]
        ep_z[i * 2] = s[2] if len(s) >= 3 else 0
        ep_feat_idx[i * 2 + 1] = idx
        ep_end_type[i * 2 + 1] = 1
        ep_xy[i * 2 + 1] = [e[0], e[1]]
        ep_z[i * 2 + 1] = e[2] if len(e) >= 3 else 0

    tree = cKDTree(ep_xy)
    pairs = tree.query_pairs(r=xy_tol)
    endpoint_neighbors: dict[tuple[int, str], list[tuple[int, str]]] = defaultdict(list)
    for i, j in pairs:
        if ep_feat_idx[i] == ep_feat_idx[j]:
            continue
        if abs(ep_z[i] - ep_z[j]) > z_tol:
            continue
        key_i = (int(ep_feat_idx[i]), 'start' if ep_end_type[i] == 0 else 'end')
        key_j = (int(ep_feat_idx[j]), 'start' if ep_end_type[j] == 0 else 'end')
        endpoint_neighbors[key_i].append(key_j)
        endpoint_neighbors[key_j].append(key_i)

    # 피처 단위 인접
    feat_neighbors: dict[int, set[int]] = defaultdict(set)
    for (idx, _end), nbrs in endpoint_neighbors.items():
        for nbr_idx, _ in nbrs:
            feat_neighbors[idx].add(nbr_idx)
            feat_neighbors[nbr_idx].add(idx)

    # BFS — 체인 탐색
    visited: set[int] = set()
    chains: list[list[int]] = []
    for idx in open_idxs:
        if idx in visited:
            continue
        queue = [idx]
        visited.add(idx)
        comp = []
        while queue:
            node = queue.pop(0)
            comp.append(node)
            for nbr in feat_neighbors[node]:
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append(nbr)
        chains.append(sorted(comp))

    def is_linear_chain(chain: list[int]) -> bool:
        cset = set(chain)
        for idx in chain:
            for end in ('start', 'end'):
                nbrs = endpoint_neighbors.get((idx, end), [])
                in_chain = [n for n in nbrs if n[0] in cset]
                if len(in_chain) > 1:
                    return False
        return True

    def merge_chain(chain: list[int]) -> list[tuple]:
        if len(chain) == 1:
            return list(linestrings[chain[0]].coords)
        cset = set(chain)
        ep_degree: dict[tuple[int, str], int] = defaultdict(int)
        local_adj: dict[int, list[tuple[int, str, str]]] = defaultdict(list)
        for idx in chain:
            for end in ('start', 'end'):
                for nbr_idx, nbr_end in endpoint_neighbors.get((idx, end), []):
                    if nbr_idx in cset:
                        ep_degree[(idx, end)] += 1
                        local_adj[idx].append((nbr_idx, end, nbr_end))
        start_feat, free_end = None, None
        for idx in chain:
            for end in ('start', 'end'):
                if ep_degree[(idx, end)] == 0:
                    start_feat, free_end = idx, end
                    break
            if start_feat is not None:
                break
        if start_feat is None:
            start_feat, free_end = chain[0], 'start'

        all_coords: list[tuple] = []
        used: set[int] = set()
        current, cur_end = start_feat, free_end
        while current is not None and current not in used:
            used.add(current)
            coords = list(linestrings[current].coords)
            if cur_end == 'end':
                coords = coords[::-1]
            if all_coords:
                coords = coords[1:]
            all_coords.extend(coords)
            connected_end = 'end' if cur_end == 'start' else 'start'
            next_feat = None
            for nbr_idx, my_end, nbr_end in local_adj[current]:
                if nbr_idx not in used and my_end == connected_end:
                    next_feat = nbr_idx
                    cur_end = nbr_end
                    break
            current = next_feat
        return all_coords

    result: list[LineString] = [linestrings[i] for i in closed_idxs]
    merged_count = branch_count = 0
    for chain in chains:
        if len(chain) == 1:
            result.append(linestrings[chain[0]])
        elif is_linear_chain(chain):
            merged = merge_chain(chain)
            if len(merged) >= 2:
                result.append(LineString(merged))
                merged_count += 1
        else:
            for idx in chain:
                result.append(linestrings[idx])
            branch_count += 1
    log(f"    merged={merged_count}, branch={branch_count}, total={len(result)}")
    return result


# ─── 닫힌 LineString → Polygon ──────────────────────────────────
def closed_lines_to_polygons(
    linestrings: list[LineString],
) -> tuple[list[Polygon], list[LineString]]:
    polys: list[Polygon] = []
    remain: list[LineString] = []
    for ls in linestrings:
        coords = list(ls.coords)
        if len(coords) < 4:
            remain.append(ls)
            continue
        s, e = coords[0][:2], coords[-1][:2]
        if abs(s[0] - e[0]) < 1e-6 and abs(s[1] - e[1]) < 1e-6:
            try:
                p = Polygon([(c[0], c[1]) for c in coords])
                if p.is_valid and not p.is_empty:
                    polys.append(p)
                else:
                    remain.append(ls)
            except Exception:
                remain.append(ls)
        else:
            remain.append(ls)
    return polys, remain


# ─── Polygon 전처리 (preprocessing 알고리즘) ──────────────────
def preprocess_polygons(polys: list[Polygon]) -> list:
    if not polys:
        return []
    gdf = gpd.GeoDataFrame(geometry=list(polys))
    # force_2d (Z 제거)
    gdf.geometry = shapely.force_2d(gdf.geometry.values)
    # make_valid (위상 수정)
    gdf.geometry = gdf.geometry.make_valid()

    # GeometryCollection 내 Polygon part 추출
    def extract(g):
        if g is None or g.is_empty:
            return None
        if g.geom_type == 'GeometryCollection':
            parts = [p for p in g.geoms if p.geom_type in ('Polygon', 'MultiPolygon')]
            if parts:
                u = unary_union(parts)
                return u if not u.is_empty else None
            return None
        return g

    gdf.geometry = gdf.geometry.apply(extract)
    gdf = gdf[~gdf.geometry.isna()].copy()
    mask = gdf.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])
    gdf = gdf[mask].copy()
    return list(gdf.geometry.values)


# ─── 메인 변환 ────────────────────────────────────────────────
def convert(
    dxf_path: str,
    out_dir: str,
    epsg: int = DEFAULT_EPSG,
    layer_prefix: str = LAYER_PREFIX,
    xy_tol: float = DEFAULT_XY_TOLERANCE,
    z_tol: float = DEFAULT_Z_TOLERANCE,
    to_polygon: bool = True,
) -> dict[str, int]:
    t0 = time.time()
    by_cat = extract_features(dxf_path, layer_prefix=layer_prefix)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(dxf_path))[0]
    crs = f"EPSG:{epsg}"
    summary: dict[str, int] = {}

    for cat, lines in by_cat.items():
        log(f"[{cat}] entity {len(lines)}건")
        merged = merge_open_lines(lines, xy_tol=xy_tol, z_tol=z_tol)

        polys: list = []
        remain_lines = merged
        if to_polygon:
            polys_raw, remain_lines = closed_lines_to_polygons(merged)
            polys = preprocess_polygons(polys_raw)
            log(f"  → polygons={len(polys)}, residual lines={len(remain_lines)}")

        if polys:
            out = os.path.join(out_dir, f"{base}_{cat}_polygon.shp")
            gpd.GeoDataFrame(geometry=polys, crs=crs).to_file(
                out, driver='ESRI Shapefile', encoding='utf-8'
            )
            log(f"  saved: {out}")
            summary[f"{cat}_polygon"] = len(polys)

        if remain_lines:
            out = os.path.join(out_dir, f"{base}_{cat}_line.shp")
            gpd.GeoDataFrame(geometry=remain_lines, crs=crs).to_file(
                out, driver='ESRI Shapefile', encoding='utf-8'
            )
            log(f"  saved: {out}")
            summary[f"{cat}_line"] = len(remain_lines)

    log(f"=== 완료 ({time.time() - t0:.1f}s) ===")
    log(f"  요약: {summary}")
    return summary


def main():
    ap = argparse.ArgumentParser(
        description='DXF → SHP 통합 변환 (건물 레이어 + 색상 분류 + 라인 체인 합치기)'
    )
    ap.add_argument('--input', required=True, help='입력 DXF 경로')
    ap.add_argument('--out-dir', required=True, help='출력 SHP 디렉터리')
    ap.add_argument('--epsg', type=int, default=DEFAULT_EPSG, help=f'좌표계 EPSG (기본 {DEFAULT_EPSG})')
    ap.add_argument(
        '--layer-prefix',
        default=LAYER_PREFIX,
        help=f'레이어 prefix (기본 "{LAYER_PREFIX}")',
    )
    ap.add_argument('--xy-tol', type=float, default=DEFAULT_XY_TOLERANCE)
    ap.add_argument('--z-tol', type=float, default=DEFAULT_Z_TOLERANCE)
    ap.add_argument(
        '--no-polygon',
        action='store_true',
        help='Polygon 변환 비활성 (LineString만 저장)',
    )
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"[fail] input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    convert(
        args.input,
        args.out_dir,
        epsg=args.epsg,
        layer_prefix=args.layer_prefix,
        xy_tol=args.xy_tol,
        z_tol=args.z_tol,
        to_polygon=not args.no_polygon,
    )


if __name__ == '__main__':
    main()
