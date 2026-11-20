"""Vérification géométrique des découpages : chaque circonscription est-elle d'un seul tenant sur la carte ?

Pour chaque circonscription (réelle et optimisée), on fusionne les contours de ses bureaux, on bouche les
interstices (fermeture de 30 m) et on compte les morceaux séparés de plus de MIN_KM2.
Les îles sont signalées à part : un morceau détaché est une « île » s'il ne touche aucun autre bureau
du département (mer, étang...), quelle que soit la circonscription.

Usage : python src/verify_geometry.py max_eps20
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).parent))
from common import PROC, RESULTS, TARGETS  # noqa: E402

MIN_KM2 = 0.05
CLOSE = 30  # m


def parts_of(geom):
    geom = geom.buffer(CLOSE).buffer(-CLOSE)
    ps = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    return sorted([p for p in ps if p.area >= MIN_KM2 * 1e6], key=lambda p: -p.area)


def check(gd: gpd.GeoDataFrame, labels: pd.Series, dep_shape, layer: str) -> list[dict]:
    rows = []
    for d, sub in gd.assign(_d=labels.values).groupby("_d"):
        ps = parts_of(unary_union(sub.geometry.values))
        main, extra = ps[0], ps[1:]
        # un morceau est une île s'il ne touche pas le reste du département
        others = dep_shape.difference(unary_union(sub.geometry.values).buffer(CLOSE))
        n_island = sum(1 for p in extra if not p.buffer(CLOSE * 2).intersects(others))
        stray = [p for p in extra if p.buffer(CLOSE * 2).intersects(others)]
        rows.append(dict(layer=layer, district=str(d), n_parts=len(ps), n_islands=n_island,
                         n_stray=len(stray), stray_km2=round(sum(p.area for p in stray) / 1e6, 2),
                         main_km2=round(main.area / 1e6, 1)))
    return rows


def main(run: str):
    U = gpd.read_parquet(PROC / "units.parquet")
    plans = RESULTS / run / "plans"
    rows = []
    for dep, gd in U.groupby("dep"):
        dep_shape = unary_union(gd.geometry.values).buffer(CLOSE).buffer(-CLOSE)
        for r in check(gd, gd.circo, dep_shape, "real"):
            rows.append(dict(dep=dep, **r))
        for tg in TARGETS:
            a = pd.Series(json.load(open(plans / f"{dep}_{tg}.json")))
            a.index = a.index.astype(int)
            for r in check(gd, gd.uid.map(a), dep_shape, tg):
                rows.append(dict(dep=dep, **r))
        print(dep, flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(RESULTS / run / "verification_geometrie.csv", index=False)
    s = R.assign(bad=R.n_stray > 0).groupby("layer").agg(circos=("bad", "size"), eclatees=("bad", "sum"),
                                                           morceaux_km2=("stray_km2", "sum"))
    print(s)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "max_eps20")
