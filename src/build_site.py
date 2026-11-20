"""Génère les données de la page locale site/ : un fichier JSON par département avec
le découpage réel et les découpages optimisés (tracés SVG simplifiés), plus un index.

Usage : python src/build_site.py max_eps20
Puis : python -m http.server -d site 8765
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon

sys.path.insert(0, str(Path(__file__).parent))
from common import PROC, RESULTS, ROOT, TARGETS, add_forces, leaders  # noqa: E402
from optimize import NUANCES  # noqa: E402

SITE = ROOT / "site"
DATA = SITE / "data"
ARR_CITIES = {"75056": "Paris", "13055": "Marseille", "69123": "Lyon"}


def ordinal(n: int) -> str:
    return "1re" if n == 1 else f"{n}e"


def to_path(geom, tf) -> str:
    polys = [geom] if isinstance(geom, Polygon) else list(getattr(geom, "geoms", []))
    out = []
    for p in polys:
        if not isinstance(p, Polygon) or p.is_empty:
            continue
        for ring in [p.exterior, *p.interiors]:
            xy = np.asarray(ring.coords)
            if len(xy) < 4:
                continue
            px, py = tf(xy[:, 0], xy[:, 1])
            pts = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(px, py))
            out.append(f"M{pts}Z")
    return "".join(out)


T2 = None  # modèle de second tour (chargé si --t2)


def t2_fields(r: pd.Series) -> dict:
    """Vainqueur probable au 2nd tour, probabilités et scénario le plus probable (modèle second_tour)."""
    import second_tour as st
    t1 = {f: 0.0 for f in st.FORCES}
    for n, f in st.FORCE_OF.items():
        if n in r.index:
            t1[f] += float(r[n])
    t1["inscrits"] = float(r.inscrits)
    t1["exprimes"] = sum(t1[f] for f in st.FORCES)
    t1["NONEXP"] = max(t1["inscrits"] - t1["exprimes"], 0.0)
    dist = st.win_distribution(T2["models"], T2["withdrawal"], T2["sigma"], t1)
    sc = max(st.scenarios(t1, T2["withdrawal"]), key=lambda x: x[2])
    out = dict(p={f: round(v, 3) for f, v in sorted(dist.items(), key=lambda kv: -kv[1]) if v >= 0.005})
    out["w2"] = max(dist, key=dist.get)
    out["pw"] = round(dist[out["w2"]], 3)
    if sc[0] == "T1":
        out["sc"] = dict(t1=True, fin=[sc[1]], pr=1.0)
    else:
        v, _ = st.predict_votes(T2["models"], t1, sc[1])
        tot = max(sum(v.values()), 1)
        out["sc"] = dict(t1=False, fin=list(sc[1]), pr=round(sc[2], 3),
                         sh={f: round(100 * x / tot, 1) for f, x in sorted(v.items(), key=lambda kv: -kv[1])})
    return out


def district_stats(V: pd.DataFrame, forces: list[str], ideal: float) -> list[dict]:
    fcols = [f"F_{f}" for f in forces]
    lead = leaders(V[fcols].values, forces)
    rows = []
    for (idx, r), ld in zip(V.iterrows(), lead):
        expr = max(r[fcols].sum(), 1)
        shares = {f: round(100 * r[f"F_{f}"] / expr, 1) for f in forces if r[f"F_{f}"] / expr >= 0.02}
        shares = dict(sorted(shares.items(), key=lambda kv: -kv[1]))
        top = sorted(r[fcols].values, reverse=True)
        row = dict(id=idx, leader=ld, shares=shares, inscrits=int(round(r.inscrits)),
                   ecart=round(100 * (r.inscrits / ideal - 1), 1),
                   marge=round(100 * (top[0] - top[1]) / expr, 1))
        if T2 is not None:
            row.update(t2_fields(r))
        rows.append(row)
    return rows


def main(run: str, t2: bool = False):
    global DATA, T2
    if t2:
        import second_tour as st
        T2 = st.load_full_model()
        DATA = SITE / "data_t2"
    DATA.mkdir(parents=True, exist_ok=True)
    plans_dir = RESULTS / run / "plans"

    U = gpd.read_parquet(PROC / "units.parquet")
    U, forces_all = add_forces(U, NUANCES)
    # circonscription réelle de chaque unité : celle portée par le contour (législatives 2022, même découpage)
    t = json.load(open(ROOT / "data/raw/contours_bv_2024.json"))
    from prepare import norm_bv
    cc = {}
    for g in t["objects"]["bureaux_vote"]["geometries"]:
        p = g["properties"]
        if p.get("codeBureauVote") and p.get("codeCirco"):
            cc[f"{p['codeCommune']}_{norm_bv(p['codeBureauVote'])}"] = p["codeCirco"]
    U["circo_geo"] = U.key.map(cc).fillna(U.circo)

    b = pd.read_parquet(PROC / "bureaux.parquet")
    real = b.groupby(["dep", "circo"])[["inscrits"] + NUANCES].sum().reset_index()
    real, _ = add_forces(real, NUANCES)
    cand = pd.read_csv(ROOT / "data/raw/leg2024_t1_candidats.csv", sep=";", dtype=str)
    cand["dep"] = cand["Code département"].map(lambda s: s.zfill(2) if s.isdigit() else s)
    dep_names = cand.groupby("dep")["Département"].first().to_dict()

    index = []
    NATIONAL.clear()
    for dep, gd in U.groupby("dep"):
        opt_all = {}
        k = int(real[real.dep == dep].shape[0])
        fcols_all = [f"F_{f}" for f in forces_all]
        tot = gd[fcols_all].sum()
        forces = [f for f in forces_all if f in TARGETS or tot[f"F_{f}"] > 0]
        minx, miny, maxx, maxy = gd.total_bounds
        w, h = maxx - minx, maxy - miny
        s = 1000 / max(w, h)
        W, H = round(w * s, 1), round(h * s, 1)

        def tf(x, y):
            return (x - minx) * s, (maxy - y) * s

        tol = max(np.hypot(w, h) / 1800, 8)
        ideal = gd.inscrits.sum() / k

        def build_map(labels: pd.Series, stats_df: pd.DataFrame, name_of, layer: str) -> list[dict]:
            geo = gd.assign(_d=labels.values).dissolve(by="_d").geometry
            # fermeture morphologique : bouche les interstices entre contours de bureaux
            geo = geo.buffer(max(tol, 15), join_style="mitre").buffer(-max(tol, 15), join_style="mitre")
            closed = geo
            geo = geo.simplify(tol, preserve_topology=True)
            stats = {r["id"]: r for r in district_stats(stats_df, forces, ideal)}
            out = []
            for d, g in geo.items():
                r = dict(stats[d])
                r["id"] = name_of(d)
                nat = dict(dep=dep, id=r["id"], leader=r["leader"], marge=r["marge"], inscrits=r["inscrits"],
                           shares=dict(list(r["shares"].items())[:4]), geom=closed[d])
                if T2 is not None:
                    nat.update(p=r["p"], w2=r["w2"], pw=r["pw"], sc=r["sc"])
                NATIONAL.setdefault(layer, []).append(nat)
                r["path"] = to_path(g, tf)
                c = g.representative_point()
                cx, cy = tf(np.array([c.x]), np.array([c.y]))
                r["cx"], r["cy"] = round(float(cx[0]), 1), round(float(cy[0]), 1)
                out.append(r)
            return out

        # découpage réel (géométrie : contours ; chiffres : résultats officiels agrégés, exacts)
        rs = real[real.dep == dep].set_index("circo")
        real_map = build_map(gd.circo_geo, rs, lambda d: f"{ordinal(int(d))} circ.", "real")
        def all_counts(districts):
            """Sièges (attendus au 2nd tour, ou circonscriptions en tête au 1er) de chaque force."""
            out = {}
            for r in districts:
                if T2 is not None:
                    for f, v in r["p"].items():
                        out[f] = out.get(f, 0) + v
                else:
                    out[r["leader"]] = out.get(r["leader"], 0) + 1
            return {f: round(v, 2) for f, v in out.items()}

        real_all = all_counts(real_map)
        real_counts = {tg: real_all.get(tg, 0) for tg in TARGETS}

        plans, wins = {}, {}
        for tg in TARGETS:
            f = plans_dir / f"{dep}_{tg}.json"
            if not f.exists():
                continue
            a = pd.Series(json.load(open(f)))
            a.index = a.index.astype(int)
            lab = gd.uid.map(a)
            if lab.isna().any():
                lab = lab.fillna(-1)
            V = gd.assign(_d=lab.values).groupby("_d")[["inscrits"] + NUANCES + [f"F_{x}" for x in forces]].sum()
            # numérotation des nouvelles circonscriptions : du nord-ouest au sud-est
            cent = gd.assign(_d=lab.values).dissolve(by="_d").geometry.centroid
            order = (-cent.y + cent.x).sort_values().index
            num = {d: i + 1 for i, d in enumerate(order)}
            plans[tg] = build_map(lab, V, lambda d: f"{ordinal(num[d])}′ circ.", tg)
            opt_all[tg] = all_counts(plans[tg])
            wins[tg] = opt_all[tg].get(tg, 0)

        # villes principales (par nombre d'inscrits) ; arrondissements pour Paris, Lyon, Marseille
        cities = []
        ins = gd.groupby("com").inscrits.sum().sort_values(ascending=False)
        names = b[b.dep == dep].groupby("com").nom_com.first()
        big = [c for c in ins.index[:7]]
        for com in big:
            sub = gd[gd.com == com]
            if com in ARR_CITIES and dep == "75":
                arr = pd.to_numeric(sub.bv, errors="coerce") // 100
                for n, sg in sub.groupby(arr):
                    if n != n or n < 1:
                        continue
                    p = sg.geometry.union_all().representative_point()
                    x, y = tf(np.array([p.x]), np.array([p.y]))
                    cities.append(dict(name=f"{int(n)}{'er' if n == 1 else 'e'}", x=round(float(x[0]), 1),
                                       y=round(float(y[0]), 1), rank=2, inscrits=int(sg.inscrits.sum())))
                if len(big) > 1 and dep != "75":
                    p = sub.geometry.union_all().representative_point()
                    x, y = tf(np.array([p.x]), np.array([p.y]))
                    cities.append(dict(name=ARR_CITIES[com], x=round(float(x[0]), 1), y=round(float(y[0]), 1),
                                       rank=0, inscrits=int(ins[com])))
                continue
            p = sub.geometry.union_all().representative_point()
            x, y = tf(np.array([p.x]), np.array([p.y]))
            nm = str(names.get(com, com)).strip()
            cities.append(dict(name=nm, x=round(float(x[0]), 1), y=round(float(y[0]), 1),
                               rank=0 if com == big[0] else 1, inscrits=int(ins[com])))

        doc = dict(dep=dep, name=dep_names.get(dep, dep), k=k, width=W, height=H, forces=forces,
                   real=real_map, plans=plans, cities=cities)
        (DATA / f"{dep}.json").write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")))
        index.append(dict(dep=dep, name=dep_names.get(dep, dep), k=k,
                          real={tg: real_counts.get(tg, 0) for tg in TARGETS},
                          opt={tg: wins.get(tg) for tg in TARGETS}, real_all=real_all, opt_all=opt_all))
        print(dep, len(plans), "plans", flush=True)

    (DATA / "index.json").write_text(json.dumps(dict(run=run, mode="t2" if T2 is not None else "t1",
                                                     departements=index), ensure_ascii=False, default=float))
    build_france(U, dep_names)


INSET_DEPS = ["75", "92", "93", "94"]


def build_france(U: gpd.GeoDataFrame, dep_names: dict):
    """Carte de France : toutes les circonscriptions (réel + chaque force), avec un encart Paris-petite couronne."""
    deps = U[["dep", "geometry"]].dissolve(by="dep").geometry
    deps = deps.buffer(300, join_style="mitre").buffer(-300, join_style="mitre")

    def frame(geoms, tol, size=1000):
        minx, miny, maxx, maxy = gpd.GeoSeries(list(geoms)).total_bounds
        w, h = maxx - minx, maxy - miny
        s = size / max(w, h)
        return dict(minx=minx, maxy=maxy, s=s, W=round(w * s, 1), H=round(h * s, 1), tol=tol)

    def tfn(fr):
        return lambda x, y: ((x - fr["minx"]) * fr["s"], (fr["maxy"] - y) * fr["s"])

    main = frame(deps.values, 900)
    inset = frame(deps.loc[INSET_DEPS].values, 80, size=1000)
    out = dict(main=dict(width=main["W"], height=main["H"]), inset=dict(width=inset["W"], height=inset["H"], deps=INSET_DEPS),
               departements=[], insetDepartements=[], layers={})
    tf_main, tf_inset = tfn(main), tfn(inset)
    for d, g in deps.items():
        out["departements"].append(dict(dep=d, name=dep_names.get(d, d),
                                        path=to_path(g.simplify(main["tol"], preserve_topology=True), tf_main)))
        if d in INSET_DEPS:
            out["insetDepartements"].append(dict(dep=d, path=to_path(g.simplify(inset["tol"]), tf_inset)))
    for layer, rows in NATIONAL.items():
        L = []
        for r in rows:
            g = r["geom"]
            item = {k: v for k, v in r.items() if k != "geom"}
            item["path"] = to_path(g.simplify(main["tol"], preserve_topology=True), tf_main)
            if r["dep"] in INSET_DEPS:
                item["ipath"] = to_path(g.simplify(inset["tol"], preserve_topology=True), tf_inset)
            L.append(item)
        out["layers"][layer] = L
    (DATA / "france.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    print("france.json :", {k: len(v) for k, v in out["layers"].items()})


NATIONAL: dict[str, list] = {}


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args[0] if args else "max_eps20", t2="--t2" in sys.argv)
