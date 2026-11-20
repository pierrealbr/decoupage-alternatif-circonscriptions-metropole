"""Prépare les unités géographiques (bureaux de vote) pour le redécoupage.

Entrées (data/raw) :
  - leg2024_t1_bv.csv           résultats définitifs 1er tour législatives 2024 par bureau
  - leg2024_t1_candidats.csv    candidats 1er tour (pour retrouver la circonscription de chaque bureau)
  - contours_bv_2024.json       contours approximatifs des bureaux (méthode INSEE, topojson)
  - communes-2024-50m.geojson.gz contours des communes 2024 (repli)

Sorties (data/processed) :
  - bureaux.parquet   une ligne par bureau de vote du fichier de résultats (France métropolitaine),
                      avec la circonscription réelle et les voix par nuance (sert au calcul de référence)
  - units.parquet     unités géographiques (GeoParquet, Lambert 93) avec voix par nuance
  - edges.parquet     arêtes d'adjacence entre unités d'un même département

Règles de raccordement résultats <-> contours :
  1. clé (code commune, code bureau normalisé) identique -> unité = le bureau.
  2. dans une commune, les bureaux de résultats non raccordés forment un « reliquat » :
     - s'il reste des contours non raccordés dans la commune, le reliquat est réparti
       à parts égales entre ces contours ;
     - sinon il est réparti au prorata des inscrits des bureaux raccordés de la commune.
  3. commune sans aucun contour -> unité = la commune entière (contour communal 2024).
"""
from __future__ import annotations

import gzip
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

DEPTS = [f"{i:02d}" for i in range(1, 96) if i != 20] + ["2A", "2B"]
NUANCES = ["RN", "UG", "ENS", "LR", "UXD", "DVD", "DVG", "DVC", "EXG", "REG", "REC", "HOR",
           "ECO", "UDI", "DIV", "DSV", "EXD", "SOC", "RDG", "FI", "COM", "VEC"]
CRS = "EPSG:2154"
TOL = 5                 # m, tolérance d'adjacence
MIN_SHARED_AREA = 300   # m², recouvrement des contours élargis de TOL (≈ 30 m de frontière commune)


def norm_bv(s) -> str:
    s = str(s).strip()
    return (s.lstrip("0") or "0") if s.isdigit() else s.upper()


def norm_dep(s: str) -> str:
    s = str(s).strip()
    return s.zfill(2) if s.isdigit() else s


def load_results() -> pd.DataFrame:
    df = pd.read_csv(RAW / "leg2024_t1_bv.csv", sep=";", dtype=str)
    df["dep"] = df["Code département"].map(norm_dep)
    df = df[df.dep.isin(DEPTS)].copy()
    df["com"] = df["Code commune"].str.zfill(5)
    df["bv"] = df["Code BV"].map(norm_bv)
    df["key"] = df.com + "_" + df.bv

    # circonscription : via le nom du candidat en panneau 1 (unique dans le département)
    cand = pd.read_csv(RAW / "leg2024_t1_candidats.csv", sep=";", dtype=str)
    cand["dep"] = cand["Code département"].map(norm_dep)
    cand["circo"] = cand["Code circonscription"].str[-2:]
    cand = cand[cand.dep.isin(DEPTS)]
    ckey = cand.dep + "|" + cand["Nom du candidat"].str.upper() + "|" + cand["Prénom du candidat"].str.upper()
    cmap = pd.Series(cand.circo.values, index=ckey)
    cmap = cmap[~cmap.index.duplicated(keep=False)]
    circos = []
    nposts = sum(1 for c in df.columns if c.startswith("Nom candidat "))
    for _, r in df.iterrows():
        found = None
        for i in range(1, nposts + 1):
            nom = r.get(f"Nom candidat {i}")
            if pd.isna(nom):
                break
            k = f"{r.dep}|{str(nom).upper()}|{str(r[f'Prénom candidat {i}']).upper()}"
            if k in cmap.index:
                found = cmap[k]
                break
        circos.append(found)
    df["circo"] = circos
    assert df.circo.notna().all(), df[df.circo.isna()][["dep", "com", "bv"]]

    out = df[["dep", "com", "Libellé commune", "bv", "key", "circo"]].rename(columns={"Libellé commune": "nom_com"})
    for c in ["Inscrits", "Votants", "Exprimés"]:
        out[c.lower().replace("é", "e")] = pd.to_numeric(df[c]).astype(int)
    votes = pd.DataFrame(0, index=df.index, columns=NUANCES, dtype=np.int64)
    for i in range(1, nposts + 1):
        n = df[f"Nuance candidat {i}"]
        v = pd.to_numeric(df[f"Voix {i}"], errors="coerce").fillna(0).astype(np.int64)
        for nu in NUANCES:
            votes[nu] += np.where(n == nu, v, 0)
    unknown = set(pd.unique(df[[f"Nuance candidat {i}" for i in range(1, nposts + 1)]].values.ravel())) - set(NUANCES) - {np.nan}
    unknown = {u for u in unknown if isinstance(u, str)}
    assert not unknown, unknown
    out = pd.concat([out, votes], axis=1)
    # quelques clés dupliquées dans le fichier source -> on somme
    num = ["inscrits", "votants", "exprimes"] + NUANCES
    out = (out.groupby("key", as_index=False)
              .agg({**{c: "first" for c in ["dep", "com", "nom_com", "bv", "circo"]}, **{c: "sum" for c in num}}))
    return out


def load_contours(communes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    g = gpd.read_file(RAW / "contours_bv_2024.json").set_crs(4326).to_crs(CRS)
    g = g[g.codeDepartement.isin(DEPTS) & g.codeBureauVote.notna()].copy()
    g["com"] = g.codeCommune
    g["bv"] = g.codeBureauVote.map(norm_bv)
    g["geometry"] = g.geometry.make_valid()
    return g[["com", "bv", "geometry"]]


def clean_geometries(U: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Rend chaque unité d'un seul tenant (sauf morceaux isolés : îles).

    Les contours INSEE découpent parfois un bureau en plusieurs morceaux disjoints. On garde le morceau
    principal ; chaque morceau secondaire est rattaché (géométrie seulement, les voix ne bougent pas) au
    bureau voisin avec lequel il partage le plus de frontière. Sans voisin (île), il reste à son bureau.
    """
    out = []
    n_moved = 0
    for dep, gd in U.groupby("dep"):
        gd = gd.copy()
        geom = gd.geometry.values
        closed = shapely.buffer(geom, TOL)
        keep, minors = [], []
        for uid, g, c in zip(gd.uid.values, geom, closed):
            cps = list(c.geoms) if c.geom_type == "MultiPolygon" else [c]
            if len(cps) == 1:
                keep.append(g)
                continue
            main = max(cps, key=lambda q: q.area)
            keep.append(g.intersection(main))
            rest = g.difference(main)
            for q in (list(rest.geoms) if hasattr(rest, "geoms") else [rest]):
                if q.geom_type == "Polygon" and q.area > 1:
                    minors.append((uid, q))
        gd["geometry"] = keep
        if minors:
            M = gpd.GeoDataFrame({"owner": [m[0] for m in minors]}, geometry=[m[1] for m in minors], crs=U.crs)
            Mb = M.copy()
            Mb["geometry"] = Mb.geometry.buffer(TOL)
            K = gd[["uid", "geometry"]].copy()
            j = gpd.sjoin(Mb, K, predicate="intersects")
            j = j[j.uid != j.owner]
            if len(j):
                inter = shapely.intersection(Mb.geometry.loc[j.index].values, K.set_index("uid").geometry.loc[j.uid].values)
                j = j.assign(a=shapely.area(inter)).sort_values("a", ascending=False)
                best = j[~j.index.duplicated()]
                target = pd.Series(best.uid.values, index=best.index)
            else:
                target = pd.Series(dtype=int)
            M["to"] = target.reindex(M.index)
            n_moved += int(M.to.notna().sum())
            M["to"] = M.to.fillna(M.owner).astype(int)
            add = M.dissolve(by="to").geometry
            g2 = gd.set_index("uid").geometry
            for uid, piece in add.items():
                g2.loc[uid] = shapely.union(g2.loc[uid], piece)
            gd["geometry"] = g2.loc[gd.uid].values
        out.append(gd)
    R = gpd.GeoDataFrame(pd.concat(out).sort_values("uid"), crs=U.crs).reset_index(drop=True)
    R["geometry"] = R.geometry.make_valid()
    print("morceaux secondaires rattachés à un voisin :", n_moved,
          "| unités encore en plusieurs morceaux (îles) :", int((R.geom_type == "MultiPolygon").sum()))
    return R


def main():
    res = load_results()
    print("bureaux (résultats)", len(res), "inscrits", res.inscrits.sum())
    res.drop(columns=[]).to_parquet(OUT / "bureaux.parquet")

    with gzip.open(RAW / "communes-2024-50m.geojson.gz") as f:
        communes = gpd.read_file(f).to_crs(CRS)
    communes = communes.rename(columns={"code": "com"})[["com", "nom", "departement", "geometry"]]

    con = load_contours(communes)
    # contours de communes absentes des résultats (communes fusionnées depuis) -> commune 2024 qui les contient
    res_coms = set(res.com)
    orphan = ~con.com.isin(res_coms)
    if orphan.any():
        pts = con.loc[orphan].copy()
        pts["geometry"] = pts.geometry.representative_point()
        j = gpd.sjoin(pts, communes[["com", "geometry"]].rename(columns={"com": "com24"}), predicate="within", how="left")
        j = j[~j.index.duplicated()]
        con.loc[orphan, "com"] = j["com24"].values
        con.loc[orphan, "bv"] = "ORPHAN"  # forcent le passage par le reliquat
        print("contours réaffectés à une commune 2024 :", int(orphan.sum()), "dont sans commune :", int(j.com24.isna().sum()))
        con = con[con.com.notna()]
    con["key"] = con.com + "_" + con.bv
    con = con.dissolve(by="key", as_index=False, aggfunc="first")

    num = ["inscrits", "votants", "exprimes"] + NUANCES
    matched_keys = set(res.key) & set(con.key)
    units = []
    # 1. bureaux raccordés
    m = con[con.key.isin(matched_keys)].merge(res[["key", "dep", "circo"] + num], on="key")
    m["kind"] = "bureau"
    units.append(m)

    # 2. reliquats par commune
    res_un = res[~res.key.isin(matched_keys)]
    con_un = con[~con.key.isin(matched_keys)]
    stats = {"reliquat_contours": 0, "reliquat_prorata": 0, "commune_entiere": 0, "contours_vides": 0}
    extra_rows = []
    for com, grp in res_un.groupby("com"):
        polys = con_un[con_un.com == com]
        tot = grp[num].sum()
        circo = grp.groupby("circo").inscrits.sum().idxmax()
        dep = grp.dep.iloc[0]
        if len(polys):
            share = tot / len(polys)
            p = polys.copy()
            for c in num:
                p[c] = share[c]
            p["dep"], p["circo"], p["kind"] = dep, circo, "reliquat"
            units.append(p)
            stats["reliquat_contours"] += len(grp)
        elif (m.com == com).any():
            extra_rows.append((com, tot))
            stats["reliquat_prorata"] += len(grp)
        else:
            cg = communes[communes.com == com]
            if len(cg) != 1:
                # fraction de commune nouvelle rattachée électoralement à un autre département :
                # pas de géométrie disponible, on l'écarte (quelques centaines d'inscrits)
                print(f"  écarté (sans géométrie) : {com} {grp.nom_com.iloc[0]} ({int(tot.inscrits)} inscrits)")
                stats.setdefault("ecartes_inscrits", 0)
                stats["ecartes_inscrits"] += int(tot.inscrits)
                continue
            p = gpd.GeoDataFrame({"com": [com], "bv": ["COMMUNE"], "key": [com + "_COMMUNE"], "dep": [dep],
                                  "circo": [circo], "kind": ["commune"]}, geometry=[cg.geometry.iloc[0]], crs=CRS)
            for c in num:
                p[c] = tot[c]
            units.append(p)
            stats["commune_entiere"] += len(grp)
    # contours non raccordés d'une commune sans reliquat : unités vides (gardées pour la contiguïté)
    used = set(res_un.com)
    empty = con_un[~con_un.com.isin(used)].copy()
    if len(empty):
        dep_of_com = res.groupby("com").dep.first()
        empty["dep"] = empty.com.map(dep_of_com)
        empty = empty[empty.dep.notna()]
        circo_of_com = res.groupby("com").circo.agg(lambda s: s.mode().iloc[0])
        empty["circo"] = empty.com.map(circo_of_com)
        for c in num:
            empty[c] = 0.0
        empty["kind"] = "vide"
        units.append(empty)
        stats["contours_vides"] = len(empty)

    U = gpd.GeoDataFrame(pd.concat(units, ignore_index=True), crs=CRS)
    for c in num:
        U[c] = U[c].astype(float)
    for com, tot in extra_rows:
        idx = U.index[(U.com == com) & (U.kind == "bureau")]
        w = U.loc[idx, "inscrits"] / U.loc[idx, "inscrits"].sum()
        for c in num:
            U.loc[idx, c] += w * tot[c]
    print("raccordement :", stats)
    tot_res, tot_u = res[num].sum(), U[num].sum()
    print("contrôle des totaux (unités - résultats) :", (tot_u - tot_res).abs().max())
    U = U[["dep", "com", "bv", "key", "kind", "circo"] + num + ["geometry"]].reset_index(drop=True)
    U["uid"] = np.arange(len(U))
    U = clean_geometries(U)
    U.to_parquet(OUT / "units.parquet")
    print("unités :", len(U), U.kind.value_counts().to_dict())

    # 3. adjacence : frontière commune d'au moins ~30 m (un simple coin ne suffit pas),
    #    avec une tolérance de 5 m pour les imprécisions de contours
    edges = []
    for dep, gd in U.groupby("dep"):
        b = gd[["uid", "geometry"]].copy()
        b["geometry"] = b.geometry.buffer(TOL)
        j = gpd.sjoin(b, b, predicate="intersects")
        j = j[j.uid_left < j.uid_right]
        geo = b.set_index("uid").geometry
        inter = shapely.intersection(geo.loc[j.uid_left].values, geo.loc[j.uid_right].values)
        j = j[shapely.area(inter) >= MIN_SHARED_AREA]
        edges.append(pd.DataFrame({"dep": dep, "u": j.uid_left.values, "v": j.uid_right.values}))
    E = pd.concat(edges, ignore_index=True).drop_duplicates()
    E.to_parquet(OUT / "edges.parquet")
    print("arêtes :", len(E))


if __name__ == "__main__":
    main()
