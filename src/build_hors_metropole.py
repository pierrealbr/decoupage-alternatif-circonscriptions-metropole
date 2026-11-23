"""Les 38 circonscriptions non redécoupées (27 d'outre-mer, 11 des Français de l'étranger), avec leur résultat réel 2024,
et la proportionnelle intégrale sur les 577 sièges.

- 1er tour : force arrivée en tête ; 2nd tour : force du député réellement élu (au 1er ou au 2nd tour).
- Proportionnelle : voix du 1er tour dans les 577 circonscriptions, sans seuil, plus fort reste.

Usage : python src/build_hors_metropole.py   -> site/hors_metropole.json, site/proportionnelle.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from second_tour import FORCE_OF  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW, SITE = ROOT / "data" / "raw", ROOT / "site"
GROUP = {**FORCE_OF, "ECO": "ECO"}
METRO_EXCLUDE = lambda dep: len(dep) == 3 or dep in ("ZX", "ZZ")  # 971-988, Saint-Martin/Saint-Barth (ZX), étranger (ZZ)
NAMES = {"ZZ": "Français établis hors de France", "ZX": "Saint-Barthélemy et Saint-Martin"}


def long_format(df: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par candidat : circo, nuance, voix, élu."""
    rows = []
    i = 1
    while f"Nuance candidat {i}" in df.columns:
        part = df[["Code département", "Libellé département", "Code circonscription législative",
                   "Libellé circonscription législative", f"Nuance candidat {i}", f"Voix {i}", f"Elu {i}"]].copy()
        part.columns = ["dep", "dep_name", "circo", "circo_name", "nuance", "voix", "elu"]
        rows.append(part.dropna(subset=["nuance"]))
        i += 1
    out = pd.concat(rows)
    out["voix"] = pd.to_numeric(out.voix, errors="coerce").fillna(0)
    out["force"] = out.nuance.map(GROUP).fillna("OTHER")
    out["elu"] = out.elu.fillna("").str.strip().str.lower().str.startswith("élu")
    return out


def main():
    t1 = long_format(pd.read_csv(RAW / "leg2024_t1_circo.csv", sep=";", dtype=str))
    t2 = long_format(pd.read_csv(RAW / "leg2024_t2_circo.csv", sep=";", dtype=str))

    # proportionnelle intégrale sur 577 sièges
    votes = t1.groupby("force").voix.sum().sort_values(ascending=False)
    tot = float(votes.sum()); quota = votes / tot * 577; seats = quota.astype(int)
    for f in (quota - seats).sort_values(ascending=False).index[: 577 - int(seats.sum())]:
        seats[f] += 1
    prop = dict(seats_total=577, exprimes=int(tot),
                forces={f: dict(voix=int(votes[f]), part=round(100 * votes[f] / tot, 2), sieges=int(seats[f])) for f in votes.index})
    json.dump(prop, open(SITE / "proportionnelle.json", "w"), ensure_ascii=False, indent=1)

    # 38 circonscriptions hors métropole
    h1 = t1[t1.dep.map(METRO_EXCLUDE)]
    circos = []
    for circo, g in h1.groupby("circo", sort=True):
        by_force = g.groupby("force").voix.sum().sort_values(ascending=False)
        e1 = g[g.elu]
        elu = e1 if len(e1) else t2[(t2.circo == circo) & t2.elu]
        assert len(elu) == 1, circo
        dep = g.dep.iloc[0]
        circos.append(dict(dep=dep, dep_name=NAMES.get(dep, g.dep_name.iloc[0]), circo=circo, name=g.circo_name.iloc[0],
                           t1=by_force.index[0], t1_share=round(100 * by_force.iloc[0] / by_force.sum(), 1),
                           elu=elu.force.iloc[0], elu_nuance=elu.nuance.iloc[0], elu_t1=bool(len(e1))))
    assert len(circos) == 38, len(circos)
    cnt = lambda k: pd.Series([c[k] for c in circos]).value_counts().to_dict()
    out = dict(n=len(circos), t1=cnt("t1"), t2=cnt("elu"), circos=circos)
    json.dump(out, open(SITE / "hors_metropole.json", "w"), ensure_ascii=False, indent=1)
    print("1er tour, en tête :", out["t1"])
    print("élus             :", out["t2"])
    print("proportionnelle  :", {f: v["sieges"] for f, v in prop["forces"].items()})


if __name__ == "__main__":
    main()
