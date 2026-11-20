"""Scénario « trois blocs à égalité » (2nd tour, sièges attendus).

Dans chaque département, on choisit un des découpages déjà calculés : le réel ou l'optimisé pour NFP, RN, ENS ou LR.
On cherche une combinaison où NFP, RN et ENS ont le même nombre de sièges attendus (écart ≤ TOL), en modifiant
le moins de départements possible par rapport au découpage réel (programme linéaire en nombres entiers).

Usage : python src/build_egalite.py   -> site/data_t2/egalite.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

SITE = Path(__file__).resolve().parent.parent / "site" / "data_t2"
EQ = ["NFP", "RN", "ENS"]
OPTIONS = ["real", "NFP", "RN", "ENS", "LR"]
TOL = 0.5  # écart maximal toléré entre deux des trois forces, en sièges attendus


def main():
    deps = json.load(open(SITE / "index.json"))["departements"]
    opts = []  # (i_dep, option, dict force -> sièges attendus)
    for i, d in enumerate(deps):
        for o in OPTIONS:
            opts.append((i, o, d["real_all"] if o == "real" else d["opt_all"][o]))
    n = len(opts)
    C = np.array([[o[2].get(f, 0.0) for f in EQ] for o in opts])

    A, lo, hi = [], [], []
    for i in range(len(deps)):  # un seul découpage par département
        A.append([1.0 if o[0] == i else 0.0 for o in opts]); lo.append(1); hi.append(1)
    for a in range(3):  # |S_a - S_b| <= TOL
        for b in range(a + 1, 3):
            A.append(C[:, a] - C[:, b]); lo.append(-TOL); hi.append(TOL)
    cost = np.array([0.0 if o[1] == "real" else 1.0 for o in opts])  # nombre de départements modifiés
    res = milp(cost, constraints=LinearConstraint(np.array(A), lo, hi), integrality=np.ones(n),
               bounds=Bounds(0, 1), options={"time_limit": 300})
    if res.x is None:
        raise SystemExit(f"pas de solution : {res.message}")
    x = res.x.round().astype(bool)

    out_deps, real_tot, sc_tot = [], {}, {}
    for (i, o, counts) in (opts[j] for j in np.where(x)[0]):
        d = deps[i]
        for f, v in d["real_all"].items(): real_tot[f] = real_tot.get(f, 0) + v
        for f, v in counts.items(): sc_tot[f] = sc_tot.get(f, 0) + v
        out_deps.append(dict(dep=d["dep"], name=d["name"], k=d["k"], choice=o,
                             real={f: d["real_all"].get(f, 0) for f in EQ + ["LR"]},
                             scen={f: counts.get(f, 0) for f in EQ + ["LR"]}))
    r2 = lambda o: {f: round(v, 2) for f, v in sorted(o.items(), key=lambda kv: -kv[1])}
    doc = dict(tol=TOL, changed=int(sum(d["choice"] != "real" for d in out_deps)),
               real=r2(real_tot), scen=r2(sc_tot), departements=out_deps)
    json.dump(doc, open(SITE / "egalite.json", "w"), ensure_ascii=False, indent=0)
    print("départements modifiés :", doc["changed"], "/", len(deps))
    print("réel     :", {f: doc["real"].get(f) for f in EQ + ["LR"]})
    print("scénario :", {f: doc["scen"].get(f) for f in EQ + ["LR"]})
    from collections import Counter
    print(Counter(d["choice"] for d in out_deps))


if __name__ == "__main__":
    main()
