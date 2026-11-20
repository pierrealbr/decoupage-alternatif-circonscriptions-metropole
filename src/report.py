"""Agrège les plans optimisés, les revérifie et les compare au découpage réel.

Usage : python src/report.py max_eps20 [--eps 0.20]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import PROC, RESULTS, TARGETS, add_forces, leaders  # noqa: E402
from optimize import NUANCES, load_department  # noqa: E402


def baseline() -> pd.DataFrame:
    b = pd.read_parquet(PROC / "bureaux.parquet")
    c = b.groupby(["dep", "circo"])[NUANCES].sum().reset_index()
    c, forces = add_forces(c, NUANCES)
    c["lead"] = leaders(c[[f"F_{f}" for f in forces]].values, forces)
    return c


def check_plan(dep: str, target: str, plan: dict, eps: float):
    G, k, forces = load_department(dep)
    t = forces.index(target)
    assign = {int(u): d for u, d in plan.items()}
    districts = sorted(set(assign.values()))
    assert len(districts) == k, (dep, target, "nombre de circonscriptions")
    pop = {d: 0.0 for d in districts}
    votes = {d: np.zeros(len(forces)) for d in districts}
    for n, data in G.nodes(data=True):
        d = assign[n]
        pop[d] += data["pop"]
        votes[d] += data["votes"]
    ideal = sum(pop.values()) / k
    max_dev = max(abs(p / ideal - 1) for p in pop.values())
    contiguous = all(nx.is_connected(G.subgraph([n for n in G if assign[n] == d])) for d in districts)
    V = np.array([votes[d] for d in districts])
    lead = np.array(forces)[V.argmax(axis=1)]
    return dict(dep=dep, target=target, k=k, wins=int((lead == target).sum()), max_dev=max_dev,
                contiguous=contiguous, valid=contiguous and max_dev <= eps + 1e-9,
                leaders=json.dumps(pd.Series(lead).value_counts().to_dict()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--eps", type=float, default=0.20)
    a = ap.parse_args()
    out = RESULTS / a.run

    base = baseline()
    deps = sorted(base.dep.unique())
    rows = []
    for tg in TARGETS:
        for dep in deps:
            plan = json.load(open(out / "plans" / f"{dep}_{tg}.json"))
            rows.append(check_plan(dep, tg, plan, a.eps))
    R = pd.DataFrame(rows)
    real = base.groupby("dep").lead.apply(lambda s: s.value_counts()).unstack(fill_value=0)
    R["reel"] = [int(real.loc[d, t]) if t in real.columns else 0 for d, t in zip(R.dep, R.target)]
    R["gain"] = R.wins - R.reel
    R.to_csv(out / "par_departement.csv", index=False)

    print(f"Plans invalides : {int((~R.valid).sum())} / {len(R)}  (écart max observé {R.max_dev.max():.3f})")
    tot = R.groupby("target")[["reel", "wins", "k"]].sum().loc[TARGETS]
    tot = tot.rename(columns={"reel": "découpage réel", "wins": "découpage optimisé", "k": "sièges"})
    tot["gain"] = tot["découpage optimisé"] - tot["découpage réel"]
    print(tot.to_string())
    tot.to_csv(out / "synthese.csv")
    print("\nPlus gros gains par force :")
    for tg in TARGETS:
        top = R[R.target == tg].nlargest(5, "gain")[["dep", "k", "reel", "wins"]]
        print(tg, ", ".join(f"{r.dep}: {r.reel}→{r.wins}/{r.k}" for r in top.itertuples()))


if __name__ == "__main__":
    main()
