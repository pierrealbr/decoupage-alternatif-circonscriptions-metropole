"""Redécoupage optimisé : maximise, département par département, le nombre de circonscriptions
où une force arrive en tête au 1er tour.

Méthode : chaîne de Markov ReCom (GerryChain) + optimisation par « short bursts »
(Cannon et al., 2020). Contraintes :
  - on reste dans chaque département, avec le même nombre de circonscriptions qu'aujourd'hui ;
  - chaque circonscription est d'un seul tenant (îles et enclaves rattachées au plus proche) ;
  - écart d'inscrits à la moyenne départementale <= EPS (±20 % par défaut, plafond légal).

Usage :
  python src/optimize.py --deps 13 59 --targets NFP RN --steps-per-seat 300
  python src/optimize.py --all --workers 10
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import PROC, RESULTS, TARGETS, add_forces  # noqa: E402

NUANCES = ["RN", "UG", "ENS", "LR", "UXD", "DVD", "DVG", "DVC", "EXG", "REG", "REC", "HOR",
           "ECO", "UDI", "DIV", "DSV", "EXD", "SOC", "RDG", "FI", "COM", "VEC"]


# --------------------------------------------------------------------------- données

def load_department(dep: str):
    U = gpd.read_parquet(PROC / "units.parquet")
    U = U[U.dep == dep].copy()
    E = pd.read_parquet(PROC / "edges.parquet")
    E = E[E.dep == dep]
    b = pd.read_parquet(PROC / "bureaux.parquet")
    k = b[b.dep == dep].circo.nunique()

    U, forces = add_forces(U, NUANCES)
    fcols = [f"F_{f}" for f in forces]
    # on ne garde que les forces qui pèsent quelque part dans le département
    tot = U[fcols].sum()
    keep = [f for f in forces if f in TARGETS or tot[f"F_{f}"] > 0]
    forces = keep

    G = nx.Graph()
    for r in U.itertuples():
        G.add_node(r.uid, pop=float(r.inscrits), votes=np.array([getattr(r, f"F_{f}") for f in forces], dtype=float))
    G.add_edges_from(zip(E.u, E.v))
    connect_components(G, U)
    return G, k, forces


def connect_components(G: nx.Graph, U: gpd.GeoDataFrame):
    """Relie îles / enclaves / trous de contours au morceau le plus proche (pont, bac...)."""
    geo = U.set_index("uid").geometry
    while True:
        comps = sorted(nx.connected_components(G), key=len)
        if len(comps) == 1:
            return
        small = list(comps[0])
        rest = geo.drop(index=small)
        best = None
        for u in small:
            d = rest.distance(geo[u])
            v = d.idxmin()
            if best is None or d[v] < best[0]:
                best = (d[v], u, v)
        G.add_edge(best[1], best[2])


# --------------------------------------------------------------------------- score

def make_score(t: int):
    def score(part) -> float:
        V = np.array([part["votes"][d] for d in part.parts])
        others = np.delete(V, t, axis=1).max(axis=1)
        expr = V.sum(axis=1)
        margin = (V[:, t] - others) / np.maximum(expr, 1)
        wins = int((margin > 0).sum())
        losing = margin[margin <= 0]
        bonus = 0.0
        if len(losing):
            # pousse le district perdant le plus proche de la bascule (bonus < 1 : une victoire vaut toujours plus)
            bonus = 0.9 * min(max(1 + losing.max() / 0.25, 0.0), 1.0)
        return wins + bonus
    return score


# --------------------------------------------------------------------------- objectif « second tour »

_T2 = None


def t2_model():
    global _T2
    if _T2 is None:
        import second_tour
        _T2 = second_tour.load_full_model()
    return _T2


def make_t2_eval(forces: list[str], target: str):
    """-> fonction (vecteur de voix 1er tour, inscrits) -> probabilité que `target` gagne au final."""
    import second_tour as st
    M = t2_model()
    idx = {}
    for i, f in enumerate(forces):
        g = f if f in st.FORCES else st.FORCE_OF.get(f, "OTHER")
        idx.setdefault(g, []).append(i)
    cache = {}

    def p_win(v: np.ndarray, pop: float) -> float:
        key = (tuple(np.round(v).astype(int)), int(round(pop)))
        if key in cache:
            return cache[key]
        t1 = {g: float(v[ii].sum()) for g, ii in idx.items()}
        for g in st.FORCES:
            t1.setdefault(g, 0.0)
        t1["exprimes"] = float(v.sum())
        t1["inscrits"] = float(pop)
        t1["NONEXP"] = max(t1["inscrits"] - t1["exprimes"], 0.0)
        p = st.win_distribution(M["models"], M["withdrawal"], M["sigma"], t1).get(target, 0.0)
        if len(cache) > 200_000:
            cache.clear()
        cache[key] = p
        return p
    return p_win


def make_score_t2(p_win):
    def score(part) -> float:
        return float(sum(p_win(part["votes"][d], part["population"][d]) for d in part.parts))
    return score


def wins_of(part, t: int) -> int:
    V = np.array([part["votes"][d] for d in part.parts])
    return int((V.argmax(axis=1) == t).sum())


# --------------------------------------------------------------------------- optimisation

def run(dep: str, target: str, steps_per_seat: int = 1000, eps: float = 0.20, burst: int = 5, seed: int = 0,
        n_seeds: int = 3, objective: str = "t1"):
    from functools import partial

    from gerrychain import Graph, Partition
    from gerrychain.constraints import within_percent_of_ideal_population
    from gerrychain.optimization import SingleMetricOptimizer
    from gerrychain.proposals import recom
    from gerrychain.partition import recursive_tree_part
    from gerrychain.updaters import Tally

    t0 = time.time()
    nxG, k, forces = load_department(dep)
    t = forces.index(target)
    p_win = make_t2_eval(forces, target) if objective == "t2" else None
    score_fn = make_score_t2(p_win) if p_win else make_score(t)
    uids = list(nxG.nodes)  # on renumérote 0..n-1 (GerryChain 1.0 utilise des identifiants internes)
    nxG = nx.relabel_nodes(nxG, {u: i for i, u in enumerate(uids)})
    G = Graph.from_networkx(nxG)
    total_pop = sum(nxG.nodes[n]["pop"] for n in nxG.nodes)
    ideal = total_pop / k

    VM = np.vstack([nxG.nodes[i]["votes"] for i in range(len(uids))])

    def part_votes(nodeset):
        return VM[list(nodeset)].sum(axis=0)

    def votes_updater(part):  # sommes vectorielles par circonscription, incrémentales
        parent = part.parent
        if parent is None:
            return {d: part_votes(part.parts[d]) for d in part.parts}
        prev = parent["votes"]
        changed = set(part.flips.values()) | {parent.assignment[n] for n in part.flips}
        out = dict(prev)
        for d in changed:
            out[d] = part_votes(part.parts[d])
        return out

    updaters = {"population": Tally("pop", alias="population"), "votes": votes_updater}

    if k == 1:
        assign = {n: 0 for n in G.nodes}
        part = Partition(G, assign, updaters)
        w = wins_of(part, t)
        sc = score_fn(part)
        return dict(dep=dep, target=target, k=k, wins=w, score=float(sc), steps=0, seconds=time.time() - t0,
                    assignment={int(u): 0 for u in uids}, forces=forces)

    n_steps = max(steps_per_seat * k, 200)
    best, best_score, best_wins = None, -1.0, -1
    for s in range(n_seeds):  # essais indépendants, on garde le meilleur
        rng = random.Random(seed * 1000 + s)
        try:
            assign = recursive_tree_part(G, range(k), ideal, "pop", eps * 0.9, rng=rng)
            init = Partition(G, assign, updaters)
            proposal = partial(recom, pop_col="pop", pop_target=ideal, epsilon=eps, node_repeats=0, rng=rng)
            cons = [within_percent_of_ideal_population(init, eps)]
            opt = SingleMetricOptimizer(proposal, cons, init, score_fn, maximize=True, rng=rng)
            for _ in opt.short_bursts(burst, n_steps // burst):
                pass
        except Exception as e:  # ex. PopulationBalanceError : on garde le meilleur trouvé jusque-là
            print(f"  {dep} {target} essai {s}: {e!r}", flush=True)
            if "opt" not in locals() or opt.best_part is None:
                continue
        if opt.best_score > best_score:
            best, best_score = opt.best_part, float(opt.best_score)
    return dict(dep=dep, target=target, k=k, wins=wins_of(best, t), score=best_score,
                steps=n_steps * n_seeds, seconds=round(time.time() - t0, 1),
                assignment={int(uids[n]): int(best.assignment[n]) for n in G.nodes}, forces=forces)


def _job(args):
    dep, target, sps, eps, seed, objective = args
    try:
        return run(dep, target, sps, eps, seed=seed, objective=objective)
    except Exception as e:  # on remonte l'erreur sans tuer le pool
        return dict(dep=dep, target=target, error=repr(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deps", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--targets", nargs="*", default=TARGETS)
    ap.add_argument("--steps-per-seat", type=int, default=1000)
    ap.add_argument("--eps", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--objective", choices=["t1", "t2"], default="t1",
                    help="t1 : circonscriptions en tête au 1er tour ; t2 : sièges attendus au 2nd tour (modèle second_tour)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="run")
    ap.add_argument("--skip-existing", action="store_true", help="ne recalcule pas les plans déjà enregistrés")
    a = ap.parse_args()

    b = pd.read_parquet(PROC / "bureaux.parquet")
    deps = sorted(b.dep.unique()) if a.all else a.deps
    # gros départements d'abord pour équilibrer la charge
    size = b.groupby("dep").circo.nunique()
    deps = sorted(deps, key=lambda d: -size[d])
    jobs = [(d, tg, a.steps_per_seat, a.eps, a.seed, a.objective) for d in deps for tg in a.targets]

    out = RESULTS / a.out
    (out / "plans").mkdir(parents=True, exist_ok=True)
    if a.skip_existing:
        jobs = [j for j in jobs if not (out / "plans" / f"{j[0]}_{j[1]}.json").exists()]
        print(f"{len(jobs)} calculs restants", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(_job, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            if "error" in r:
                print(f"[{i}/{len(jobs)}] ERREUR {r['dep']} {r['target']}: {r['error']}", flush=True)
                rows.append(r)
                continue
            json.dump(r["assignment"], open(out / "plans" / f"{r['dep']}_{r['target']}.json", "w"))
            r = {k: v for k, v in r.items() if k not in ("assignment", "forces")}
            rows.append(r)
            print(f"[{i}/{len(jobs)}] {r['dep']} {r['target']}: {r['wins']}/{r['k']} score {r['score']:.2f} ({r['seconds']}s)", flush=True)
    summ = out / "summary.csv"
    df = pd.DataFrame(rows)
    if a.skip_existing and summ.exists():
        df = pd.concat([pd.read_csv(summ, dtype={"dep": str}), df], ignore_index=True)
    df.to_csv(summ, index=False)


if __name__ == "__main__":
    main()
