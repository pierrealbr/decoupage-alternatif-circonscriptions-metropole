"""Sièges attendus au 2nd tour (modèle second_tour) : découpage réel contre découpages optimisés.

Usage :
  python src/report_t2.py max_t2            -> rapport complet (vérification + synthèse)
  python src/report_t2.py max_t2 --watch    -> une ligne par département terminé (suivi en direct)
"""
from __future__ import annotations

import json
import sys
import time
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import second_tour as st  # noqa: E402
from common import PROC, RESULTS, TARGETS  # noqa: E402

M = st.load_full_model()


def to_t1(df: pd.DataFrame) -> pd.DataFrame:
    """Colonnes de nuances -> forces du modèle de second tour (+ inscrits, exprimés, non-exprimés)."""
    out = pd.DataFrame(index=df.index)
    for f in st.FORCES:
        out[f] = 0.0
    for n, f in st.FORCE_OF.items():
        if n in df.columns:
            out[f] += df[n]
    out["inscrits"] = df.inscrits
    out["exprimes"] = df[[n for n in st.FORCE_OF if n in df.columns]].sum(axis=1)
    out["NONEXP"] = (out.inscrits - out.exprimes).clip(lower=0)
    return out


def dist(row) -> dict:
    return st.win_distribution(M["models"], M["withdrawal"], M["sigma"], row.to_dict())


def real_expected() -> pd.DataFrame:
    b = pd.read_parquet(PROC / "bureaux.parquet")
    nu = [c for c in b.columns if c in st.FORCE_OF]
    C = to_t1(b.groupby(["dep", "circo"])[["inscrits"] + nu].sum())
    rows = []
    for (dep, circo), r in C.iterrows():
        d = dist(r)
        rows.append(dict(dep=dep, circo=circo, **{f"p_{t}": d.get(t, 0.0) for t in TARGETS}))
    return pd.DataFrame(rows)


def plan_expected(U: pd.DataFrame, dep: str, target: str, run: str) -> dict:
    a = pd.Series(json.load(open(RESULTS / run / "plans" / f"{dep}_{target}.json")))
    a.index = a.index.astype(int)
    g = U[U.dep == dep]
    nu = [c for c in g.columns if c in st.FORCE_OF]
    V = to_t1(g.assign(_d=g.uid.map(a).values).groupby("_d")[["inscrits"] + nu].sum())
    ps = [dist(r).get(target, 0.0) for _, r in V.iterrows()]
    return dict(expected=float(np.sum(ps)), probable=int(np.sum(np.array(ps) > 0.5)), k=len(ps))


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "max_t2"
    watch = "--watch" in sys.argv
    U = pd.read_parquet(PROC / "units.parquet").drop(columns="geometry")
    R = real_expected()
    real_dep = R.groupby("dep")[[f"p_{t}" for t in TARGETS]].sum()
    deps = sorted(R.dep.unique())
    if watch:
        state = Path(RESULTS / run / ".reported")
        done = set(state.read_text().split()) if state.exists() else set()
        while True:
            for dep in deps:
                if dep in done or not all((RESULTS / run / "plans" / f"{dep}_{t}.json").exists() for t in TARGETS):
                    continue
                parts = []
                for t in TARGETS:
                    e = plan_expected(U, dep, t, run)
                    parts.append(f"{t} {real_dep.loc[dep, f'p_{t}']:.1f}→{e['expected']:.1f}")
                k = int((R.dep == dep).sum())
                done.add(dep)
                state.write_text(" ".join(sorted(done)))
                print(f"Dép. {dep} terminé ({k} circos, {len(done)}/{len(deps)}) : " + ", ".join(parts), flush=True)
            if len(done) == len(deps):
                print("TOUS LES DÉPARTEMENTS TERMINÉS", flush=True)
                return
            if subprocess.run(["pgrep", "-f", f"optimize.py --all.*--out {run}"], capture_output=True).returncode != 0:
                time.sleep(5)
                if len(done) < len(deps) and not all(
                        (RESULTS / run / "plans" / f"{d}_{t}.json").exists() for d in deps for t in TARGETS):
                    print(f"ARRÊT du calcul détecté avec {len(done)}/{len(deps)} départements", flush=True)
                    return
            time.sleep(15)

    rows = []
    for dep in deps:
        for t in TARGETS:
            e = plan_expected(U, dep, t, run)
            rows.append(dict(dep=dep, target=t, k=e["k"], reel=real_dep.loc[dep, f"p_{t}"],
                             attendu=e["expected"], probable=e["probable"]))
    D = pd.DataFrame(rows)
    D["gain"] = D.attendu - D.reel
    D.to_csv(RESULTS / run / "par_departement_t2.csv", index=False)
    real_prob = {t: int((R[f"p_{t}"] > 0.5).sum()) for t in TARGETS}
    S = D.groupby("target")[["reel", "attendu", "probable"]].sum().loc[TARGETS]
    S.insert(1, "réel (proba > 0,5)", [real_prob[t] for t in TARGETS])
    print(S.round(1).to_string())
    for t in TARGETS:
        top = D[D.target == t].nlargest(5, "gain")
        print(t, ", ".join(f"{r.dep}: {r.reel:.1f}→{r.attendu:.1f}/{r.k}" for r in top.itertuples()))


if __name__ == "__main__":
    main()
