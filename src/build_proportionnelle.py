"""Proportionnelle intégrale : 539 sièges métropolitains répartis selon les voix du 1er tour 2024 (France métropolitaine,
circonscription nationale unique, sans seuil, méthode du plus fort reste).

Usage : python src/build_proportionnelle.py   -> site/proportionnelle.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import PROC  # noqa: E402
from second_tour import FORCE_OF  # noqa: E402

N_SEATS = 539
GROUP = {**FORCE_OF, "ECO": "ECO"}  # les écologistes hors NFP restent distincts, comme sur les cartes du 1er tour


def main():
    b = pd.read_parquet(PROC / "bureaux.parquet")
    nu = [c for c in b.columns if c in GROUP]
    votes = b[nu].sum().groupby(GROUP).sum().sort_values(ascending=False)
    total = float(votes.sum())
    quota = votes / total * N_SEATS
    seats = quota.astype(int)
    for f in (quota - seats).sort_values(ascending=False).index[: N_SEATS - int(seats.sum())]:
        seats[f] += 1
    out = dict(seats_total=N_SEATS, exprimes=int(total),
               forces={f: dict(voix=int(votes[f]), part=round(100 * votes[f] / total, 2), sieges=int(seats[f]))
                       for f in votes.index})
    path = Path(__file__).resolve().parent.parent / "site" / "proportionnelle.json"
    json.dump(out, open(path, "w"), ensure_ascii=False, indent=1)
    for f, v in out["forces"].items():
        print(f"{f:6s} {v['part']:5.2f} %  {v['sieges']:3d}")


if __name__ == "__main__":
    main()
