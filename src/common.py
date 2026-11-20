"""Définitions partagées : forces politiques, chargement des données."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

# Regroupement des nuances du ministère en « forces ».
# Les nuances non listées restent chacune une force à part entière (elles peuvent arriver en tête).
FORCES = {
    "NFP": ["UG"],
    "RN": ["RN", "UXD"],          # RN + alliés ciottistes
    "ENS": ["ENS", "HOR", "UDI"],  # Ensemble pour la République
    "LR": ["LR"],
}
TARGETS = ["NFP", "RN", "ENS", "LR"]


def force_columns(nuances: list[str]) -> dict[str, list[str]]:
    grouped = {n for ns in FORCES.values() for n in ns}
    out = dict(FORCES)
    for n in nuances:
        if n not in grouped:
            out[n] = [n]
    return out


def add_forces(df: pd.DataFrame, nuances: list[str]) -> tuple[pd.DataFrame, list[str]]:
    fc = force_columns(nuances)
    for f, ns in fc.items():
        df[f"F_{f}"] = df[ns].sum(axis=1)
    return df, list(fc)


def leaders(votes: np.ndarray, forces: list[str]) -> np.ndarray:
    """votes : (n_districts, n_forces) -> nom de la force en tête par district."""
    return np.array(forces)[votes.argmax(axis=1)]
