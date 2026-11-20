"""Modèle de second tour : prédit le vainqueur final d'une circonscription (réelle ou redessinée)
à partir des voix du 1er tour agrégées.

Chaîne de simulation (règles et comportements observés aux législatives 2024) :
  1. élection au 1er tour : > 50 % des exprimés et >= 25 % des inscrits ;
  2. qualification : les deux premiers + toute force >= 12,5 % des inscrits ;
  3. désistements (« front républicain ») : dans une triangulaire où l'extrême droite est qualifiée,
     un candidat NFP / ENS / divers gauche / EXG / autre classé 3e ou au-delà se retire
     (observé en 2024 : 96 % des NFP, 84 % des ENS ; RN, LR et DVD se maintiennent) ;
  4. reports de voix : régression non négative (NNLS), bureau par bureau, des voix du 2nd tour de chaque
     finaliste sur les voix du 1er tour (sa propre force, celles des autres finalistes, celles des
     éliminés, les non-exprimés), estimée séparément par configuration de second tour
     (ex. gauche contre extrême droite) ;
  5. incertitude : bruit gaussien sur les parts des finalistes, calibré par validation croisée.

Usage : python src/second_tour.py   -> validation croisée sur le découpage réel 2024
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent))
from common import PROC, RESULTS, ROOT  # noqa: E402

RAW = ROOT / "data" / "raw"
DEPTS = [f"{i:02d}" for i in range(1, 96) if i != 20] + ["2A", "2B"]

# forces utilisées par le modèle (nuances du ministère regroupées)
FORCE_OF = {"UG": "NFP", "RN": "RN", "UXD": "RN", "ENS": "ENS", "HOR": "ENS", "UDI": "ENS", "LR": "LR",
            "DVD": "DVD", "DVC": "DVC", "DVG": "DVG", "SOC": "DVG", "FI": "DVG", "COM": "DVG", "VEC": "DVG",
            "RDG": "DVG", "EXG": "EXG", "EXD": "EXD", "REC": "EXD", "REG": "REG", "ECO": "OTHER",
            "DIV": "OTHER", "DSV": "OTHER"}
FORCES = ["NFP", "RN", "ENS", "LR", "DVD", "DVC", "DVG", "EXG", "EXD", "REG", "OTHER"]
BLOC = {"NFP": "G", "DVG": "G", "EXG": "G", "ENS": "C", "DVC": "C", "LR": "D", "DVD": "D",
        "RN": "X", "EXD": "X", "REG": "R", "OTHER": "O"}
BLOC_ORDER = "XGCDRO"
WITHDRAW = {"NFP", "ENS", "DVG", "EXG", "OTHER"}   # se retirent en 3e position face à l'extrême droite
MIN_BUREAUX = 300                                 # en dessous : modèle générique (duel / triangulaire)
FEATS_OTHERS = FORCES + ["NONEXP"]


def norm_bv(s) -> str:
    s = str(s).strip()
    return (s.lstrip("0") or "0") if s.isdigit() else s.upper()


def norm_dep(s) -> str:
    s = str(s).strip()
    return s.zfill(2) if s.isdigit() else s


# --------------------------------------------------------------------------- données

def t1_bureaux() -> pd.DataFrame:
    """Voix du 1er tour par bureau, regroupées par force du modèle."""
    b = pd.read_parquet(PROC / "bureaux.parquet")
    out = b[["key", "dep", "com", "circo", "inscrits", "exprimes"]].copy()
    for f in FORCES:
        out[f] = 0.0
    for n, f in FORCE_OF.items():
        if n in b.columns:
            out[f] += b[n]
    out["NONEXP"] = out.inscrits - out.exprimes
    return out


def parse_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Format long : une ligne par candidat (force, voix, élu)."""
    parts = []
    i = 1
    while f"Nuance candidat {i}" in df.columns:
        p = df[["_id", f"Nuance candidat {i}", f"Voix {i}", f"Elu {i}"]].copy()
        p.columns = ["_id", "nuance", "voix", "elu"]
        parts.append(p.dropna(subset=["nuance"]))
        i += 1
    L = pd.concat(parts)
    L["force"] = L.nuance.map(FORCE_OF).fillna("OTHER")
    L["voix"] = pd.to_numeric(L.voix).astype(float)
    L["elu"] = L.elu.astype(str).str.lower().isin(["élu", "true", "oui"])
    return L


def circo_results(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path, sep=";", dtype=str)
    df["dep"] = df["Code département"].map(norm_dep)
    df = df[df.dep.isin(DEPTS)].copy()
    df["circo"] = df["Code circonscription législative"].str[-2:]
    df["_id"] = df.dep + "_" + df.circo
    L = parse_candidates(df)
    return df.set_index("_id"), L


def t2_bureaux() -> pd.DataFrame:
    df = pd.read_csv(RAW / "leg2024_t2_bv.csv", sep=";", dtype=str)
    df["dep"] = df["Code département"].map(norm_dep)
    df = df[df.dep.isin(DEPTS)].copy()
    df["key"] = df["Code commune"].str.zfill(5) + "_" + df["Code BV"].map(norm_bv)
    df["_id"] = df.key
    L = parse_candidates(df)
    W = L.pivot_table(index="_id", columns="force", values="voix", aggfunc="sum").fillna(0)
    return W


def actual_outcomes() -> pd.DataFrame:
    """Par circonscription réelle : élu au 1er tour ?, finalistes du 2nd tour, force élue."""
    c1, L1 = circo_results(RAW / "leg2024_t1_circo.csv")
    c2, L2 = circo_results(RAW / "leg2024_t2_circo.csv")
    rows = []
    for cid in c1.index:
        e1 = L1[(L1._id == cid) & L1.elu]
        if len(e1):
            rows.append(dict(cid=cid, t1_elected=True, finalists=(), winner=e1.force.iloc[0]))
            continue
        l2 = L2[L2._id == cid]
        fins = tuple(l2.groupby("force").voix.sum().sort_values(ascending=False).index)
        rows.append(dict(cid=cid, t1_elected=False, finalists=fins, winner=l2[l2.elu].force.iloc[0]))
    return pd.DataFrame(rows).set_index("cid")


# --------------------------------------------------------------------------- modèle

def config_key(finalists) -> str:
    return "-".join(sorted((BLOC[f] for f in finalists), key=BLOC_ORDER.index))


def slot_order(finalists, t1votes: dict | None = None):
    """Ordre canonique des finalistes : par bloc ; pour le modèle générique, par voix du 1er tour."""
    if t1votes is not None:
        return sorted(finalists, key=lambda f: -t1votes.get(f, 0))
    return sorted(finalists, key=lambda f: (BLOC_ORDER.index(BLOC[f]), f))


def design(X: pd.DataFrame, slots) -> np.ndarray:
    """Variables explicatives : voix 1er tour des finalistes (dans l'ordre des slots), puis des autres forces."""
    own = np.column_stack([X[f].values for f in slots])
    others = np.column_stack([np.where(f in slots, 0.0, X[f].values) if f in FORCES else X[f].values
                              for f in FEATS_OTHERS])
    return np.hstack([own, others])


def fit(train: pd.DataFrame, T2: pd.DataFrame, outcomes: pd.DataFrame) -> dict:
    """train : bureaux (1er tour) des circonscriptions d'entraînement ; T2 : voix 2nd tour par bureau."""
    models = {}
    groups = {}
    for cid, g in train.groupby("cid"):
        o = outcomes.loc[cid]
        if o.t1_elected or len(set(BLOC[f] for f in o.finalists)) < len(o.finalists):
            continue
        g = g[g.key.isin(T2.index)]
        k = config_key(o.finalists)
        groups.setdefault(("cfg", k), []).append((g, slot_order(o.finalists), o.finalists))
        n = "DUEL" if len(o.finalists) == 2 else "TRI" if len(o.finalists) == 3 else None
        if n:
            t1v = g[list(o.finalists)].sum().to_dict()
            groups.setdefault(("gen", n), []).append((g, slot_order(o.finalists, t1v), o.finalists))
    for (kind, k), items in groups.items():
        nb = sum(len(g) for g, _, _ in items)
        if kind == "cfg" and nb < MIN_BUREAUX:
            continue
        Xs, Ys = [], []
        for g, slots, fins in items:
            Xs.append(design(g, slots))
            Ys.append(np.column_stack([T2.loc[g.key, f].values if f in T2.columns else np.zeros(len(g)) for f in slots]))
        Xa, Ya = np.vstack(Xs), np.vstack(Ys)
        coefs = [nnls(Xa, Ya[:, j], maxiter=5000)[0] for j in range(Ya.shape[1])]
        models[k if kind == "cfg" else k] = dict(coefs=coefs, n_bureaux=nb, n_circos=len(items), generic=(kind == "gen"))
    return models


def first_round(t1: dict, inscrits: float, exprimes: float):
    """-> ('T1', vainqueur) ou ('T2', finalistes)."""
    cands = sorted(((f, v) for f, v in t1.items() if f in FORCES and v > 0), key=lambda kv: -kv[1])
    if cands[0][1] > 0.5 * exprimes and cands[0][1] >= 0.25 * inscrits:
        return "T1", cands[0][0]
    q = [f for i, (f, v) in enumerate(cands) if i < 2 or v >= 0.125 * inscrits]
    if len(q) >= 3 and any(BLOC[f] == "X" for f in q):
        q = q[:2] + [f for f in q[2:] if f not in WITHDRAW]
    return "T2", tuple(q)


def predict_votes(models: dict, t1: dict, finalists) -> tuple[dict, str]:
    k = config_key(finalists)
    if k in models and len(set(BLOC[f] for f in finalists)) == len(finalists):
        slots, m, used = slot_order(finalists), models[k], k
    else:
        gen = "DUEL" if len(finalists) == 2 else "TRI"
        if len(finalists) > 3:  # quadrangulaire : on garde les trois premiers
            finalists = slot_order(finalists, t1)[:3]
        slots, m, used = slot_order(finalists, t1), models[gen], gen
    x = np.array([t1.get(f, 0.0) for f in slots]
                 + [0.0 if f in slots else t1.get(f, 0.0) for f in FORCES] + [t1["NONEXP"]])
    return {f: float(np.dot(c, x)) for f, c in zip(slots, m["coefs"])}, used


def win_probabilities(votes: dict, sigma: float, n: int = 4000, rng=None) -> dict:
    """Probabilité de victoire de chaque finaliste : bruit gaussien (écart-type sigma) sur les parts."""
    fs = list(votes)
    s = np.array([votes[f] for f in fs])
    s = s / s.sum()
    if len(fs) == 2:
        p0 = float(norm.cdf((s[0] - s[1]) / (sigma * np.sqrt(2))))
        return {fs[0]: p0, fs[1]: 1 - p0}
    rng = rng or np.random.default_rng(0)
    draws = s + rng.normal(0, sigma, size=(n, len(fs)))
    w = np.bincount(draws.argmax(axis=1), minlength=len(fs)) / n
    return dict(zip(fs, w))


def simulate(models: dict, sigma: float, t1: dict, inscrits: float, exprimes: float) -> dict:
    """Simulation complète d'une circonscription -> vainqueur probable et probabilités."""
    stage, res = first_round(t1, inscrits, exprimes)
    if stage == "T1":
        return dict(stage="T1", finalists=(res,), votes={res: t1[res]}, winner=res, proba={res: 1.0}, model="T1")
    votes, used = predict_votes(models, t1, res)
    proba = win_probabilities(votes, sigma)
    return dict(stage="T2", finalists=res, votes=votes, winner=max(votes, key=votes.get), proba=proba, model=used)


# --------------------------------------------------------------------------- retraits probabilistes

WCLASS = {"NFP": "G", "DVG": "G", "EXG": "G", "OTHER": "G", "ENS": "C", "DVC": "C",
          "LR": "D", "DVD": "D", "REG": "D", "RN": "X", "EXD": "X"}
WCLASSES = ["G", "C", "D", "X"]


def withdrawal_rows(C: pd.DataFrame, O: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par candidat qualifié classé 3e ou au-delà : se maintient-il au 2nd tour ?"""
    rows = []
    for cid, r in C.iterrows():
        o = O.loc[cid]
        if o.t1_elected:
            continue
        t1 = r.to_dict()
        cands = sorted(((f, t1[f]) for f in FORCES if t1[f] > 0), key=lambda kv: -kv[1])
        q = [(f, v) for i, (f, v) in enumerate(cands) if i < 2 or v >= 0.125 * t1["inscrits"]]
        if len(q) < 3:
            continue
        for f, v in q[2:]:
            rows.append(dict(cid=cid, force=f, cls=WCLASS[f], gap=(q[1][1] - v) / t1["exprimes"],
                             x_q=float(any(BLOC[g] == "X" for g, _ in q)), stays=float(f in o.finalists)))
    return pd.DataFrame(rows)


class Withdrawal:
    """P(se maintenir) = logistique( a[classe] + b·écart au 2e + c·(extrême droite qualifiée) ), pénalité L2 légère."""

    def fit(self, W: pd.DataFrame, l2: float = 0.5):
        from scipy.optimize import minimize
        X = np.column_stack([(W.cls == c).astype(float) for c in WCLASSES] + [W.gap * 10, 1 - W.x_q])
        y = W.stays.values

        def nll(w):
            z = X @ w
            return np.sum(np.logaddexp(0, z) - y * z) + l2 * np.sum(w[len(WCLASSES):] ** 2) + 0.05 * np.sum(w ** 2)

        self.w = minimize(nll, np.zeros(X.shape[1]), method="L-BFGS-B").x
        return self

    def p_stay(self, force: str, gap: float, x_q: bool) -> float:
        z = self.w[WCLASSES.index(WCLASS[force])] + self.w[-2] * gap * 10 + self.w[-1] * (0.0 if x_q else 1.0)
        return float(1 / (1 + np.exp(-z)))


def scenarios(t1: dict, wmodel: "Withdrawal | None"):
    """Configurations de 2nd tour possibles et leurs probabilités.
    -> [('T1', vainqueur, 1.0)] ou [('T2', finalistes, proba), ...]"""
    cands = sorted(((f, v) for f, v in t1.items() if f in FORCES and v > 0), key=lambda kv: -kv[1])
    if cands[0][1] > 0.5 * t1["exprimes"] and cands[0][1] >= 0.25 * t1["inscrits"]:
        return [("T1", cands[0][0], 1.0)]
    q = [(f, v) for i, (f, v) in enumerate(cands) if i < 2 or v >= 0.125 * t1["inscrits"]]
    top, rest = [f for f, _ in q[:2]], q[2:]
    if not rest:
        return [("T2", tuple(top), 1.0)]
    x_q = any(BLOC[f] == "X" for f, _ in q)
    if wmodel is None:  # règle déterministe
        keep = [f for f, _ in rest if not (x_q and f in WITHDRAW)]
        return [("T2", tuple(top + keep), 1.0)]
    ps = [wmodel.p_stay(f, (q[1][1] - v) / t1["exprimes"], x_q) for f, v in rest]
    out = []
    for mask in range(2 ** len(rest)):
        pr, keep = 1.0, []
        for i, (f, _) in enumerate(rest):
            if mask >> i & 1:
                pr *= ps[i]
                keep.append(f)
            else:
                pr *= 1 - ps[i]
        if pr > 1e-4:
            out.append(("T2", tuple(top + keep), pr))
    return out


_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(24)
_GH_W = _GH_W / _GH_W.sum()


def win_probs_exact(votes: dict, sigma: float) -> dict:
    """P(chaque finaliste arrive en tête) avec bruit gaussien indépendant sur les parts (quadrature de Gauss-Hermite)."""
    fs = list(votes)
    s = np.array([max(votes[f], 0.0) for f in fs])
    s = s / max(s.sum(), 1e-9)
    if len(fs) == 2:
        p0 = float(norm.cdf((s[0] - s[1]) / (sigma * np.sqrt(2))))
        return {fs[0]: p0, fs[1]: 1 - p0}
    out = {}
    for i, f in enumerate(fs):
        xi = s[i] + sigma * _GH_X
        pr = np.ones_like(xi)
        for j in range(len(fs)):
            if j != i:
                pr *= norm.cdf((xi - s[j]) / sigma)
        out[f] = float(np.dot(_GH_W, pr))
    tot = sum(out.values())
    return {f: p / tot for f, p in out.items()}


def win_distribution(models: dict, wmodel, sigma: float, t1: dict) -> dict:
    """Probabilité finale de victoire de chaque force, tous scénarios de retrait confondus."""
    res: dict[str, float] = {}
    for stage, fins, pr in scenarios(t1, wmodel):
        if stage == "T1":
            res[fins] = res.get(fins, 0.0) + pr
            continue
        v, _ = predict_votes(models, t1, fins)
        for f, p in win_probs_exact(v, sigma).items():
            res[f] = res.get(f, 0.0) + pr * p
    return res


# --------------------------------------------------------------------------- validation

def validate(n_folds: int = 5, seed: int = 0):
    B = t1_bureaux()
    B["cid"] = B.dep + "_" + B.circo
    T2 = t2_bureaux()
    O = actual_outcomes()
    C = B.groupby("cid")[FORCES + ["NONEXP", "inscrits", "exprimes"]].sum()

    rng = np.random.default_rng(seed)
    deps = np.array(sorted(B.dep.unique()))
    fold_of = dict(zip(deps, rng.permutation(np.arange(len(deps)) % n_folds)))
    WR = withdrawal_rows(C, O)
    T2c = T2.join(B.set_index("key").cid, how="inner").groupby("cid").sum()
    rows, resid, wlog = [], [], []
    for k in range(n_folds):
        test_deps = {d for d, f in fold_of.items() if f == k}
        models = fit(B[~B.dep.isin(test_deps)], T2, O)
        wtrain = WR[~WR.cid.str.split("_").str[0].isin(test_deps)]
        wtest = WR[WR.cid.str.split("_").str[0].isin(test_deps)]
        wm = Withdrawal().fit(wtrain)
        for w in wtest.itertuples():
            p = min(max(wm.p_stay(w.force, w.gap, bool(w.x_q)), 1e-6), 1 - 1e-6)
            pd_ = 0.999 if (not w.x_q or w.force not in WITHDRAW) else 0.001
            wlog.append((w.stays, p, pd_))
        for cid in C.index[C.index.str[:2].isin(test_deps) | C.index.str[:3].str[:2].isin(test_deps)]:
            if cid.split("_")[0] not in test_deps:
                continue
            t1 = C.loc[cid].to_dict()
            o = O.loc[cid]
            stage, res = first_round(t1, t1["inscrits"], t1["exprimes"])
            r = dict(cid=cid, actual=o.winner, actual_cfg=config_key(o.finalists) if not o.t1_elected else "T1",
                     pred_cfg=("T1" if stage == "T1" else config_key(res)),
                     cfg_ok=(stage == "T1") == o.t1_elected and (stage == "T1" or set(res) == set(o.finalists)))
            # (a) configuration réelle connue : teste les reports seuls
            if o.t1_elected:
                r["pred_known_cfg"] = o.winner
            else:
                v, _ = predict_votes(models, t1, o.finalists)
                r["pred_known_cfg"] = max(v, key=v.get)
                tot = sum(v.values())
                act = {f: float(T2c.loc[cid, f]) if f in T2c.columns else 0.0 for f in v}
                atot = sum(act.values())
                resid += [v[f] / tot - act[f] / atot for f in v]
            # (b) chaîne complète depuis le 1er tour
            if stage == "T1":
                r["pred_full"], r["model"], r["votes"] = res, "T1", {res: 1.0}
            else:
                v, used = predict_votes(models, t1, res)
                r["pred_full"], r["model"], r["votes"] = max(v, key=v.get), used, v
            r["t1"] = t1
            r["wm"] = wm
            r["models"] = models
            rows.append(r)
    R = pd.DataFrame(rows).set_index("cid")
    sigma = float(np.std(resid))
    # probabilités complètes (retraits probabilistes + bruit sur les parts), hors échantillon
    dists = [win_distribution(r.models, r.wm, sigma, r.t1) for r in R.itertuples()]
    R["p_actual_prob"] = [d.get(a, 0.0) for d, a in zip(dists, R.actual)]
    R["pred_prob"] = [max(d, key=d.get) for d in dists]
    R["p_pred_prob"] = [max(d.values()) for d in dists]
    R["dist"] = [json.dumps({k: round(v, 4) for k, v in d.items()}) for d in dists]
    WL = pd.DataFrame(wlog, columns=["y", "p", "p_det"])
    ll = lambda y, p: -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    R.attrs["withdrawal"] = dict(n=len(WL), logloss_logit=ll(WL.y, WL.p), logloss_rule=ll(WL.y, WL.p_det),
                                 acc_logit=float(((WL.p > .5) == WL.y).mean()), acc_rule=float(((WL.p_det > .5) == WL.y).mean()))
    R = R.drop(columns=["t1", "wm", "models"])
    R["p_win_actual"] = [1.0 if r.model == "T1" and r.pred_full == r.actual else
                         win_probabilities(r.votes, sigma).get(r.actual, 0.0) if r.model != "T1" else 0.0
                         for r in R.itertuples()]
    R["p_pred"] = [max(win_probabilities(r.votes, sigma).values()) if r.model != "T1" else 1.0 for r in R.itertuples()]
    return R, sigma


def main():
    R, sigma = validate()
    n = len(R)
    out = RESULTS / "second_tour"
    out.mkdir(parents=True, exist_ok=True)
    R.drop(columns=["votes"]).to_csv(out / "validation.csv")
    w = R.attrs["withdrawal"]
    print(f"Retraits ({w['n']} cas hors échantillon) : log-loss logistique {w['logloss_logit']:.3f} contre règle {w['logloss_rule']:.3f} ; "
          f"exactitude {w['acc_logit']:.1%} contre {w['acc_rule']:.1%}")
    print(f"Circonscriptions : {n}")
    print(f"Configuration de 2nd tour correctement prédite : {R.cfg_ok.mean():.1%}")
    print(f"Vainqueur correct, configuration réelle connue : {(R.pred_known_cfg == R.actual).mean():.1%}")
    print(f"Vainqueur correct, chaîne complète depuis le 1er tour : {(R.pred_full == R.actual).mean():.1%}")
    print(f"Écart-type des erreurs de part (sigma) : {sigma:.3f}")
    brier = np.mean([(1 - p) ** 2 for p in R.p_win_actual])
    brier2 = np.mean([(1 - p) ** 2 for p in R.p_actual_prob])
    print(f"Score de Brier (proba du vrai vainqueur) : règle déterministe {brier:.3f} ; retraits probabilistes {brier2:.3f}")
    print(f"Vainqueur le plus probable correct (retraits probabilistes) : {(R.pred_prob == R.actual).mean():.1%}")
    exp = {}
    for d in R.dist:
        for f, p in json.loads(d).items():
            exp[f] = exp.get(f, 0) + p
    seats = pd.DataFrame({"réel": R.actual.value_counts(), "prédit (config. connue)": R.pred_known_cfg.value_counts(),
                          "prédit (chaîne complète)": R.pred_full.value_counts(),
                          "sièges attendus": pd.Series(exp).round(0)}).fillna(0).astype(int)
    print(seats.sort_values("réel", ascending=False))
    # calibration
    R["bin"] = pd.cut(R.p_pred_prob, [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1.0], include_lowest=True)
    cal = R.groupby("bin", observed=True).apply(lambda g: pd.Series({"n": len(g), "proba moyenne": g.p_pred_prob.mean(),
                                                                    "taux de réussite": (g.pred_prob == g.actual).mean()}))
    print(cal.round(2))
    json.dump(dict(sigma=sigma), open(out / "sigma.json", "w"))
    save_full_model(sigma)


def save_full_model(sigma: float):
    """Modèle final, estimé sur toutes les circonscriptions (sans objets Python : réutilisable partout)."""
    import pickle
    B = t1_bureaux()
    B["cid"] = B.dep + "_" + B.circo
    O = actual_outcomes()
    C = B.groupby("cid")[FORCES + ["NONEXP", "inscrits", "exprimes"]].sum()
    wm = Withdrawal().fit(withdrawal_rows(C, O))
    pickle.dump(dict(models=fit(B, t2_bureaux(), O), withdrawal_w=wm.w, sigma=sigma),
                open(RESULTS / "second_tour" / "model.pkl", "wb"))
    print("coefficients retrait :", dict(zip(WCLASSES + ["écart(x10pts)", "sans ext. droite"], np.round(wm.w, 2))))


def load_full_model() -> dict:
    import pickle
    M = pickle.load(open(RESULTS / "second_tour" / "model.pkl", "rb"))
    wm = Withdrawal()
    wm.w = M["withdrawal_w"]
    M["withdrawal"] = wm
    return M


if __name__ == "__main__":
    main()
