"""Figures et page « Méthode » : exemple détaillé d'un département à 3 circonscriptions (Aveyron).

Usage : python src/build_method.py [dep] [force] [run]
Produit site/methode.html à partir de src/methode_template.html.
"""
from __future__ import annotations

import json
import random
import sys
from functools import partial
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from build_site import to_path  # noqa: E402
from common import PROC, RESULTS, ROOT, TARGETS, add_forces, leaders  # noqa: E402
from optimize import NUANCES, load_department, make_score, wins_of  # noqa: E402

DEP = sys.argv[1] if len(sys.argv) > 1 else "12"
TG = sys.argv[2] if len(sys.argv) > 2 else "ENS"
RUN = sys.argv[3] if len(sys.argv) > 3 else "max_eps20_v2"
KNOWN = ["NFP", "RN", "ENS", "LR", "DVD", "DVG", "DVC", "REG", "EXD", "ECO"]
EPS = 0.20


def col(f):
    return f"var(--f-{f if f in KNOWN else 'other'})"


# --------------------------------------------------------------------------- données
U = gpd.read_parquet(PROC / "units.parquet")
gd = U[U.dep == DEP].copy()
gd, forces_all = add_forces(gd, NUANCES)
G, K, forces = load_department(DEP)
FC = [f"F_{f}" for f in forces]
gd = gd.set_index("uid")
ideal = gd.inscrits.sum() / K

minx, miny, maxx, maxy = gd.total_bounds
W0, H0 = maxx - minx, maxy - miny
S = 600 / max(W0, H0)
W, H = W0 * S, H0 * S
TOL = max(np.hypot(W0, H0) / 1500, 8)


def tf(x, y):
    return (x - minx) * S, (maxy - y) * S


def pt(geom):
    p = geom.representative_point()
    x, y = tf(np.array([p.x]), np.array([p.y]))
    return float(x[0]), float(y[0])


def district_shapes(labels: pd.Series):
    geo = gd.assign(_d=labels).dissolve(by="_d").geometry
    geo = geo.buffer(30, join_style="mitre").buffer(-30, join_style="mitre").simplify(TOL)
    return geo


def stats(labels: pd.Series):
    V = gd.assign(_d=labels).groupby("_d")[["inscrits"] + FC].sum()
    rows = []
    for d, r in V.iterrows():
        expr = r[FC].sum()
        sh = {f: 100 * r[f"F_{f}"] / expr for f in forces}
        srt = sorted(sh.items(), key=lambda kv: -kv[1])
        rows.append(dict(d=d, inscrits=int(r.inscrits), ecart=100 * (r.inscrits / ideal - 1), shares=sh,
                         leader=srt[0][0], marge=srt[0][1] - srt[1][1]))
    return rows


def svg_open(w=W, h=H, pad=8, cls="fig"):
    return f'<svg class="{cls}" viewBox="{-pad} {-pad} {w + 2 * pad:.0f} {h + 2 * pad:.0f}" role="img">'


def bureaux_layer(fill_of=None, opacity_of=None, stroke="var(--panel)", sw=0.4):
    out = []
    for uid, g in gd.geometry.items():
        f = fill_of(uid) if fill_of else "var(--muted-fill)"
        o = opacity_of(uid) if opacity_of else 1
        out.append(f'<path d="{to_path(g.simplify(TOL / 2), tf)}" fill="{f}" fill-opacity="{o:.2f}" '
                   f'stroke="{stroke}" stroke-width="{sw}" vector-effect="non-scaling-stroke"/>')
    return "".join(out)


def outline_layer(labels, width=2.2, color="var(--ink)", label=True, names=None):
    geo = district_shapes(labels)
    out = []
    for d, g in geo.items():
        out.append(f'<path d="{to_path(g, tf)}" fill="none" stroke="{color}" stroke-width="{width}" '
                   f'vector-effect="non-scaling-stroke" stroke-linejoin="round"/>')
    if label:
        for d, g in geo.items():
            x, y = pt(g)
            nm = names[d] if names else str(d)
            out.append(f'<text class="dlabel" x="{x:.1f}" y="{y:.1f}">{nm}</text>')
    return "".join(out)


def cities_layer(n=4):
    b = pd.read_parquet(PROC / "bureaux.parquet")
    names = b[b.dep == DEP].groupby("com").nom_com.first()
    ins = gd.groupby("com").inscrits.sum().sort_values(ascending=False)
    out = []
    for com in ins.index[:n]:
        x, y = pt(gd[gd.com == com].geometry.union_all())
        out.append(f'<g class="city"><circle cx="{x:.1f}" cy="{y:.1f}" r="3"/>'
                   f'<text x="{x + 6:.1f}" y="{y + 4:.1f}">{names[com].strip()}</text></g>')
    return "".join(out)


# --------------------------------------------------------------------------- figure 1 : bureaux
lead_b = pd.Series(leaders(gd[FC].values, forces), index=gd.index)
sh_b = gd[FC].div(gd[FC].sum(axis=1).replace(0, 1), axis=0)
srt = np.sort(sh_b.values, axis=1)
marge_b = pd.Series(srt[:, -1] - srt[:, -2], index=gd.index)
real_lab = gd.circo.astype(int)
real_names = {d: f"{d}<tspan class='sup'>{'re' if d == 1 else 'e'}</tspan>" for d in real_lab.unique()}
fig_bureaux = (svg_open() + bureaux_layer(lambda u: col(lead_b[u]), lambda u: 0.3 + 0.7 * min(marge_b[u] / 0.25, 1))
               + outline_layer(real_lab, names=real_names) + cities_layer() + "</svg>")
lead_counts = lead_b.value_counts().to_dict()

# --------------------------------------------------------------------------- figure 2 : graphe
pos = {u: pt(g) for u, g in gd.geometry.items()}
edges_svg = "".join(f'<line x1="{pos[a][0]:.1f}" y1="{pos[a][1]:.1f}" x2="{pos[b][0]:.1f}" y2="{pos[b][1]:.1f}"/>'
                    for a, b in G.edges)
rmax = gd.inscrits.max()
nodes_svg = "".join(f'<circle cx="{pos[u][0]:.1f}" cy="{pos[u][1]:.1f}" r="{1.2 + 4.5 * np.sqrt(gd.inscrits[u] / rmax):.1f}" '
                    f'fill="{col(lead_b[u])}"><title>{gd.com[u]} bureau {gd.bv[u]} · {int(gd.inscrits[u])} inscrits</title></circle>'
                    for u in gd.index)
fig_graph = (svg_open() + bureaux_layer(None, None, stroke="var(--line)", sw=0.5)
             + f'<g class="edges">{edges_svg}</g><g class="nodes">{nodes_svg}</g></svg>')
n_nodes, n_edges = G.number_of_nodes(), G.number_of_edges()
deg = np.mean([d for _, d in G.degree])

# --------------------------------------------------------------------------- figure 3 : un pas ReCom
rng = random.Random(7)
pair = (1, 2)
merged = [u for u in gd.index if real_lab[u] in pair]
Hs = G.subgraph(merged).copy()
for a, b in Hs.edges:
    Hs[a][b]["w"] = rng.random()
T = nx.minimum_spanning_tree(Hs, weight="w")
root = merged[0]
parent = dict(nx.bfs_predecessors(T, root))
order = list(nx.bfs_tree(T, root))
sub = {u: gd.inscrits[u] for u in merged}
for u in reversed(order):
    if u in parent:
        sub[parent[u]] += sub[u]
tot = sum(gd.inscrits[u] for u in merged)
balanced = [(parent[u], u) for u in order if u in parent
            and abs(sub[u] / ideal - 1) <= EPS and abs((tot - sub[u]) / ideal - 1) <= EPS]
cut = balanced[rng.randrange(len(balanced))]
T2 = T.copy()
T2.remove_edge(*cut)
side = nx.node_connected_component(T2, cut[1])
new_lab = real_lab.copy()
for u in merged:
    new_lab[u] = pair[0] if u in side else pair[1]
in_pair = set(merged)


def pair_fill(lab):
    pal = {pair[0]: "var(--a1)", pair[1]: "var(--a2)"}
    return lambda u: pal[lab[u]] if u in in_pair else "var(--muted-fill)"


tree_svg = "".join(f'<line x1="{pos[a][0]:.1f}" y1="{pos[a][1]:.1f}" x2="{pos[b][0]:.1f}" y2="{pos[b][1]:.1f}"/>'
                   for a, b in T.edges)
bal_svg = "".join(f'<line x1="{pos[a][0]:.1f}" y1="{pos[a][1]:.1f}" x2="{pos[b][0]:.1f}" y2="{pos[b][1]:.1f}"/>'
                  for a, b in balanced)
cx = (pos[cut[0]][0] + pos[cut[1]][0]) / 2
cy = (pos[cut[0]][1] + pos[cut[1]][1]) / 2
cut_svg = (f'<line class="cut" x1="{pos[cut[0]][0]:.1f}" y1="{pos[cut[0]][1]:.1f}" x2="{pos[cut[1]][0]:.1f}" '
           f'y2="{pos[cut[1]][1]:.1f}"/><circle class="cutmark" cx="{cx:.1f}" cy="{cy:.1f}" r="9"/>')
pair_only = real_lab.where(real_lab.isin(pair), 0)
fig_recom_a = svg_open() + bureaux_layer(pair_fill(real_lab), None) + outline_layer(real_lab, names=real_names) + "</svg>"
fig_recom_b = (svg_open() + bureaux_layer(lambda u: "var(--merge)" if u in in_pair else "var(--muted-fill)", None)
               + f'<g class="tree">{tree_svg}</g><g class="balanced">{bal_svg}</g>{cut_svg}</svg>')
fig_recom_c = svg_open() + bureaux_layer(pair_fill(new_lab), None) + outline_layer(new_lab, names=real_names) + "</svg>"
pop_pair_before = {d: int(gd.inscrits[real_lab == d].sum()) for d in pair}
pop_pair_after = {d: int(gd.inscrits[new_lab == d].sum()) for d in pair}

# --------------------------------------------------------------------------- figure 4 : trajectoire de l'optimisation
from gerrychain import Graph, Partition  # noqa: E402
from gerrychain.constraints import within_percent_of_ideal_population  # noqa: E402
from gerrychain.optimization import SingleMetricOptimizer  # noqa: E402
from gerrychain.partition import recursive_tree_part  # noqa: E402
from gerrychain.proposals import recom  # noqa: E402
from gerrychain.updaters import Tally  # noqa: E402

t = forces.index(TG)
uids = list(G.nodes)
Gi = nx.relabel_nodes(G, {u: i for i, u in enumerate(uids)})
GG = Graph.from_networkx(Gi)
VM = np.vstack([Gi.nodes[i]["votes"] for i in range(len(uids))])


def vu(part):
    p = part.parent
    if p is None:
        return {d: VM[list(part.parts[d])].sum(0) for d in part.parts}
    out = dict(p["votes"])
    for d in set(part.flips.values()) | {p.assignment[n] for n in part.flips}:
        out[d] = VM[list(part.parts[d])].sum(0)
    return out


N_STEPS, BURST = 1000 * K, 5
score_fn = make_score(t)
traj = []
for s in range(3):
    r = random.Random(s)
    init = Partition(GG, recursive_tree_part(GG, range(K), ideal, "pop", EPS * 0.9, rng=r),
                     {"population": Tally("pop", alias="population"), "votes": vu})
    opt = SingleMetricOptimizer(partial(recom, pop_col="pop", pop_target=ideal, epsilon=EPS, node_repeats=0, rng=r),
                                [within_percent_of_ideal_population(init, EPS)], init, score_fn, maximize=True, rng=r)
    cur, best = [], []
    for part in opt.short_bursts(BURST, N_STEPS // BURST):
        cur.append(score_fn(part))
        best.append(opt.best_score)
    traj.append(dict(cur=cur, best=best, wins=wins_of(opt.best_part, t)))
    traj[-1]["reach"] = next(i for i, v in enumerate(best) if v >= traj[-1]["wins"])

# graphique (SVG, axes et grille récessifs)
CW, CH, ML, MR, MT, MB = 640, 300, 44, 70, 12, 34
ymax = K + 1


# on montre le début de la marche : au-delà, le meilleur score ne bouge plus
N_SHOW = max(60, int(np.ceil(1.6 * max(tr["reach"] for tr in traj) / 20) * 20))


def X(i):
    return ML + (CW - ML - MR) * i / N_SHOW


def Y(v):
    return MT + (CH - MT - MB) * (1 - v / ymax)


chart = [f'<svg class="chart" viewBox="0 0 {CW} {CH}" role="img" aria-label="Trajectoire du score">']
for v in range(0, ymax + 1):
    chart.append(f'<line class="grid" x1="{ML}" x2="{CW - MR}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>'
                 f'<text class="tick" x="{ML - 8}" y="{Y(v) + 4:.1f}" text-anchor="end">{v}</text>')
for i in range(0, N_SHOW + 1, N_SHOW // 5):
    chart.append(f'<line class="grid" x1="{X(i):.1f}" x2="{X(i):.1f}" y1="{CH - MB}" y2="{CH - MB + 4}"/>'
                 f'<text class="tick" x="{X(i):.1f}" y="{CH - MB + 18}" text-anchor="middle">{i}</text>')
chart.append(f'<text class="axis" x="{(ML + CW - MR) / 2}" y="{CH - 2}" text-anchor="middle">pas de la chaîne</text>')
step = 1
shades = ["var(--s1)", "var(--s2)", "var(--s3)"]
for k_, tr in enumerate(traj):
    pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(tr["cur"][:N_SHOW + 1]))
    chart.append(f'<polyline class="cur" points="{pts}" style="stroke:{shades[k_]}"/>')
for k_, tr in enumerate(traj):
    pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(tr["best"][:N_SHOW + 1]))
    chart.append(f'<polyline class="best" points="{pts}" style="stroke:{shades[k_]}"><title>Essai {k_ + 1} : '
                 f'meilleur score {tr["best"][-1]:.2f} ({tr["wins"]} circonscription(s))</title></polyline>')
    yl = Y(tr["best"][-1]) + (k_ - 1) * 13
    chart.append(f'<text class="dl" x="{CW - MR + 6}" y="{yl + 4:.1f}" style="fill:var(--ink)">essai {k_ + 1}</text>')
chart.append("</svg>")
fig_traj = "".join(chart)

# --------------------------------------------------------------------------- figure 5 : réel vs optimisé + parts de voix
plan = pd.Series(json.load(open(RESULTS / RUN / "plans" / f"{DEP}_{TG}.json")))
plan.index = plan.index.astype(int)
opt_lab = plan.reindex(gd.index)
# numérotation stable des circonscriptions optimisées : de l'ouest vers l'est
cent = gd.assign(_d=opt_lab).dissolve(by="_d").geometry.centroid
ren = {d: i + 1 for i, d in enumerate(cent.x.sort_values().index)}
opt_lab = opt_lab.map(ren)
opt_names = {d: f"{chr(64 + d)}" for d in ren.values()}


def leader_fill(lab):
    st = {r["d"]: r for r in stats(lab)}
    return (lambda u: col(st[lab[u]]["leader"])), (lambda u: 0.35 + 0.65 * min(st[lab[u]]["marge"] / 25, 1))


f1, o1 = leader_fill(real_lab)
f2, o2 = leader_fill(opt_lab)
fig_real = svg_open() + bureaux_layer(f1, o1, stroke="none") + outline_layer(real_lab, names=real_names) + cities_layer(3) + "</svg>"
fig_opt = svg_open() + bureaux_layer(f2, o2, stroke="none") + outline_layer(opt_lab, names=opt_names) + cities_layer(3) + "</svg>"
st_real, st_opt = stats(real_lab), stats(opt_lab)


def bars(rows, names):
    show = [f for f in ["ENS", "RN", "NFP", "LR"] if f in forces]
    BW, BH, LW = 640, 34, 70
    h = len(rows) * (BH + 14) + 6
    out = [f'<svg class="bars" viewBox="0 0 {BW} {h}" role="img">']
    for i, r in enumerate(sorted(rows, key=lambda r: r["d"])):
        y = 4 + i * (BH + 14)
        out.append(f'<text class="blabel" x="0" y="{y + BH / 2 + 5:.1f}">{names[r["d"]]}</text>')
        x = LW
        parts = [(f, r["shares"][f]) for f in show] + [("other", 100 - sum(r["shares"][f] for f in show))]
        for f, v in parts:
            w = (BW - LW - 4) * v / 100
            if w <= 0:
                continue
            fill = col(f) if f != "other" else "var(--f-other)"
            out.append(f'<rect x="{x:.1f}" y="{y}" width="{max(w - 2, 0.5):.1f}" height="{BH}" rx="3" fill="{fill}">'
                       f'<title>{names[r["d"]].split("<")[0]} · {f if f != "other" else "autres"} : {v:.1f} %</title></rect>')
            if v >= 9:
                out.append(f'<text class="bval" x="{x + 6:.1f}" y="{y + BH / 2 + 5:.1f}">{f if f != "other" else "autres"} {v:.0f}</text>')
            x += w
    out.append("</svg>")
    return "".join(out)


real_plain = {d: f"{d}{'re' if d == 1 else 'e'} circ." for d in real_lab.unique()}
opt_plain = {d: f"Circ. {chr(64 + d)}" for d in ren.values()}
fig_bars_real = bars(st_real, real_plain)
fig_bars_opt = bars(st_opt, opt_plain)


def tab(rows, names):
    h = ("<table><thead><tr><th>Circonscription</th><th class='n'>Inscrits</th><th class='n'>Écart à p̄</th>"
         "<th>En tête</th><th class='n'>Avance</th></tr></thead><tbody>")
    for r in sorted(rows, key=lambda r: r["d"]):
        h += (f"<tr><td>{names[r['d']]}</td><td class='n'>{r['inscrits']:,}</td><td class='n'>{r['ecart']:+.1f} %</td>"
              f"<td><span class='sw' style='background:{col(r['leader'])}'></span>{r['leader']}</td>"
              f"<td class='n'>{r['marge']:.1f} pts</td></tr>").replace(",", " ")
    return h + "</tbody></table>"


b = pd.read_parquet(PROC / "bureaux.parquet")
cand = pd.read_csv(ROOT / "data/raw/leg2024_t1_candidats.csv", sep=";", dtype=str)
cand["dep"] = cand["Code département"].map(lambda s: s.zfill(2) if s.isdigit() else s)
dep_name = cand[cand.dep == DEP]["Département"].iloc[0]
tot_sh = {f: 100 * gd[f"F_{f}"].sum() / gd[FC].sum().sum() for f in forces}

V = dict(
    DEP=DEP, DEP_NAME=dep_name, TG=TG, K=K, N_NODES=f"{n_nodes:,}".replace(",", " "),
    N_EDGES=f"{n_edges:,}".replace(",", " "), DEG=f"{deg:.1f}".replace(".", ","),
    INSCRITS=f"{int(gd.inscrits.sum()):,}".replace(",", " "), IDEAL=f"{ideal:,.0f}".replace(",", " "),
    LOW=f"{ideal * (1 - EPS):,.0f}".replace(",", " "), HIGH=f"{ideal * (1 + EPS):,.0f}".replace(",", " "),
    N_COMMUNES=str(gd.com.nunique()),
    SH_ENS=f"{tot_sh.get('ENS', 0):.1f}".replace(".", ","), SH_RN=f"{tot_sh.get('RN', 0):.1f}".replace(".", ","),
    SH_NFP=f"{tot_sh.get('NFP', 0):.1f}".replace(".", ","), SH_LR=f"{tot_sh.get('LR', 0):.1f}".replace(".", ","),
    LEAD_ENS=str(lead_counts.get("ENS", 0)), LEAD_RN=str(lead_counts.get("RN", 0)),
    LEAD_NFP=str(lead_counts.get("NFP", 0)), LEAD_LR=str(lead_counts.get("LR", 0)),
    N_TREE=str(T.number_of_edges()), N_BAL=str(len(balanced)),
    POP_B1=f"{pop_pair_before[1]:,}".replace(",", " "), POP_B2=f"{pop_pair_before[2]:,}".replace(",", " "),
    POP_A1=f"{pop_pair_after[1]:,}".replace(",", " "), POP_A2=f"{pop_pair_after[2]:,}".replace(",", " "),
    N_STEPS=f"{N_STEPS:,}".replace(",", " "), N_BURSTS=str(N_STEPS // BURST), BURST=str(BURST),
    WINS_REAL=str(sum(1 for r in st_real if r["leader"] == TG)), WINS_OPT=str(sum(1 for r in st_opt if r["leader"] == TG)),
    N_SHOW=str(N_SHOW), REACH=", ".join(str(tr["reach"]) for tr in traj),
    TRAJ_WINS=" ; ".join(f"essai {i + 1} : {tr['wins']}" for i, tr in enumerate(traj)),
    FIG_BUREAUX=fig_bureaux, FIG_GRAPH=fig_graph, FIG_RECOM_A=fig_recom_a, FIG_RECOM_B=fig_recom_b,
    FIG_RECOM_C=fig_recom_c, FIG_TRAJ=fig_traj, FIG_REAL=fig_real, FIG_OPT=fig_opt,
    FIG_BARS_REAL=fig_bars_real, FIG_BARS_OPT=fig_bars_opt, TAB_REAL=tab(st_real, real_plain), TAB_OPT=tab(st_opt, opt_plain),
)
summary = {k: v for k, v in V.items() if not k.startswith(("FIG_", "TAB_"))}
summary["real"] = [{k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items() if k != "shares"} | {"shares": {f: round(x, 1) for f, x in r["shares"].items() if x > 1}} for r in st_real]
summary["opt"] = [{k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items() if k != "shares"} | {"shares": {f: round(x, 1) for f, x in r["shares"].items() if x > 1}} for r in st_opt]
print(json.dumps(summary, ensure_ascii=False, indent=1, default=str))

tpl = (Path(__file__).parent / "methode_template.html")
if tpl.exists():
    html = tpl.read_text()
    for k, v in V.items():
        html = html.replace("{{" + k + "}}", str(v))
    (ROOT / "site" / "methode.html").write_text(html)
    print("site/methode.html écrit")
