# Découpage alternatif des circonscriptions (métropole)

Site : https://pierrealbr.github.io/decoupage-alternatif-circonscriptions-metropole/ · [méthode](https://pierrealbr.github.io/decoupage-alternatif-circonscriptions-metropole/methode.html)

Question : avec les mêmes voix, combien de circonscriptions une force peut-elle gagner (arriver en tête au 1er tour)
si on redessine les circonscriptions à son avantage ?

Scrutin testé : **législatives 2024, 1er tour**, France métropolitaine (539 circonscriptions ; étranger et outre-mer exclus).
Sur le site, les 38 circonscriptions d'outre-mer et des Français de l'étranger sont ajoutées sans être redécoupées (force en tête au 1er tour, ou député réellement élu au 2nd tour) pour afficher l'Assemblée complète de 577 sièges : `python src/build_hors_metropole.py` (produit aussi la proportionnelle intégrale sur 577 sièges).

## Pipeline

```bash
python -m venv .venv && .venv/bin/pip install pandas geopandas shapely networkx gerrychain pyarrow
.venv/bin/python src/prepare.py                       # unités géographiques + graphe d'adjacence
.venv/bin/python src/optimize.py --all --workers 10 --out max_eps20
.venv/bin/python src/report.py max_eps20              # vérification + synthèse
```

## Données (data/raw, data.gouv.fr)

- Résultats définitifs du 1er tour par bureau de vote (ministère de l'Intérieur).
- Liste des candidats du 1er tour : sert à retrouver la circonscription de chaque bureau.
- Contours approximatifs des bureaux de vote, méthode INSEE (version publiée par C. Rossi, juillet 2024).
- Contours des communes 2024 (Etalab) : repli pour les communes sans contours de bureaux.

## Hypothèses

- **Forces** : NFP = UG ; RN = RN + UXD ; ENS = ENS + HOR + UDI ; LR = LR. Les autres nuances (DVD, DVG, REG…)
  restent des forces séparées qui peuvent arriver en tête.
- Les voix d'une force dans une nouvelle circonscription = somme de ses voix dans les bureaux qui la composent
  (même si ces bureaux avaient des candidats différents).
- **Contraintes** de découpage : on reste dans chaque département avec le même nombre de circonscriptions ;
  chaque circonscription est d'un seul tenant ; écart d'**inscrits** à la moyenne départementale ≤ ±20 %
  (plafond légal, appliqué ici aux inscrits faute de population par bureau).
- **Raccordement** résultats ↔ contours : 98 % des bureaux sont raccordés directement. Les bureaux renumérotés
  (Bordeaux, une partie de Paris et de Marseille…) sont fondus en un reliquat communal réparti sur les contours
  restants ; quelques communes sans contours sont traitées comme une seule unité ; 4 fractions de communes
  nouvelles interdépartementales (1 947 inscrits) sont écartées.
- **Algorithme** : chaîne ReCom (GerryChain) + optimisation par « short bursts » (rafales de 5 pas),
  1 000 pas par siège × 3 essais indépendants par département et par force. C'est une heuristique : les
  maxima trouvés sont des **bornes basses** du vrai maximum.

## Résultats (run `max_eps20_v2`, ±20 % d'inscrits, graphe corrigé)

Circonscriptions où la force arrive **en tête au 1er tour** (métropole, 539) :

| Force | Découpage réel | Découpage optimisé pour elle | Gain |
|---|---|---|---|
| NFP | 147 | 276 | +129 |
| RN + UXD | 293 | 425 | +132 |
| ENS + HOR + UDI | 60 | 154 | +94 |
| LR | 15 | 20 | +5 |

Vérifications (`src/report.py`, `src/verify_geometry.py`) : les 384 plans (96 départements × 4 forces) sont connexes
dans le graphe et respectent ±20 % ; sur la carte, 2 à 4 circonscriptions par force gardent un fragment détaché
(≤ 1,4 km²), contre 21 pour le découpage réel tracé avec les mêmes contours.

Correction du 20/11/2026 : 12 % des contours INSEE sont en plusieurs morceaux ; une première version (`max_eps20`)
produisait des circonscriptions connexes dans le graphe mais éclatées sur la carte. `prepare.py` rattache désormais
les morceaux secondaires au bureau voisin et n'admet une adjacence qu'à partir de ~30 m de frontière commune.

Pages locales : `python3 -m http.server 8765 --directory site` puis http://localhost:8765 (cartes) et /methode.html.

## Modèle de second tour (`src/second_tour.py`)

Prédit le vainqueur final d'une circonscription à partir de ses voix du 1er tour : élection au 1er tour
(> 50 % des exprimés et ≥ 25 % des inscrits), qualification (2 premiers + ≥ 12,5 % des inscrits), désistements
observés en 2024 (NFP / ENS / divers gauche classés 3e face à l'extrême droite se retirent), puis reports de voix
estimés bureau par bureau (régression non négative) par configuration de second tour, avec une incertitude
gaussienne sur les parts (σ = 3,4 points, calibré par validation croisée).

Validation croisée par départements (5 groupes, découpage réel 2024, 539 circonscriptions) :
configuration de 2nd tour correcte 92,6 % ; vainqueur correct 90,0 % (91,5 % si la configuration est connue) ;
sièges prédits / réels : NFP 182 / 173, ENS 150 / 149, RN+UXD 136 / 140, LR 36 / 39, DVD 22 / 22.
Détail : `results/second_tour/validation.csv`.
