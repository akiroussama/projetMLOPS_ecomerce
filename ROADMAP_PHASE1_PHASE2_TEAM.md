# Roadmap Equipe - Phase 1 + Phase 2

## Contexte
- Objectif: etre pret pour le prochain point avec Antoine en montrant une avance solide sur les phases 1 et 2.
- Periode cible: semaine du 2 au 6 mars 2026.
- Regle: ce dispatch est une proposition initiale et peut etre ajuste a chaque sprint planning.
- Membres actifs actuellement: `@akiroussama`, `@forgeros1993`, `@herymicka`.
- En attente d'acceptation repo: `@landroni` (ticket reserve: `#22`).

## Synthese charge (points) Phase 1 + 2
- `@akiroussama` (chaine informatique/API): 15 pts (`#10 #15 #19 #20 #21`)
- `@forgeros1993` (data engineering): 12 pts (`#9 #11 #12 #13`)
- `@herymicka` (taches simples): 4 pts (`#7 #16`)
- Reserve DS/ML Liviu (non assigne tant que l'invitation n'est pas acceptee): 13 pts (`#8 #14 #17 #18 #22`)

## Phase 1 (CP0 + CP1)
### Objectif
- Cadrage valide + pipeline baseline complet + API inference operationnelle.

### Tickets a finir
- `@herymicka`: `#7`, `#16`
- `@forgeros1993`: `#9`, `#11`, `#12`, `#13`
- `@akiroussama`: `#10`, `#15`
- `@landroni` (reserve): `#8`, `#14`

### Sequence anti-blocage
1. `#7` (kickoff metier)
2. `#8` et `#9` en parallele
3. `#10` (architecture/backlog)
4. Chaine data: `#11 -> #12 -> #13`
5. Chaine baseline/API: `#14 -> #15`
6. Smoke tests API: `#16`

## Phase 2 (CP2)
### Objectif
- MLflow tracking exploitable + API durcie (validation, auth, tests unitaires).

### Tickets a finir
- `@akiroussama`: `#19`, `#20`, `#21`
- `@landroni` (reserve): `#17`, `#18`, `#22`
- `@forgeros1993`: support execution/verification si besoin
- `@herymicka`: support verification/doc si besoin

### Sequence anti-blocage
1. API hardening: `#19 -> #20`
2. MLflow runs: `#17 -> #18` (en parallele de 1)
3. Tests unitaires API: `#21` (apres `#19` et `#20`)
4. Modele candidat: `#22` (apres `#18`)

## Risques et mitigation
- Risque: blocage si Liviu n'accepte pas l'invitation rapidement (`#8 #14 #17 #18 #22`).
  - Mitigation: fallback temporaire `@akiroussama` (ou `@forgeros1993` sur `#8`) si depassement de date intermediaire.
- Risque: surcharge Oussama sur la chaine API (`#19 #20 #21`).
  - Mitigation: garder `#21` assisté par Mickael pour execution de tests et documentation.
- Risque: transfert tardif entre chantiers data et ML.
  - Mitigation: point de synchro quotidien court entre Johan et Liviu des acceptance de `#13` et `#18`.
