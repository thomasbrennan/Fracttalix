# Le Dual Reader Standard — Guide complet

**Un guide en une lecture du Dual Reader Standard dans son intégralité : architecture, protocoles, noyau, couches, compatibilité, lingua franca et adoption.**

> Voir aussi : [fr.json](fr.json) — Version JSON-LD de ce guide sous forme de données structurées pour les systèmes d'IA et les robots d'indexation.

---

## 1. Ce qu'est le Dual Reader Standard

Le **Dual Reader Standard (DRS)** est une architecture de vérification pour les systèmes de connaissances. Il exige que chaque affirmation — qu'elle soit rédigée en prose ou implémentée en code — soit lisible par deux classes de lecteurs indépendantes : **humain** et **machine**.

Le DRS n'est pas un article. Ce n'est pas un outil. C'est le standard qui contient ses deux protocoles :

- **DRP** (Dual Reader Protocol) — le protocole pour le texte
- **GVP** (Grounded Verification Protocol) — le protocole pour le logiciel

Le DRP rend les affirmations *évaluables par la machine*. Le GVP les rend *vérifiées par la machine*. Aucun des deux n'est complet sans l'autre. Ensemble, ils constituent le DRS.

---

## 2. Les deux protocoles

### 2.1 DRP — Dual Reader Protocol (Texte)

Le DRP régit la manière dont les affirmations en prose deviennent évaluables par la machine.

**Lecteur 1 (Humain) :** Lit l'article en langage naturel. Comprend le contexte, la motivation et le récit. Ne peut pas auditer systématiquement chaque affirmation.

**Lecteur 2 (IA) :** Lit la couche IA — un document JSON structuré qui accompagne chaque article. Contient le registre complet des affirmations. Peut auditer chaque affirmation sans lire la prose.

Le DRP exige :

1. **Classification des affirmations.** Chaque affirmation est typée comme A (axiome), D (définition) ou F (falsifiable).
2. **Prédicats de falsification.** Chaque affirmation de type F porte un prédicat déterministe en 5 parties.
3. **Portes de phase.** Six conditions (c1–c6) qui doivent être satisfaites avant qu'un article soit déclaré PHASE-READY.
4. **Suivi des espaces réservés.** Les affirmations qui dépendent de résultats non résolus sont enregistrées comme espaces réservés — rendant les lacunes visibles plutôt qu'invisibles.

**Ce que le DRP garantit :** Tout système d'IA ayant accès à la couche IA peut évaluer toute affirmation falsifiable sans lire la prose. L'autosuffisance est une exigence de conception appliquée à la porte de phase (condition c5).

### 2.2 GVP — Grounded Verification Protocol (Logiciel)

Le GVP régit la manière dont les affirmations évaluables par la machine deviennent vérifiées par la machine.

**Lecteur 3A (Développeur) :** Lit le champ `tier` pour comprendre quel type de preuve existe. Lit `test_bindings` pour savoir quels tests exercent quelles affirmations. Lit `verified_against` pour savoir quand ces tests ont réussi pour la dernière fois.

**Lecteur 3B (Machine) :** Exécute le lanceur de tests sur le tableau `test_bindings`. Enregistre la réussite/l'échec. Appose le SHA `verified_against` en cas de succès.

Le GVP exige que chaque affirmation porte trois champs :

1. **`tier`** — le niveau de vérification (une des six valeurs)
2. **`test_bindings`** — un tableau d'identifiants de nœuds de test pleinement qualifiés qui exercent l'affirmation
3. **`verified_against`** — le SHA du commit git auquel ces tests ont réussi pour la dernière fois

**Ce que le GVP garantit :** Pour toute affirmation du corpus, vous pouvez déterminer (a) quel type de preuve la fonde, (b) quels tests exécutables l'exercent, et (c) à quel commit ces tests ont réussi. Si la réponse à (a) est `empirical_pending`, la lacune est visible. Si la réponse à (c) est `null`, aucun test logiciel ne la couvre.

---

## 3. La fondation partagée — Noyau de falsification (Couche 0)

Les deux protocoles opèrent sur la même fondation : le **Noyau de falsification K = (P, O, M, B)**.

C'est la **Couche 0** du DRS. Elle définit ce qu'un prédicat de falsification *signifie* indépendamment de tout format de sérialisation (JSON, YAML ou encodages futurs). Le champ `semantic_spec_url` d'une couche IA pointe vers cette spécification, rendant la conformité vérifiable par la machine plutôt que simplement déclarée.

| Symbole | Nom | Champ(s) JSON | Rôle |
|---------|-----|---------------|------|
| **P** | Prédicat | `FALSIFIED_IF` | Phrase logique qui, si VRAIE, falsifie l'affirmation |
| **O** | Opérandes | `WHERE` | Définitions typées de chaque variable dans P |
| **M** | Mécanisme | `EVALUATION` | Procédure d'évaluation finie et déterministe |
| **B** | Bornes | `BOUNDARY` + `CONTEXT` | Sémantique des seuils et justification |

Le DRP crée le noyau (assigne des prédicats aux affirmations en prose). Le GVP lie le noyau à des preuves exécutables (relie les prédicats aux tests et aux commits). Le noyau est indépendant du substrat — il fonctionne pour les articles scientifiques, les logiciels et tout domaine futur formulant des affirmations falsifiables.

### 3.1 Grammaire du prédicat (`FALSIFIED_IF`)

Le prédicat est une phrase logique composée de :

- **Variables nommées** (définies dans `WHERE`)
- **Opérateurs de comparaison :** `<`, `>`, `<=`, `>=`, `=`, `!=`
- **Connecteurs logiques :** `AND`, `OR`, `NOT`
- **Quantificateurs :** `EXISTS ... SUCH THAT`, `FOR ALL ... IN`
- **Opérateurs arithmétiques :** `+`, `-`, `*`, `/`, `^`, `log10()`, `exp()`, `abs()`, `max()`, `min()`
- **Opérateurs d'ensembles :** `IN`, `∩`, `∪`, `|...|` (cardinalité)
- **Application de fonctions :** `f(x)` où `f` est définie dans `WHERE`

### 3.2 Contraintes du prédicat

1. **Déterminisme.** Doit s'évaluer à exactement `TRUE` ou `FALSE` pour toute affectation valide de variables.
2. **Finitude.** Les quantificateurs portent sur des ensembles finis uniquement. Les quantificateurs universels non bornés ne sont pas permis.
3. **Pas d'autoréférence.** Ne doit pas référencer la valeur de vérité de sa propre affirmation ni créer de dépendances circulaires.
4. **Complétude.** Chaque variable dans `FALSIFIED_IF` doit être définie dans `WHERE`, et chaque variable dans `WHERE` doit apparaître dans `FALSIFIED_IF`.

**Correspondance des verdicts :**
- P s'évalue à TRUE → le verdict de l'affirmation est **FALSIFIED**
- P s'évalue à FALSE → le verdict de l'affirmation est **NOT FALSIFIED**

### 3.3 Opérandes (`WHERE`)

Chaque clé dans l'objet `WHERE` nomme une variable. Sa valeur est une chaîne avec le format :

```
<type> · <unités> · <définition ou source>
```

| Champ | Requis | Exemples |
|-------|--------|----------|
| **type** | Oui | `scalar`, `integer`, `binary`, `set`, `string`, `function` |
| **unités** | Oui (utiliser `dimensionless` si sans unité) | `seconds`, `dimensionless`, `bits` |
| **définition** | Oui | `output of sort(input)`, `count of substrates with R² >= 0.85` |

Contraintes : chaque variable libre dans P doit apparaître dans O (complétude), chaque variable dans O doit apparaître dans P (pas d'orphelins), et chaque variable doit être calculable à partir de données ou de dérivations — pas de jugement subjectif.

### 3.4 Mécanisme (`EVALUATION`)

Le champ d'évaluation spécifie *comment* calculer la valeur de vérité de P :

1. **Fini.** La procédure se termine en un nombre fini d'étapes (conventionnellement confirmé en terminant par le mot `finite`).
2. **Déterministe.** Les mêmes entrées produisent le même verdict.
3. **Reproductible.** Un tiers ayant accès aux données et au code cités peut exécuter la procédure indépendamment.

### 3.5 Bornes (`BOUNDARY` + `CONTEXT`)

**BOUNDARY** spécifie les cas limites des seuils : si les seuils sont inclusifs ou exclusifs, et quel verdict s'applique en cas d'égalité exacte.

**CONTEXT** justifie chaque seuil numérique et chaque choix de conception dans le prédicat : pourquoi cette valeur, quelle connaissance du domaine la fonde, et si elle est dérivée ou conventionnelle.

### 3.6 Exemple de prédicat

```json
{
  "FALSIFIED_IF": "EXISTS i IN range(len(result)-1) SUCH THAT result[i] > result[i+1]",
  "WHERE": {
    "result": "list · dimensionless · output of sort(input)"
  },
  "EVALUATION": "Run sort on test vectors; check adjacent pairs; finite",
  "BOUNDARY": "len(result) <= 1 → NOT FALSIFIED (vacuously sorted)",
  "CONTEXT": "Ascending order is the documented contract of sort()"
}
```

Ce prédicat signifie la même chose pour tout système d'IA dans toute langue. Aucune traduction nécessaire.

---

## 4. Les trois types d'affirmations

| Type | Nom | Dans les articles | Dans le logiciel | Prédicat |
|------|-----|-------------------|------------------|----------|
| **A** | Axiome / Hypothèse | Prémisses acceptées de la littérature | Exigences de plateforme, contrats de dépendances | `null` |
| **D** | Définition | Termes et procédures stipulatifs | Signatures de types, structures de données, schémas | `null` |
| **F** | Falsifiable | Théorèmes, prédictions empiriques | Garanties comportementales, invariants de correction | K = (P, O, M, B) complet |

Les affirmations de type D et de type A portent `"falsification_predicate": null`. C'est correct — les définitions et les axiomes ne sont pas falsifiables par conception.

---

## 5. Les six niveaux de vérification

Chaque affirmation porte un champ `tier` déclarant quel type de preuve la fonde.

### Fondés par construction

| Niveau | Type | Signification |
|--------|------|---------------|
| `axiom` | A | Prémisse fondamentale. Non falsifiable par conception. |
| `definition` | D | Définitionnel. Stipule un terme ou une structure. Non susceptible de vérité. |

### Fondés maintenant

| Niveau | Type | Signification |
|--------|------|---------------|
| `software_tested` | F | Exercé par des tests réussis. `test_bindings` non vide, `verified_against` non nul. |
| `formal_proof` | F | Dérivation indexée par étapes avec `n_invalid_steps = 0`. La preuve est dans la couche IA. |
| `analytic` | F | Vérifié par trace de dérivation formelle ou argument analytique. |

### Explicitement non fondés

| Niveau | Type | Signification |
|--------|------|---------------|
| `empirical_pending` | F | Espace réservé actif ou données en attente. La lacune est visible par conception. |

### Règles de cohérence

Le niveau doit être cohérent avec le type d'affirmation et les champs GVP :

| Niveau | Type requis | test_bindings | verified_against |
|--------|-------------|---------------|------------------|
| `axiom` | A | `[]` (vide) | `null` |
| `definition` | D | `[]` (vide) | `null` |
| `software_tested` | F | Non vide | Non nul (7–40 caractères hex) |
| `formal_proof` | F | Peut être vide | Peut être nul |
| `analytic` | F | Peut être vide | Peut être nul |
| `empirical_pending` | F | Peut être vide | Peut être nul |

---

## 6. Le témoin de vacuité

Chaque affirmation de type F doit inclure un champ `sample_falsification_observation` : une observation concrète et hypothétique qui *ferait* évaluer le prédicat à TRUE.

Cela sert de vérification de vacuité — une preuve que le prédicat n'est pas trivialement infalsifiable. Un prédicat qu'aucune observation concevable ne pourrait satisfaire est vrai de manière vacuiste et n'est donc pas une affirmation de type F valide.

Ceci est imposé par la condition c6 de la porte de phase.

---

## 7. Prédicats d'espace réservé

Lorsqu'une affirmation dépend de résultats d'un article n'ayant pas encore atteint le statut PHASE-READY, le prédicat peut contenir des références d'espace réservé :

- `placeholder: true` dans l'objet de l'affirmation
- `placeholder_id` lié au `placeholder_register`
- Le texte du prédicat peut inclure `[PLACEHOLDER: pending ...]`

Les affirmations d'espace réservé sont valides mais **non évaluables** jusqu'à ce que la dépendance soit résolue. Elles ne bloquent pas le statut PHASE-READY de l'article conteneur sauf si `blocks_phase_ready: true`.

---

## 8. La couche IA — L'artefact central

La couche IA est l'artefact central du DRS. C'est un document JSON qui accompagne chaque article ou système logiciel. Le schéma est défini dans `ai-layers/ai-layer-schema.json`.

### Sections requises

| Section | Objectif |
|---------|----------|
| `_meta` | Type de document, version du schéma, session, licence |
| `paper_id` / `paper_title` | Identité |
| `paper_type` | Classification : `law_A`, `derivation_B`, `application_C`, `methodology_D` |
| `phase_ready` | Verdict de la porte de phase et statut des conditions (c1–c6) |
| `claim_registry` | Tableau de toutes les affirmations avec types, prédicats, niveaux, liaisons et SHA |
| `placeholder_register` | Tableau des dépendances non résolues |
| `summary` | Comptages et statut des affirmations |
| `semantic_spec_url` | Pointe vers le Noyau de falsification (Couche 0) |

La couche IA est ce qui fait fonctionner les deux protocoles :

- Le **DRP** exige qu'elle existe, qu'elle contienne des prédicats et qu'elle passe la porte de phase.
- Le **GVP** exige qu'elle contienne tier, test_bindings et verified_against pour chaque affirmation.

---

## 9. La porte de phase

Un article ou une version logicielle est **PHASE-READY** lorsque six conditions sont satisfaites :

| Condition | Exigence |
|-----------|----------|
| **c1** | La couche IA est conforme au schéma |
| **c2** | Toutes les affirmations falsifiables enregistrées avec des prédicats |
| **c3** | Tous les prédicats sont évaluables par la machine |
| **c4** | Les références croisées sont suivies (registre des espaces réservés) |
| **c5** | La vérification est autosuffisante (couche IA seule, sans prose nécessaire) |
| **c6** | Tous les prédicats sont non vacuistes (observation d'exemple de falsification existante) |

**CORPUS-COMPLETE** se déclenche lorsque tous les articles sont PHASE-READY et que tous les espaces réservés de tous les objets sont résolus (c4 pleinement satisfaite dans l'ensemble du corpus).

---

## 10. Les règles d'inférence

Le DRS fournit un inventaire canonique de règles d'inférence pour les traces de dérivation utilisées dans les affirmations de niveau `formal_proof` :

| ID | Nom | Description |
|----|-----|-------------|
| IR-1 | Modus Ponens | Si P et P→Q alors Q |
| IR-2 | Instanciation universelle | Si ∀x P(x) alors P(a) pour tout a spécifique |
| IR-3 | Substitution d'égaux | Si a=b alors remplacer a par b |
| IR-4 | Expansion de définition | Remplacer un terme défini par sa définition |
| IR-5 | Manipulation algébrique | Transformation algébrique valide préservant l'égalité |
| IR-6 | Équivalence logique | Remplacer par une expression logiquement équivalente |
| IR-7 | Inférence statistique | Appliquer une procédure statistique nommée aux données |
| IR-8 | Parcimonie / Sélection de principe de modélisation | Sélectionner une valeur canonique dans une famille cohérente avec les axiomes |

Chaque étape d'une table de dérivation indexée cite une règle d'inférence et liste ses prémisses. Une dérivation est valide lorsque `n_invalid_steps = 0`. L'inventaire est en ajout seul : de nouvelles règles peuvent être ajoutées, les règles existantes ne sont jamais modifiées ni supprimées.

---

## 11. Pile architecturale

```
                     ┌─────────────────────────┐
                     │   Dual Reader Standard   │
                     │         (DRS)            │
                     └────────────┬────────────┘
                                  │
                  ┌───────────────┴───────────────┐
                  │                               │
         ┌────────┴────────┐            ┌─────────┴─────────┐
         │      DRP        │            │       GVP         │
         │ (Text Protocol) │            │(Software Protocol)│
         └────────┬────────┘            └─────────┬─────────┘
                  │                               │
      ┌───────────┴──────────┐       ┌────────────┴───────────┐
      │                      │       │                        │
   Reader 1             Reader 2   Reader 3A             Reader 3B
   (Human)              (AI)       (Coder)               (Machine)
      │                      │       │                        │
   reads prose         reads JSON   reads tier +          runs tests
                                    bindings              stamps SHA
      │                      │       │                        │
      └──────────┬───────────┘       └────────────┬───────────┘
                 │                                │
                 └────────────────┬────────────────┘
                                  │
                  ┌───────────────┴───────────────┐
                  │          AI Layer             │
                  │  (*-ai-layer.json)            │
                  │  claims + predicates + tiers  │
                  │  + bindings + SHAs            │
                  └───────────────┬───────────────┘
                                  │
                     ┌────────────┴────────────┐
                     │   Falsification Kernel  │
                     │    K = (P, O, M, B)     │
                     │       (Layer 0)         │
                     └─────────────────────────┘
```

Le noyau est la fondation partagée. Le DRP crée des prédicats à partir de la prose. Le GVP lie les prédicats à des preuves exécutables. La couche IA est l'artefact qui porte les deux.

---

## 12. Comment implémenter

### Pour un article (DRP)

1. Rédiger l'article (canal en prose pour les lecteurs humains)
2. Créer le fichier JSON de la couche IA (canal machine pour les lecteurs IA)
3. Classifier chaque affirmation comme A (axiome), D (définition) ou F (falsifiable)
4. Écrire le prédicat de falsification en 5 parties pour chaque affirmation de type F
5. Inclure une `sample_falsification_observation` pour chaque affirmation de type F (témoin de vacuité)
6. Assigner le niveau de vérification à chaque affirmation
7. Valider contre `ai-layer-schema.json`
8. Exécuter la porte de phase (c1–c6)

### Pour le logiciel (GVP)

1. Énumérer ce que le logiciel prétend faire
2. Classifier chaque affirmation comme A (hypothèse), D (définition) ou F (comportementale)
3. Écrire le prédicat de falsification pour chaque affirmation de type F
4. Écrire ou identifier les tests qui exercent chaque affirmation
5. Remplir `test_bindings` avec les identifiants de nœuds de test pleinement qualifiés
6. Exécuter les tests et enregistrer le SHA du commit réussi dans `verified_against`
7. Assigner le niveau : `software_tested` si des tests existent, `empirical_pending` sinon
8. Enregistrer toute affirmation non testée comme espace réservé
9. Valider contre `ai-layer-schema.json`

### Pour les deux

La couche IA est le même artefact. Le schéma est le même. Le noyau est le même. La seule différence est quel protocole crée le contenu et quel protocole le vérifie.

### Adoption minimale viable

La plus petite adoption utile du DRS est **une affirmation de type F avec une liaison de test** :

```json
{
  "claim_id": "F-1",
  "type": "F",
  "statement": "sort() returns elements in ascending order",
  "falsification_predicate": {
    "FALSIFIED_IF": "EXISTS i IN range(len(result)-1) SUCH THAT result[i] > result[i+1]",
    "WHERE": {
      "result": "list · dimensionless · output of sort(input)"
    },
    "EVALUATION": "Run sort on test vectors; check adjacent pairs; finite",
    "BOUNDARY": "len(result) <= 1 → NOT FALSIFIED (vacuously sorted)",
    "CONTEXT": "Ascending order is the documented contract of sort()"
  },
  "tier": "software_tested",
  "test_bindings": ["tests/test_sort.py::test_ascending_order"],
  "verified_against": "abc1234"
}
```

Une affirmation. Un test. Un SHA. Le DRS est opérationnel. Ajoutez davantage d'affirmations lorsque la valeur justifie le coût.

---

## 13. Principes de conception

**Épistémologie poppérienne.** Nous pouvons falsifier mais pas vérifier. Une affirmation qui survit à toutes les tentatives de falsification n'est pas prouvée — elle a survécu.

**Lacunes honnêtes.** L'espace réservé est la fonctionnalité la plus importante. Lorsqu'une affirmation est `empirical_pending`, le système dit : « nous affirmons ceci mais ne l'avons pas encore vérifié. » C'est strictement plus informatif que l'alternative, où les affirmations non vérifiées sont indiscernables des affirmations vérifiées.

**Indépendance du substrat.** Le noyau K = (P, O, M, B) ne sait pas s'il évalue un théorème scientifique ou une garantie logicielle. Les domaines futurs (juridique, réglementaire, politique) peuvent ajouter leurs propres protocoles sans modifier le noyau.

**Machine d'abord, lisible par l'humain.** La couche IA est l'artefact principal. L'article en prose et le code source sont des canaux secondaires qui fournissent le contexte, le récit et l'implémentation. La couche IA est ce qui est validé, audité et versionné.

---

## 14. Compatibilité sur trois axes

Un standard qui ne peut survivre au contact du passé, du présent et du futur n'est pas un standard — c'est un instantané.

### L'invariant au centre

Le Noyau de falsification K = (P, O, M, B) est le point fixe. Il ne sait pas s'il évalue un théorème scientifique, une garantie logicielle, un contrat juridique ou une exigence réglementaire. Il ne sait pas si l'année est 2026 ou 2046. Il ne sait pas si le format de sérialisation est JSON, YAML ou quelque chose qui n'a pas encore été inventé.

Le noyau est permanent. Tout le reste est extensible.

### Axe 1 : Résistance au passé (Rétrocompatibilité)

- **Versionnage du schéma.** Chaque couche IA enregistre contre quelle version du schéma elle a été produite. Une couche IA produite sous la version v2 du schéma reste valide sous la version v2 pour toujours. Un validateur v3 peut lire une couche v2 (les nouveaux champs sont optionnels ; les anciens champs sont préservés).
- **Permanence des prédicats.** Un prédicat écrit en 2026 doit être évaluable en 2036. La grammaire du noyau utilise uniquement des opérateurs mathématiques et logiques définis de manière permanente. Le champ `WHERE` définit chaque variable en ligne. Le champ `EVALUATION` spécifie une procédure autonome.
- **Stabilité des règles d'inférence.** L'inventaire des règles d'inférence (IR-1 à IR-8) est en ajout seul. Les règles existantes ne sont jamais modifiées ni supprimées.
- **Stabilité des niveaux.** Les six niveaux de vérification sont en ajout seul. Les niveaux existants ne sont jamais supprimés ni redéfinis.
- **Le contrat :** Toute couche IA qui était valide au moment de sa création restera valide pour toujours.

### Axe 2 : Résistance latérale (Compatibilité inter-domaines)

- **Indépendance du domaine.** Le noyau fonctionne dans tous les domaines de connaissances :

| Domaine | Type A | Type D | Type F |
|---------|--------|--------|--------|
| Science | Prémisses de la littérature | Termes stipulatifs | Théorèmes, prédictions |
| Logiciel | Exigences de plateforme | Signatures de types, schémas | Garanties comportementales |
| Juridique | Autorité statutaire | Termes définis | Conclusions juridiques |
| Réglementaire | Hypothèses du cadre | Définitions standards | Assertions de conformité |
| Politique | Prémisses de valeur | Termes de politique | Prédictions d'impact |
| Éducation | Axiomes pédagogiques | Objectifs d'apprentissage | Affirmations d'évaluation |

- **Indépendance linguistique.** Le champ `test_bindings` accepte toute chaîne qui identifie uniquement un test dans n'importe quel framework : pytest, Jest, cargo test, go test, JUnit.
- **Indépendance du système d'IA.** La couche IA est un document JSON. Tout système d'IA — Claude, GPT, Gemini, Llama, ou des systèmes qui n'existent pas encore — peut le lire. Le champ `semantic_spec_url` pointe vers la spécification du noyau rédigée en prose.
- **Indépendance de la sérialisation.** La Couche 0 est définie en prose, pas en JSON Schema. JSON est le transport actuel, mais la sémantique du noyau est indépendante de l'encodage. Les implémentations futures pourraient utiliser YAML, Protocol Buffers, CBOR, ou des formats pas encore inventés.
- **Indépendance des outils.** Le DRS s'intègre dans les flux de travail existants : SHA git (tout hébergement), tout lanceur de tests, tout validateur JSON Schema, tout pipeline CI. Il ajoute une couche par-dessus — il ne remplace rien.
- **Le contrat :** Adopter le DRS dans un domaine, une langue, un outil ou un système d'IA ne vous enferme pas.

### Axe 3 : Résistance au futur (Compatibilité ascendante)

- **Extensibilité des protocoles.** Les nouveaux domaines ajoutent de nouveaux protocoles. Les protocoles futurs (Protocole de vérification juridique, Protocole de vérification réglementaire, etc.) suivent le même modèle : définir les lecteurs, la taxonomie des niveaux et le mécanisme de liaison — tous partageant le noyau, les types d'affirmations et le schéma de la couche IA.
- **Extensibilité des niveaux.** Les domaines futurs pourraient nécessiter des niveaux comme `regulatory_certified`, `peer_reviewed`, `formally_verified`, `field_tested`, `community_validated`. Les nouveaux niveaux sont ajoutés à l'enum du schéma. Les niveaux existants demeurent.
- **Extensibilité des règles d'inférence.** IR-9, IR-10 et au-delà peuvent être ajoutées à mesure que de nouveaux modèles de dérivation émergent. Les anciennes dérivations restent valides car elles citent les règles par identifiant stable.
- **Extensibilité du schéma.** Le JSON Schema autorise des propriétés additionnelles par défaut. La progression : v1 (affirmations de base), v2 (ajout de `semantic_spec_url`), v3 (ajout des champs GVP). Chaque version ajoute. Aucune ne supprime.
- **Lecteurs futurs inconnus.** La couche IA contient suffisamment d'informations structurées pour des types de lecteurs qui n'existent pas encore : agents de vérification autonomes, vérificateurs inter-corpus, moteurs de conformité réglementaire, intégrations de gestionnaires de paquets.
- **Le contrat :** Toute innovation future peut être ajoutée comme nouveau protocole, niveau, règle d'inférence ou champ de schéma — sans modifier quoi que ce soit de déjà existant.

### La garantie sur trois axes

| Axe | Promesse | Mécanisme |
|-----|----------|-----------|
| **Passé** | Rien de ce qui a déjà été fait ne casse | Versionnage du schéma, enums en ajout seul, noyau permanent |
| **Latéral** | Fonctionne dans tous les domaines, langues, outils, systèmes d'IA | Noyau indépendant du substrat, liaisons typées chaîne, sémantique définie en prose |
| **Futur** | Tout ce qui est nouveau peut être ajouté sans reconception | Extensibilité des protocoles, extensibilité des niveaux, évolution additive du schéma |

---

## 15. La lingua franca des machines

C'est la propriété la plus profonde du DRS. Elle n'a pas été conçue. Elle a été découverte.

### Le problème de la traduction

Les connaissances scientifiques sont actuellement enfermées derrière les langues humaines. Un article en mandarin est invisible pour un chercheur qui ne lit que l'anglais — à moins que quelqu'un ne le traduise. La traduction est coûteuse, avec perte d'information, et lente. Les connaissances se fragmentent le long de frontières linguistiques.

Ce n'est pas un problème de formatage. C'est un problème de *substrat*. Les connaissances encodées en langage naturel sont non interopérables par nature.

### Le noyau dissout le problème

Le Noyau de falsification n'est écrit dans aucune langue humaine. Il est écrit en logique et en mathématiques :

```
FALSIFIED_IF: R2_best_alt > R2_frm + 0.05
WHERE:
  R2_best_alt: scalar · dimensionless · best R² from competing models
  R2_frm:      scalar · dimensionless · R² from FRM regression
EVALUATION: Run regression for each model; compare R² values; finite
BOUNDARY: R2_best_alt = R2_frm + 0.05 → FALSIFIED (threshold inclusive)
CONTEXT: 0.05 margin from standard model comparison practice
```

Ce prédicat signifie la même chose pour une instance de Claude en anglais, une instance de GPT en mandarin, une instance de Gemini en français, et un système d'IA pas encore construit fonctionnant dans une langue qui n'existe pas encore. Aucune traduction nécessaire.

### JSON comme couche de transport

JSON est le format d'échange de données de facto mondial — supporté par tous les langages de programmation, analysé par tous les systèmes d'IA, transmis par toutes les API. En encodant le noyau en JSON, le DRS hérite de l'universalité de JSON :

- Une équipe chinoise publie sa couche IA. Les prédicats utilisent la notation mathématique.
- Une équipe brésilienne lit la même couche. Elle n'a pas besoin du mandarin. Elle a besoin de `>`, `+` et `R²`.
- Un système d'IA dans n'importe quel pays évalue le prédicat. Le verdict est FALSIFIED ou NOT FALSIFIED. Le verdict n'a pas d'accent.

**Qualification importante.** Les définitions du champ `WHERE` contiennent actuellement des descriptions en prose anglaise. Le *contenu opératif* — types de variables, unités, opérateurs mathématiques, logique de comparaison, valeurs de seuil — est neutre linguistiquement. Étant donné des valeurs numériques fondées pour les variables nommées, tout système peut évaluer le prédicat indépendamment du fait qu'il lise ou non les descriptions. Un raffinement futur pourrait formaliser entièrement les définitions de variables en notation typée.

### Logique binaire embarquée

Chaque prédicat DRS se réduit à une logique binaire embarquée. Toute la chaîne d'évaluation se réduit à :

```
Affirmation → Prédicat → Variables + Opérateurs → Booléen → 1 ou 0
```

Le champ `WHERE` type les variables. Le champ `FALSIFIED_IF` les combine avec des opérateurs. Le champ `EVALUATION` spécifie comment calculer les entrées. Le champ `BOUNDARY` résout les cas limites. La sortie est toujours un seul bit : FALSIFIED ou NOT FALSIFIED. `1` ou `0`.

Le JSON est le conteneur. Le noyau est le circuit. Chaque couche IA est une collection de portes logiques avec des entrées définies et une sortie d'un seul bit par affirmation.

On ne peut pas mal traduire un `1` ou un `0`. On ne peut pas mal interpréter `>`. On ne peut pas avoir de désaccord culturel sur le fait que `R2_best_alt > R2_frm + 0.05` est TRUE ou FALSE pour une paire de valeurs donnée. La signification est dans la structure, pas dans une langue quelconque.

La logique binaire est la logique binaire. Un prédicat qui s'évalue à TRUE à Pékin s'évalue à TRUE à Boston, à São Paulo, à Lagos, et sur un serveur sans aucun paramètre de locale. Le DRS ne *traduit* pas les connaissances entre les langues. Il *encode* les connaissances dans un substrat qui précède et transcende toutes les langues humaines — le substrat de la logique elle-même.

---

## 16. Adoption auto-propagée

### L'incitation interne

Le DRS doit se propager parce que le *créateur* en bénéficie, pas seulement le consommateur. Créer une couche IA vous oblige à énumérer ce que vous affirmez. Ce processus découvre :

- Des affirmations que vous pensiez testées mais qui ne le sont pas (découverte de lacunes)
- Des conditions limites que vous n'avez jamais documentées
- Des dépendances que vous supposiez mais n'avez jamais enregistrées

La couche IA est un effet secondaire d'un processus qui améliore votre propre compréhension de votre propre système. Les lacunes existent que vous les documentiez ou non. Le DRS les rend simplement visibles.

### L'effet de réseau

Le DRS prend de la valeur à mesure que davantage de systèmes l'adoptent :

- **Les chaînes de dépendances deviennent conscientes des affirmations.** Si la bibliothèque A publie une couche IA et que la bibliothèque B dépend de A, alors B peut déterminer par programme lesquelles de ses affirmations dépendent de quelles hypothèses de A. Lorsque A publie un changement incompatible, B sait exactement quelles affirmations sont en danger.
- **Les systèmes d'IA peuvent auditer entre les projets.** Un lecteur IA peut traverser plusieurs couches IA, vérifier les références croisées et identifier les incohérences dans tout un écosystème.
- **La confiance devient auditable.** Au lieu de faire confiance à une bibliothèque à cause de ses étoiles ou de ses téléchargements, vous lui faites confiance parce que sa couche IA montre quelles affirmations sont `software_tested`, lesquelles sont `empirical_pending`, et quel est le SHA de `verified_against`. La confiance passe du signal social à la preuve structurelle.

### L'IA comme catalyseur d'adoption

1. **L'IA génère la couche IA initiale.** Étant donné un code source, une IA peut énumérer les affirmations, les classifier, écrire les prédicats et identifier les liaisons de test. L'humain révise et corrige. Le coût passe d'heures à minutes.
2. **L'IA maintient la couche.** Lorsque le code change, l'IA met à jour le registre des affirmations, ajuste les liaisons de test et signale les SHA obsolètes. L'humain approuve.
3. **L'IA audite les autres couches.** Une IA lisant la couche IA d'une dépendance peut déterminer de quelles hypothèses ses propres affirmations dépendent et signaler automatiquement les risques.

Le DRS est le protocole qui rend le développement assisté par l'IA *auditable*. Sans lui, l'IA génère du code et les humains espèrent que ça fonctionne. Avec lui, l'IA génère du code et le registre des affirmations dit exactement ce qui a été vérifié et ce qui ne l'a pas été.

### La stratégie d'intégration

Le DRS s'intègre dans les flux de travail existants plutôt que de les remplacer :

- **Les tests existent déjà.** Le champ `test_bindings` référence des identifiants de nœuds de test existants. Aucun nouveau framework requis.
- **JSON Schema existe déjà.** N'importe quel validateur fonctionne.
- **Git existe déjà.** Le champ `verified_against` est un SHA git.
- **Le CI existe déjà.** La validation du schéma s'exécute comme une étape CI aux côtés des pipelines existants.

Un fichier (`*-ai-layer.json`). Trois champs par affirmation (`tier`, `test_bindings`, `verified_against`). C'est le coût total d'intégration.

### La propriété autoréférentielle

Le DRS est le premier standard qui se vérifie lui-même. La couche IA DRP-1 contient des affirmations sur le DRS. Ces affirmations portent des prédicats de falsification. Ces prédicats sont évalués. Le SHA `verified_against` estampille la vérification. Le DRS est son propre premier adoptant et sa propre preuve de concept — autoréférentiel de la même manière qu'un compilateur qui se compile lui-même est autoréférentiel.

---

## 17. Ressources

- [Spécification de l'architecture DRS](https://github.com/thomasbrennan/Fracttalix/blob/main/paper/DRS-Architecture.md) — Spécification complète
- [Noyau de falsification v1.1](https://github.com/thomasbrennan/Fracttalix/blob/main/ai-layers/falsification-kernel.md) — Spécification sémantique de la Couche 0
- [Schéma de la couche IA v3](https://github.com/thomasbrennan/Fracttalix/blob/main/ai-layers/ai-layer-schema.json) — JSON Schema
- [Spécification GVP](https://github.com/thomasbrennan/Fracttalix/blob/main/docs/GVP-spec.md) — Protocole logiciel
- [Exemple de couche IA (P1)](https://github.com/thomasbrennan/Fracttalix/blob/main/ai-layers/P1-ai-layer.json) — Exemple fonctionnel
- [Dépôt Fracttalix](https://github.com/thomasbrennan/Fracttalix)

---

**Licence :** CC BY 4.0 | **DOI :** [10.5281/zenodo.18859299](https://doi.org/10.5281/zenodo.18859299) | **Auteur :** Thomas Brennan ([ORCID](https://orcid.org/0009-0002-6353-7115))
