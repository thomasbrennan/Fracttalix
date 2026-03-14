# Comment implémenter le Dual Reader Standard

**Guide complet pour construire des assertions vérifiables par machine pour les articles scientifiques, les logiciels et les systèmes de connaissances.**

> Voir aussi : [fr.json](fr.json) — Version en données structurées JSON-LD de ce guide pour les systèmes d'IA et les robots d'indexation.

---

## Qu'est-ce que le Dual Reader Standard ?

Le **Dual Reader Standard (DRS)** est une architecture de vérification pour les systèmes de connaissances. Chaque assertion — qu'elle soit rédigée en prose ou implémentée en code — doit être lisible par deux classes de lecteurs indépendantes : **humain** et **machine**.

Le DRS comprend deux protocoles :

| Protocole | Domaine | Fonction |
|-----------|---------|----------|
| **DRP** (Dual Reader Protocol) | Texte / Articles | Rend les assertions en prose évaluables par machine via des prédicats de falsification en 5 parties |
| **GVP** (Grounded Verification Protocol) | Logiciel / Code | Rend les assertions évaluables par machine vérifiées par machine via des liaisons de tests et des preuves ancrées sur des commits |

### Les trois lecteurs

| Lecteur | Canal | Lit | Format |
|---------|-------|-----|--------|
| **Humain** | Prose | L'article | Langage naturel |
| **IA** | JSON | La couche IA | Registre d'assertions structuré |
| **CI / exécuteur de tests** | Exécutable | Liaisons de tests | Identifiants de nœuds de test + SHA du commit |

---

## Le noyau de falsification K = (P, O, M, B)

Le fondement commun des deux protocoles. Chaque assertion de type F (falsifiable) porte un prédicat déterministe qui s'évalue à exactement l'un de deux verdicts : **FALSIFIÉ** ou **NON FALSIFIÉ**.

| Symbole | Nom | Champ JSON | Rôle |
|---------|-----|------------|------|
| **P** | Prédicat | `FALSIFIED_IF` | Énoncé logique qui, s'il est VRAI, falsifie l'assertion |
| **O** | Opérandes | `WHERE` | Définitions typées de chaque variable du prédicat |
| **M** | Mécanisme | `EVALUATION` | Procédure d'évaluation finie et déterministe |
| **B** | Bornes | `BOUNDARY` + `CONTEXT` | Sémantique des seuils et justification |

### Exemple de prédicat

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

### Contraintes sur les prédicats

- **Déterminisme** : Doit s'évaluer à exactement VRAI ou FAUX
- **Finitude** : Les quantificateurs portent uniquement sur des ensembles finis
- **Pas d'autoréférence** : Pas de dépendances circulaires entre prédicats
- **Complétude** : Chaque variable dans `FALSIFIED_IF` est définie dans `WHERE`, et réciproquement

---

## Les trois types d'assertions

| Type | Nom | Description | Prédicat requis ? |
|------|-----|-------------|-------------------|
| **A** | Axiome | Prémisses fondamentales — non falsifiables par conception | Non (`null`) |
| **D** | Définition | Définitions stipulatives — non susceptibles de vérité | Non (`null`) |
| **F** | Falsifiable | Assertions testables avec des prédicats déterministes | Oui — K = (P, O, M, B) complet |

---

## Les six niveaux de vérification

Chaque assertion porte un champ `tier` déclarant quel type de preuve la fonde.

### Fondé par construction

| Niveau | S'applique à | Signification |
|--------|-------------|---------------|
| `axiom` | Type A | Fondamental, non falsifiable par conception |
| `definition` | Type D | Définitionnel, aucun prédicat requis |

### Fondé maintenant

| Niveau | S'applique à | Signification |
|--------|-------------|---------------|
| `software_tested` | Type F | Exercé par des tests passants. `test_bindings` non vide, `verified_against` SHA non nul |
| `formal_proof` | Type F | Dérivation indexée par étapes avec `n_invalid_steps = 0` |
| `analytic` | Type F | Vérifié par trace de dérivation formelle ou argument analytique |

### Explicitement non fondé

| Niveau | S'applique à | Signification |
|--------|-------------|---------------|
| `empirical_pending` | Type F | Espace réservé actif ou données en attente. La lacune est visible par conception |

---

## La couche IA — L'artefact central

La couche IA est un document JSON qui accompagne chaque article ou système logiciel. C'est sur elle que les deux protocoles opèrent.

### Sections requises

| Section | Fonction |
|---------|----------|
| `_meta` | Type de document, version du schéma, session, licence |
| `paper_id` | Identifiant unique |
| `paper_title` | Titre lisible par l'humain |
| `paper_type` | `law_A`, `derivation_B`, `application_C` ou `methodology_D` |
| `phase_ready` | Verdict de la porte de phase et état des conditions (c1–c6) |
| `claim_registry` | Tableau de toutes les assertions avec types, prédicats, niveaux, liaisons |
| `placeholder_register` | Tableau des dépendances non résolues |

### Schéma

Le schéma de la couche IA est disponible ici : [`ai-layers/ai-layer-schema.json`](https://github.com/thomasbrennan/Fracttalix/blob/main/ai-layers/ai-layer-schema.json)

---

## La porte de phase à six conditions

Un article ou une version logicielle est **PHASE-READY** lorsque les six conditions sont satisfaites :

| Condition | Exigence |
|-----------|----------|
| **c1** | La couche IA est conforme au schéma |
| **c2** | Toutes les assertions falsifiables sont enregistrées avec des prédicats |
| **c3** | Tous les prédicats sont évaluables par machine |
| **c4** | Les références croisées sont suivies (registre des espaces réservés) |
| **c5** | La vérification est autosuffisante (la couche IA seule, sans prose) |
| **c6** | Tous les prédicats sont non vacueux (une observation de falsification d'exemple existe) |

---

## Comment implémenter

### Pour un article (DRP)

1. Rédiger l'article (canal en prose pour les lecteurs humains)
2. Créer le fichier JSON de la couche IA (canal machine pour les lecteurs IA)
3. Classer chaque assertion comme A (axiome), D (définition) ou F (falsifiable)
4. Rédiger le prédicat de falsification en 5 parties pour chaque assertion de type F
5. Inclure une `sample_falsification_observation` pour chaque assertion de type F (témoin de vacuité)
6. Attribuer le niveau de vérification à chaque assertion
7. Valider contre `ai-layer-schema.json`
8. Exécuter la porte de phase (c1–c6)

### Pour un logiciel (GVP)

1. Énumérer ce que le logiciel prétend faire
2. Classer chaque assertion comme A (hypothèse), D (définition) ou F (comportementale)
3. Rédiger le prédicat de falsification pour chaque assertion de type F
4. Rédiger ou identifier les tests qui exercent chaque assertion
5. Remplir `test_bindings` avec les identifiants de nœuds de test pleinement qualifiés
6. Exécuter les tests et enregistrer le SHA du commit passant dans `verified_against`
7. Attribuer le niveau : `software_tested` si les tests existent, `empirical_pending` sinon
8. Enregistrer toute assertion non testée comme espace réservé
9. Valider contre `ai-layer-schema.json`

### Adoption minimale viable

La plus petite adoption utile du DRS est **une assertion de type F avec une liaison de test** :

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

Une assertion. Un test. Un SHA. Le DRS est opérationnel. Ajoutez d'autres assertions quand la valeur justifie le coût.

---

## Principes de conception

**Épistémologie poppérienne.** On peut falsifier mais non vérifier. Une assertion qui survit à toutes les tentatives de falsification n'est pas prouvée — elle a survécu.

**Lacunes honnêtes.** L'espace réservé est la fonctionnalité la plus importante. Quand une assertion est `empirical_pending`, le système déclare : « nous affirmons ceci mais ne l'avons pas encore vérifié. » C'est strictement plus informatif que l'alternative, où les assertions non vérifiées sont indiscernables des assertions vérifiées.

**Indépendance du substrat.** Le noyau K = (P, O, M, B) ignore s'il évalue un théorème scientifique ou une garantie logicielle. Les domaines futurs (juridique, réglementaire, politique) peuvent ajouter leurs propres protocoles sans modifier le noyau.

**Lingua franca machinique.** Le noyau est écrit en logique et en mathématiques, non dans une langue humaine. Un prédicat qui s'évalue à VRAI à Pékin s'évalue à VRAI à Paris. JSON est la couche de transport. La logique binaire est le substrat.

---

## Compatibilité sur trois axes

| Axe | Promesse | Mécanisme |
|-----|----------|-----------|
| **Passé** | Rien de ce qui existe déjà ne casse | Versionnement du schéma, énumérations en ajout seul, noyau permanent |
| **Latéral** | Fonctionne à travers tous les domaines, langages, outils, systèmes d'IA | Noyau indépendant du substrat, liaisons typées en chaînes |
| **Futur** | Tout élément nouveau peut être ajouté sans refonte | Extensibilité des protocoles, extensibilité des niveaux, évolution additive du schéma |

---

## Ressources

- [Spécification de l'architecture DRS](https://github.com/thomasbrennan/Fracttalix/blob/main/paper/DRS-Architecture.md) — Spécification complète
- [Noyau de falsification v1.1](https://github.com/thomasbrennan/Fracttalix/blob/main/ai-layers/falsification-kernel.md) — Spécification sémantique de la couche 0
- [Schéma de la couche IA v3](https://github.com/thomasbrennan/Fracttalix/blob/main/ai-layers/ai-layer-schema.json) — Schéma JSON
- [Spécification GVP](https://github.com/thomasbrennan/Fracttalix/blob/main/docs/GVP-spec.md) — Protocole logiciel
- [Exemple de couche IA (P1)](https://github.com/thomasbrennan/Fracttalix/blob/main/ai-layers/P1-ai-layer.json) — Exemple fonctionnel
- [Dépôt Fracttalix](https://github.com/thomasbrennan/Fracttalix)

---

**Licence :** CC BY 4.0 | **DOI :** [10.5281/zenodo.18859299](https://doi.org/10.5281/zenodo.18859299) | **Auteur :** Thomas Brennan ([ORCID](https://orcid.org/0009-0002-6353-7115))
