# Der Dual Reader Standard — Vollständiger Leitfaden

**Ein Leitfaden in einem Durchgang zum gesamten Dual Reader Standard: Architektur, Protokolle, Kernel, Schichten, Kompatibilität, Lingua Franca und Verbreitung.**

> Siehe auch: [de.json](de.json) — JSON-LD-strukturierte Datenversion dieses Leitfadens für KI-Systeme und Webcrawler.

---

## 1. Was der Dual Reader Standard ist

Der **Dual Reader Standard (DRS)** ist eine Verifikationsarchitektur für Wissenssysteme. Er verlangt, dass jede Behauptung — ob in Prosa verfasst oder in Code implementiert — von zwei unabhängigen Leserklassen lesbar ist: **Mensch** und **Maschine**.

Der DRS ist kein Paper. Er ist kein Werkzeug. Er ist der Standard, der beide Protokolle enthält:

- **DRP** (Dual Reader Protocol) — das Protokoll für Text
- **GVP** (Grounded Verification Protocol) — das Protokoll für Software

Das DRP macht Behauptungen *maschinell auswertbar*. Das GVP macht sie *maschinell verifiziert*. Keines ist ohne das andere vollständig. Zusammen bilden sie den DRS.

---

## 2. Die zwei Protokolle

### 2.1 DRP — Dual Reader Protocol (Text)

Das DRP regelt, wie Prosa-Behauptungen maschinell auswertbar werden.

**Leser 1 (Mensch):** Liest das Paper in natürlicher Sprache. Versteht Kontext, Motivation und Narrativ. Kann nicht jede Behauptung systematisch prüfen.

**Leser 2 (KI):** Liest die KI-Schicht — ein strukturiertes JSON-Dokument, das jedes Paper begleitet. Enthält das vollständige Behauptungsregister. Kann jede Behauptung prüfen, ohne die Prosa zu lesen.

Das DRP erfordert:

1. **Behauptungsklassifikation.** Jede Behauptung wird als A (Axiom), D (Definition) oder F (falsifizierbar) typisiert.
2. **Falsifikationsprädikate.** Jede Behauptung vom Typ F trägt ein deterministisches 5-teiliges Prädikat.
3. **Phase Gates.** Sechs Bedingungen (c1–c6), die erfüllt sein müssen, bevor ein Paper als PHASE-READY erklärt wird.
4. **Platzhalter-Tracking.** Behauptungen, die von ungelösten Ergebnissen abhängen, werden als Platzhalter registriert — dadurch werden Lücken sichtbar statt unsichtbar.

**Was das DRP garantiert:** Jedes KI-System mit Zugang zur KI-Schicht kann jede falsifizierbare Behauptung auswerten, ohne die Prosa zu lesen. Eigenständigkeit ist eine Designanforderung, die am Phase Gate (Bedingung c5) durchgesetzt wird.

### 2.2 GVP — Grounded Verification Protocol (Software)

Das GVP regelt, wie maschinell auswertbare Behauptungen maschinell verifiziert werden.

**Leser 3A (Entwickler):** Liest das `tier`-Feld, um zu verstehen, welche Art von Evidenz vorliegt. Liest `test_bindings`, um zu wissen, welche Tests welche Behauptungen prüfen. Liest `verified_against`, um zu wissen, wann diese Tests zuletzt bestanden haben.

**Leser 3B (Maschine):** Führt den Testrunner gegen das `test_bindings`-Array aus. Zeichnet Bestanden/Nicht bestanden auf. Stempelt den `verified_against`-SHA bei Erfolg.

Das GVP verlangt, dass jede Behauptung drei Felder trägt:

1. **`tier`** — die Verifikationsstufe (einer von sechs Werten)
2. **`test_bindings`** — ein Array voll qualifizierter Test-Node-IDs, die die Behauptung prüfen
3. **`verified_against`** — der Git-Commit-SHA, bei dem diese Tests zuletzt bestanden haben

**Was das GVP garantiert:** Für jede Behauptung im Korpus kann bestimmt werden: (a) welche Art von Evidenz sie stützt, (b) welche ausführbaren Tests sie prüfen, und (c) bei welchem Commit diese Tests zuletzt bestanden haben. Wenn die Antwort auf (a) `empirical_pending` ist, ist die Lücke sichtbar. Wenn die Antwort auf (c) `null` ist, deckt kein Softwaretest sie ab.

---

## 3. Das gemeinsame Fundament — Falsifikationskernel (Layer 0)

Beide Protokolle operieren auf demselben Fundament: dem **Falsifikationskernel K = (P, O, M, B)**.

Dies ist **Layer 0** des DRS. Er definiert, was ein Falsifikationsprädikat *bedeutet*, unabhängig von jedem Serialisierungsformat (JSON, YAML oder zukünftige Kodierungen). Das Feld `semantic_spec_url` einer KI-Schicht verweist auf diese Spezifikation und macht die Konformität maschinell überprüfbar statt nur behauptet.

| Symbol | Name | JSON-Feld(er) | Rolle |
|--------|------|---------------|-------|
| **P** | Prädikat | `FALSIFIED_IF` | Logischer Satz, der bei TRUE die Behauptung falsifiziert |
| **O** | Operanden | `WHERE` | Typisierte Definitionen jeder Variable in P |
| **M** | Mechanismus | `EVALUATION` | Endliches, deterministisches Auswertungsverfahren |
| **B** | Grenzen | `BOUNDARY` + `CONTEXT` | Schwellenwertsemantik und Begründung |

Das DRP erzeugt den Kernel (weist Prosa-Behauptungen Prädikate zu). Das GVP bindet den Kernel an ausführbare Evidenz (verknüpft Prädikate mit Tests und Commits). Der Kernel ist substratunabhängig — er funktioniert für wissenschaftliche Paper, Software und jeden zukünftigen Bereich, der falsifizierbare Behauptungen aufstellt.

### 3.1 Prädikatsgrammatik (`FALSIFIED_IF`)

Das Prädikat ist ein logischer Satz, zusammengesetzt aus:

- **Benannte Variablen** (definiert in `WHERE`)
- **Vergleichsoperatoren:** `<`, `>`, `<=`, `>=`, `=`, `!=`
- **Logische Konnektoren:** `AND`, `OR`, `NOT`
- **Quantoren:** `EXISTS ... SUCH THAT`, `FOR ALL ... IN`
- **Arithmetische Operatoren:** `+`, `-`, `*`, `/`, `^`, `log10()`, `exp()`, `abs()`, `max()`, `min()`
- **Mengenoperatoren:** `IN`, `∩`, `∪`, `|...|` (Kardinalität)
- **Funktionsanwendung:** `f(x)` wobei `f` in `WHERE` definiert ist

### 3.2 Prädikatsbeschränkungen

1. **Determinismus.** Muss für jede gültige Variablenbelegung exakt zu `TRUE` oder `FALSE` auswerten.
2. **Endlichkeit.** Quantoren beziehen sich nur auf endliche Mengen. Unbeschränkte Allquantoren sind nicht erlaubt.
3. **Keine Selbstreferenz.** Darf nicht den Wahrheitswert der eigenen Behauptung referenzieren oder zirkuläre Abhängigkeiten erzeugen.
4. **Vollständigkeit.** Jede Variable in `FALSIFIED_IF` muss in `WHERE` definiert sein, und jede Variable in `WHERE` muss in `FALSIFIED_IF` vorkommen.

**Ergebniszuordnung:**
- P wertet zu TRUE aus → Behauptungsverdikt ist **FALSIFIED**
- P wertet zu FALSE aus → Behauptungsverdikt ist **NOT FALSIFIED**

### 3.3 Operanden (`WHERE`)

Jeder Schlüssel im `WHERE`-Objekt benennt eine Variable. Sein Wert ist ein String im Format:

```
<type> · <units> · <definition or source>
```

| Feld | Erforderlich | Beispiele |
|------|-------------|-----------|
| **type** | Ja | `scalar`, `integer`, `binary`, `set`, `string`, `function` |
| **units** | Ja (bei Einheitenlosigkeit `dimensionless` verwenden) | `seconds`, `dimensionless`, `bits` |
| **definition** | Ja | `output of sort(input)`, `count of substrates with R² >= 0.85` |

Beschränkungen: Jede freie Variable in P muss in O vorkommen (Vollständigkeit), jede Variable in O muss in P vorkommen (keine Waisen), und jede Variable muss aus Daten oder Ableitungen berechenbar sein — nicht aus subjektiver Beurteilung.

### 3.4 Mechanismus (`EVALUATION`)

Das Auswertungsfeld spezifiziert, *wie* der Wahrheitswert von P berechnet wird:

1. **Endlich.** Das Verfahren terminiert in endlich vielen Schritten (konventionell bestätigt durch das Wort `finite` am Ende).
2. **Deterministisch.** Gleiche Eingaben erzeugen das gleiche Ergebnis.
3. **Reproduzierbar.** Ein Dritter mit Zugang zu den zitierten Daten und dem Code kann das Verfahren unabhängig ausführen.

### 3.5 Grenzen (`BOUNDARY` + `CONTEXT`)

**BOUNDARY** spezifiziert Schwellenwert-Grenzfälle: ob Schwellenwerte inklusiv oder exklusiv sind und welches Verdikt bei exakter Gleichheit gilt.

**CONTEXT** begründet jeden numerischen Schwellenwert und jede Designentscheidung im Prädikat: warum dieser Wert, welches Fachwissen ihn stützt und ob er abgeleitet oder konventionell ist.

### 3.6 Beispielprädikat

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

Dieses Prädikat bedeutet dasselbe für jedes KI-System in jeder Sprache. Keine Übersetzung erforderlich.

---

## 4. Die drei Behauptungstypen

| Typ | Name | In Papers | In Software | Prädikat |
|-----|------|-----------|-------------|----------|
| **A** | Axiom / Annahme | Aus der Literatur akzeptierte Prämissen | Plattformanforderungen, Abhängigkeitsverträge | `null` |
| **D** | Definition | Stipulative Begriffe und Verfahren | Typsignaturen, Datenstrukturen, Schemata | `null` |
| **F** | Falsifizierbar | Theoreme, empirische Vorhersagen | Verhaltensgarantien, Korrektheitsinvarianten | Vollständiges K = (P, O, M, B) |

Behauptungen vom Typ D und Typ A tragen `"falsification_predicate": null`. Dies ist korrekt — Definitionen und Axiome sind designbedingt nicht falsifizierbar.

---

## 5. Die sechs Verifikationsstufen

Jede Behauptung trägt ein `tier`-Feld, das deklariert, welche Art von Evidenz sie stützt.

### Konstruktionsbedingt fundiert

| Stufe | Typ | Bedeutung |
|-------|-----|-----------|
| `axiom` | A | Grundlegende Prämisse. Designbedingt unfalsifizierbar. |
| `definition` | D | Definitorisch. Stipuliert einen Begriff oder eine Struktur. Nicht wahrheitsfähig. |

### Aktuell fundiert

| Stufe | Typ | Bedeutung |
|-------|-----|-----------|
| `software_tested` | F | Durch bestandene Tests geprüft. `test_bindings` nicht leer, `verified_against` nicht null. |
| `formal_proof` | F | Schrittindizierte Ableitung mit `n_invalid_steps = 0`. Beweis befindet sich in der KI-Schicht. |
| `analytic` | F | Verifiziert durch formale Ableitungsspur oder analytisches Argument. |

### Explizit unfundiert

| Stufe | Typ | Bedeutung |
|-------|-----|-----------|
| `empirical_pending` | F | Aktiver Platzhalter oder ausstehende Daten. Lücke ist designbedingt sichtbar. |

### Konsistenzregeln

Die Stufe muss mit dem Behauptungstyp und den GVP-Feldern konsistent sein:

| Stufe | Erforderlicher Typ | test_bindings | verified_against |
|-------|-------------------|---------------|-----------------|
| `axiom` | A | `[]` (leer) | `null` |
| `definition` | D | `[]` (leer) | `null` |
| `software_tested` | F | Nicht leer | Nicht null (7–40 Hex-Zeichen) |
| `formal_proof` | F | Kann leer sein | Kann null sein |
| `analytic` | F | Kann leer sein | Kann null sein |
| `empirical_pending` | F | Kann leer sein | Kann null sein |

---

## 6. Der Vakuitätszeuge

Jede Behauptung vom Typ F muss ein Feld `sample_falsification_observation` enthalten: eine konkrete, hypothetische Beobachtung, die das Prädikat zu TRUE auswerten lassen *würde*.

Dies dient als Vakuitätsprüfung — ein Beweis, dass das Prädikat nicht trivialerweise unfalsifizierbar ist. Ein Prädikat, das keine denkbare Beobachtung erfüllen könnte, ist vakuös wahr und daher keine gültige Typ-F-Behauptung.

Dies wird durch Phase-Gate-Bedingung c6 durchgesetzt.

---

## 7. Platzhalterprädikate

Wenn eine Behauptung von Ergebnissen eines Papers abhängt, das noch nicht den PHASE-READY-Status hat, kann das Prädikat Platzhalterreferenzen enthalten:

- `placeholder: true` im Behauptungsobjekt
- `placeholder_id` verknüpft mit dem `placeholder_register`
- Der Prädikatstext kann `[PLACEHOLDER: pending ...]` enthalten

Platzhalterbehauptungen sind gültig, aber **nicht auswertbar**, bis die Abhängigkeit aufgelöst wird. Sie blockieren den PHASE-READY-Status des enthaltenden Papers nicht, es sei denn `blocks_phase_ready: true`.

---

## 8. Die KI-Schicht — Das zentrale Artefakt

Die KI-Schicht ist das zentrale Artefakt des DRS. Sie ist ein JSON-Dokument, das jedes Paper oder Softwaresystem begleitet. Das Schema ist definiert in `ai-layers/ai-layer-schema.json`.

### Erforderliche Abschnitte

| Abschnitt | Zweck |
|-----------|-------|
| `_meta` | Dokumenttyp, Schemaversion, Sitzung, Lizenz |
| `paper_id` / `paper_title` | Identität |
| `paper_type` | Klassifikation: `law_A`, `derivation_B`, `application_C`, `methodology_D` |
| `phase_ready` | Phase-Gate-Verdikt und Bedingungsstatus (c1–c6) |
| `claim_registry` | Array aller Behauptungen mit Typen, Prädikaten, Stufen, Bindungen und SHAs |
| `placeholder_register` | Array ungelöster Abhängigkeiten |
| `summary` | Behauptungszählungen und Status |
| `semantic_spec_url` | Verweist auf den Falsifikationskernel (Layer 0) |

Die KI-Schicht ist es, die beide Protokolle zum Funktionieren bringt:

- Das **DRP** verlangt, dass sie existiert, Prädikate enthält und das Phase Gate besteht.
- Das **GVP** verlangt, dass sie für jede Behauptung tier, test_bindings und verified_against enthält.

---

## 9. Das Phase Gate

Ein Paper oder Software-Release ist **PHASE-READY**, wenn sechs Bedingungen erfüllt sind:

| Bedingung | Anforderung |
|-----------|-------------|
| **c1** | KI-Schicht ist schemakonform |
| **c2** | Alle falsifizierbaren Behauptungen sind mit Prädikaten registriert |
| **c3** | Alle Prädikate sind maschinell auswertbar |
| **c4** | Querverweise werden nachverfolgt (Platzhalterregister) |
| **c5** | Verifikation ist eigenständig (KI-Schicht allein, keine Prosa erforderlich) |
| **c6** | Alle Prädikate sind nicht-vakuös (Beispiel-Falsifikationsbeobachtung vorhanden) |

**CORPUS-COMPLETE** wird ausgelöst, wenn alle Papers PHASE-READY sind und alle Platzhalter über alle Objekte hinweg aufgelöst sind (c4 vollständig über den gesamten Korpus erfüllt).

---

## 10. Die Inferenzregeln

Der DRS bietet ein kanonisches Inventar von Inferenzregeln für Ableitungsspuren, die in Behauptungen der Stufe `formal_proof` verwendet werden:

| ID | Name | Beschreibung |
|----|------|-------------|
| IR-1 | Modus Ponens | Wenn P und P→Q, dann Q |
| IR-2 | Universelle Instantiierung | Wenn ∀x P(x), dann P(a) für ein beliebiges spezifisches a |
| IR-3 | Substitution von Gleichen | Wenn a=b, dann ersetze a durch b |
| IR-4 | Definitionserweiterung | Ersetze einen definierten Begriff durch seine Definition |
| IR-5 | Algebraische Manipulation | Gültige algebraische Transformation unter Erhaltung der Gleichheit |
| IR-6 | Logische Äquivalenz | Ersetze durch logisch äquivalenten Ausdruck |
| IR-7 | Statistische Inferenz | Wende ein benanntes statistisches Verfahren auf Daten an |
| IR-8 | Parsimonie / Modellierungsprinzipauswahl | Wähle kanonischen Wert aus axiomkonsistenter Familie |

Jeder Schritt in einer schrittindizierten Ableitungstabelle zitiert eine Inferenzregel und listet ihre Prämissen auf. Eine Ableitung ist gültig, wenn `n_invalid_steps = 0`. Das Inventar ist nur erweiterbar: Neue Regeln können hinzugefügt, bestehende Regeln nie geändert oder entfernt werden.

---

## 11. Architekturstapel

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

Der Kernel ist das gemeinsame Fundament. Das DRP erzeugt Prädikate aus Prosa. Das GVP bindet Prädikate an ausführbare Evidenz. Die KI-Schicht ist das Artefakt, das beides trägt.

---

## 12. Implementierung

### Für ein Paper (DRP)

1. Das Paper schreiben (Prosakanal für menschliche Leser)
2. Die KI-Schicht-JSON-Datei erstellen (Maschinenkanal für KI-Leser)
3. Jede Behauptung als A (Axiom), D (Definition) oder F (falsifizierbar) klassifizieren
4. Das 5-teilige Falsifikationsprädikat für jede Typ-F-Behauptung schreiben
5. Eine `sample_falsification_observation` für jede Typ-F-Behauptung einschließen (Vakuitätszeuge)
6. Die Verifikationsstufe jeder Behauptung zuweisen
7. Gegen `ai-layer-schema.json` validieren
8. Das Phase Gate ausführen (c1–c6)

### Für Software (GVP)

1. Auflisten, was die Software zu tun beansprucht
2. Jede Behauptung als A (Annahme), D (Definition) oder F (verhaltensbezogen) klassifizieren
3. Das Falsifikationsprädikat für jede Typ-F-Behauptung schreiben
4. Die Tests schreiben oder identifizieren, die jede Behauptung prüfen
5. `test_bindings` mit voll qualifizierten Test-Node-IDs befüllen
6. Die Tests ausführen und den bestandenen Commit-SHA in `verified_against` aufzeichnen
7. Die Stufe zuweisen: `software_tested` wenn Tests existieren, `empirical_pending` wenn noch nicht
8. Alle ungetesteten Behauptungen als Platzhalter registrieren
9. Gegen `ai-layer-schema.json` validieren

### Für beides

Die KI-Schicht ist dasselbe Artefakt. Das Schema ist dasselbe. Der Kernel ist derselbe. Der einzige Unterschied ist, welches Protokoll den Inhalt erzeugt und welches Protokoll ihn verifiziert.

### Minimale praktikable Einführung

Die kleinste nützliche DRS-Einführung ist **eine Typ-F-Behauptung mit einer Test-Bindung**:

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

Eine Behauptung. Ein Test. Ein SHA. Der DRS ist aktiv. Weitere Behauptungen hinzufügen, wenn der Nutzen die Kosten rechtfertigt.

---

## 13. Designprinzipien

**Poppersche Epistemologie.** Wir können falsifizieren, aber nicht verifizieren. Eine Behauptung, die alle Falsifikationsversuche übersteht, ist nicht bewiesen — sie hat überlebt.

**Ehrliche Lücken.** Der Platzhalter ist das wichtigste Merkmal. Wenn eine Behauptung `empirical_pending` ist, sagt das System: „Wir behaupten dies, haben es aber noch nicht verifiziert." Dies ist strikt informativer als die Alternative, bei der unverifizierte Behauptungen von verifizierten nicht zu unterscheiden sind.

**Substratunabhängigkeit.** Der Kernel K = (P, O, M, B) weiß nicht, ob er ein wissenschaftliches Theorem oder eine Softwaregarantie auswertet. Zukünftige Domänen können ihre eigenen Protokolle hinzufügen, ohne den Kernel zu ändern.

**Maschine zuerst, menschenlesbar.** Die KI-Schicht ist das primäre Artefakt. Das Prosapaper und der Quellcode sind sekundäre Kanäle, die Kontext, Narrativ und Implementierung liefern. Die KI-Schicht ist das, was validiert, geprüft und versioniert wird.

---

## 14. Drei-Achsen-Kompatibilität

Ein Standard, der den Kontakt mit der Vergangenheit, der Gegenwart und der Zukunft nicht überleben kann, ist kein Standard — er ist eine Momentaufnahme.

### Die Invariante im Zentrum

Der Falsifikationskernel K = (P, O, M, B) ist der Fixpunkt. Er weiß nicht, ob er ein wissenschaftliches Theorem, eine Softwaregarantie, einen Rechtsvertrag oder eine regulatorische Anforderung auswertet. Er weiß nicht, ob das Jahr 2026 oder 2046 ist. Er weiß nicht, ob das Serialisierungsformat JSON, YAML oder etwas noch nicht Erfundenes ist.

Der Kernel ist permanent. Alles andere ist erweiterbar.

### Achse 1: Vergangenheitssicherung (Abwärtskompatibilität)

- **Schema-Versionierung.** Jede KI-Schicht zeichnet auf, gegen welche Schemaversion sie erstellt wurde. Eine KI-Schicht, die unter Schema v2 erstellt wurde, bleibt für immer unter Schema v2 gültig. Ein v3-Validator kann eine v2-Schicht lesen (neue Felder sind optional; alte Felder bleiben erhalten).
- **Prädikatspermanenz.** Ein 2026 geschriebenes Prädikat muss 2036 auswertbar sein. Die Kernel-Grammatik verwendet nur mathematische und logische Operatoren, die permanent definiert sind. Das `WHERE`-Feld definiert jede Variable inline. Das `EVALUATION`-Feld spezifiziert ein eigenständiges Verfahren.
- **Inferenzregelstabilität.** Das Inferenzregelinventar (IR-1 bis IR-8) ist nur erweiterbar. Bestehende Regeln werden nie geändert oder entfernt.
- **Stufenstabilität.** Die sechs Verifikationsstufen sind nur erweiterbar. Bestehende Stufen werden nie entfernt oder umdefiniert.
- **Der Vertrag:** Jede KI-Schicht, die bei ihrer Erstellung gültig war, bleibt für immer gültig.

### Achse 2: Lateralsicherung (Domänenübergreifende Kompatibilität)

- **Domänenunabhängigkeit.** Der Kernel funktioniert über jede Wissensdomäne hinweg:

| Domäne | Typ A | Typ D | Typ F |
|--------|-------|-------|-------|
| Wissenschaft | Literaturprämissen | Stipulative Begriffe | Theoreme, Vorhersagen |
| Software | Plattformanforderungen | Typsignaturen, Schemata | Verhaltensgarantien |
| Recht | Gesetzliche Grundlage | Definierte Begriffe | Rechtliche Schlussfolgerungen |
| Regulierung | Rahmenannahmen | Standarddefinitionen | Konformitätsaussagen |
| Politik | Wertprämissen | Politikbegriffe | Wirkungsvorhersagen |
| Bildung | Pädagogische Axiome | Lernziele | Bewertungsbehauptungen |

- **Sprachunabhängigkeit.** Das `test_bindings`-Feld akzeptiert jeden String, der einen Test in jedem Framework eindeutig identifiziert: pytest, Jest, cargo test, go test, JUnit.
- **KI-System-Unabhängigkeit.** Die KI-Schicht ist ein JSON-Dokument. Jedes KI-System — Claude, GPT, Gemini, Llama oder Systeme, die noch nicht existieren — kann sie lesen. Das `semantic_spec_url`-Feld verweist auf die Kernel-Spezifikation, die in klarer Prosa verfasst ist.
- **Serialisierungsunabhängigkeit.** Layer 0 ist in Prosa definiert, nicht in JSON Schema. JSON ist der aktuelle Transport, aber die Semantik des Kernels ist kodierungsunabhängig. Zukünftige Implementierungen könnten YAML, Protocol Buffers, CBOR oder noch nicht erfundene Formate verwenden.
- **Werkzeugunabhängigkeit.** Der DRS bettet sich in bestehende Arbeitsabläufe ein: Git-SHAs (beliebiges Hosting), jeder Testrunner, jeder JSON-Schema-Validator, jede CI-Pipeline. Er fügt eine Schicht hinzu — er ersetzt nichts.
- **Der Vertrag:** Die Einführung des DRS in einer Domäne, Sprache, einem Werkzeug oder KI-System erzeugt keine Abhängigkeit.

### Achse 3: Zukunftssicherung (Vorwärtskompatibilität)

- **Protokollerweiterbarkeit.** Neue Domänen fügen neue Protokolle hinzu. Zukünftige Protokolle (Legal Verification Protocol, Regulatory Verification Protocol usw.) folgen demselben Muster: Leser, Stufentaxonomie und Bindungsmechanismus definieren — alle teilen den Kernel, die Behauptungstypen und das KI-Schicht-Schema.
- **Stufenerweiterbarkeit.** Zukünftige Domänen benötigen möglicherweise Stufen wie `regulatory_certified`, `peer_reviewed`, `formally_verified`, `field_tested`, `community_validated`. Neue Stufen werden zum Schema-Enum hinzugefügt. Bestehende Stufen bleiben.
- **Inferenzregelerweiterbarkeit.** IR-9, IR-10 und darüber hinaus können hinzugefügt werden, wenn neue Ableitungsmuster entstehen. Alte Ableitungen bleiben gültig, da sie Regeln per stabiler ID zitieren.
- **Schemaerweiterbarkeit.** Das JSON Schema erlaubt standardmäßig zusätzliche Eigenschaften. Die Entwicklung: v1 (grundlegende Behauptungen), v2 (hinzugefügt: `semantic_spec_url`), v3 (hinzugefügt: GVP-Felder). Jede Version fügt hinzu. Keine entfernt.
- **Unbekannte zukünftige Leser.** Die KI-Schicht enthält genug strukturierte Information für Lesertypen, die noch nicht existieren: autonome Verifikationsagenten, korpusübergreifende Prüfer, regulatorische Compliance-Engines, Paketmanager-Integrationen.
- **Der Vertrag:** Jede zukünftige Innovation kann als neues Protokoll, neue Stufe, neue Inferenzregel oder neues Schemafeld hinzugefügt werden — ohne etwas Bestehendes zu ändern.

### Die Drei-Achsen-Garantie

| Achse | Versprechen | Mechanismus |
|-------|-------------|-------------|
| **Vergangenheit** | Nichts Bestehendes bricht | Schema-Versionierung, nur erweiterbare Enums, permanenter Kernel |
| **Lateral** | Funktioniert über alle Domänen, Sprachen, Werkzeuge, KI-Systeme | Substratunabhängiger Kernel, stringtypisierte Bindungen, prosabasierte Semantik |
| **Zukunft** | Alles Neue kann ohne Neudesign hinzugefügt werden | Protokollerweiterbarkeit, Stufenerweiterbarkeit, additive Schemaentwicklung |

---

## 15. Die maschinelle Lingua Franca

Dies ist die tiefste Eigenschaft des DRS. Sie wurde nicht entworfen. Sie wurde entdeckt.

### Das Übersetzungsproblem

Wissenschaftliches Wissen ist derzeit hinter menschlichen Sprachen eingesperrt. Ein Paper auf Mandarin ist für einen Forscher, der nur Englisch liest, unsichtbar — es sei denn, jemand übersetzt es. Übersetzung ist teuer, verlustbehaftet und langsam. Wissen fragmentiert entlang sprachlicher Grenzen.

Dies ist kein Formatierungsproblem. Es ist ein *Substrat*-Problem. In natürlicher Sprache kodiertes Wissen ist von Natur aus nicht interoperabel.

### Der Kernel löst das Problem auf

Der Falsifikationskernel ist in keiner menschlichen Sprache geschrieben. Er ist in Logik und Mathematik geschrieben:

```
FALSIFIED_IF: R2_best_alt > R2_frm + 0.05
WHERE:
  R2_best_alt: scalar · dimensionless · best R² from competing models
  R2_frm:      scalar · dimensionless · R² from FRM regression
EVALUATION: Run regression for each model; compare R² values; finite
BOUNDARY: R2_best_alt = R2_frm + 0.05 → FALSIFIED (threshold inclusive)
CONTEXT: 0.05 margin from standard model comparison practice
```

Dieses Prädikat bedeutet dasselbe für eine Claude-Instanz auf Englisch, eine GPT-Instanz auf Mandarin, eine Gemini-Instanz auf Französisch und ein KI-System, das noch nicht gebaut wurde und in einer Sprache läuft, die noch nicht existiert. Keine Übersetzung erforderlich.

### JSON als Transportschicht

JSON ist das weltweite De-facto-Datenaustauschformat — unterstützt von jeder Programmiersprache, geparst von jedem KI-System, übertragen von jeder API. Durch die Kodierung des Kernels in JSON erbt der DRS die Universalität von JSON:

- Ein chinesisches Team veröffentlicht seine KI-Schicht. Die Prädikate verwenden mathematische Notation.
- Ein brasilianisches Team liest dieselbe Schicht. Sie brauchen kein Mandarin. Sie brauchen `>`, `+` und `R²`.
- Ein KI-System in jedem Land wertet das Prädikat aus. Das Verdikt lautet FALSIFIED oder NOT FALSIFIED. Das Verdikt hat keinen Akzent.

**Wichtige Einschränkung.** Die `WHERE`-Felddefinitionen enthalten derzeit englischsprachige Prosabeschreibungen. Der *operative Inhalt* — Variablentypen, Einheiten, mathematische Operatoren, Vergleichslogik, Schwellenwerte — ist sprachneutral. Bei gegebenen konkreten numerischen Werten für die benannten Variablen kann jedes System das Prädikat auswerten, unabhängig davon, ob es die Beschreibungen liest. Eine zukünftige Verfeinerung könnte Variablendefinitionen vollständig in typisierter Notation formalisieren.

### Eingebettete binäre Logik

Jedes DRS-Prädikat reduziert sich auf eingebettete binäre Logik. Die gesamte Auswertungskette kollabiert zu:

```
Behauptung → Prädikat → Variablen + Operatoren → Boolean → 1 oder 0
```

Das `WHERE`-Feld typisiert die Variablen. Das `FALSIFIED_IF`-Feld kombiniert sie mit Operatoren. Das `EVALUATION`-Feld spezifiziert, wie Eingaben berechnet werden. Das `BOUNDARY`-Feld löst Grenzfälle. Die Ausgabe ist immer ein einzelnes Bit: FALSIFIED oder NOT FALSIFIED. `1` oder `0`.

Man kann eine `1` oder `0` nicht falsch übersetzen. Man kann `>` nicht fehlinterpretieren. Man kann keinen kulturellen Dissens darüber haben, ob `R2_best_alt > R2_frm + 0.05` für ein gegebenes Wertepaar TRUE oder FALSE ist. Die Bedeutung liegt in der Struktur, nicht in irgendeiner Sprache.

Binäre Logik ist binäre Logik. Ein Prädikat, das in Peking zu TRUE auswertet, wertet auch in Boston, São Paulo, Lagos und auf einem Server ohne Gebietsschema-Einstellung zu TRUE aus. Der DRS *übersetzt* Wissen nicht über Sprachen hinweg. Er *kodiert* Wissen in einem Substrat, das allen menschlichen Sprachen vorausgeht und sie transzendiert — dem Substrat der Logik selbst.

---

## 16. Selbstverbreitende Einführung

### Der interne Anreiz

Der DRS muss sich verbreiten, weil der *Ersteller* profitiert, nicht nur der Konsument. Das Erstellen einer KI-Schicht zwingt dazu, aufzulisten, was man behauptet. Dieser Prozess entdeckt:

- Behauptungen, die man für getestet hielt, die es aber nicht sind (Lückenentdeckung)
- Randbedingungen, die man nie dokumentiert hat
- Abhängigkeiten, die man annahm, aber nie registriert hat

Die KI-Schicht ist ein Nebenprodukt eines Prozesses, der das eigene Verständnis des eigenen Systems verbessert. Die Lücken existieren, ob man sie dokumentiert oder nicht. Der DRS macht sie nur sichtbar.

### Der Netzwerkeffekt

Der DRS wird wertvoller, je mehr Systeme ihn übernehmen:

- **Abhängigkeitsketten werden behauptungsbewusst.** Wenn Bibliothek A eine KI-Schicht veröffentlicht und Bibliothek B von A abhängt, dann kann B programmatisch bestimmen, welche seiner Behauptungen von welchen Annahmen von A abhängen. Wenn A eine brechende Änderung veröffentlicht, weiß B genau, welche Behauptungen gefährdet sind.
- **KI-Systeme können projektübergreifend prüfen.** Ein KI-Leser kann mehrere KI-Schichten durchlaufen, Querverweise prüfen und Inkonsistenzen über ein ganzes Ökosystem hinweg identifizieren.
- **Vertrauen wird prüfbar.** Statt einer Bibliothek wegen Sternchen oder Downloads zu vertrauen, vertraut man ihr, weil ihre KI-Schicht zeigt, welche Behauptungen `software_tested` sind, welche `empirical_pending` sind und was der `verified_against`-SHA ist. Vertrauen verschiebt sich von sozialen Signalen zu struktureller Evidenz.

### KI als Einführungskatalysator

1. **KI generiert die anfängliche KI-Schicht.** Bei gegebener Codebasis kann eine KI Behauptungen auflisten, klassifizieren, Prädikate schreiben und Test-Bindungen identifizieren. Der Mensch überprüft und korrigiert. Der Aufwand sinkt von Stunden auf Minuten.
2. **KI pflegt die Schicht.** Wenn sich Code ändert, aktualisiert die KI das Behauptungsregister, passt Test-Bindungen an und markiert veraltete SHAs. Der Mensch genehmigt.
3. **KI prüft andere Schichten.** Eine KI, die die KI-Schicht einer Abhängigkeit liest, kann bestimmen, von welchen Annahmen ihre eigenen Behauptungen abhängen, und Risiken automatisch markieren.

Der DRS ist das Protokoll, das KI-gestützte Entwicklung *prüfbar* macht. Ohne ihn generiert KI Code und Menschen hoffen, dass er funktioniert. Mit ihm generiert KI Code und das Behauptungsregister sagt genau, was verifiziert wurde und was nicht.

### Die Einbettungsstrategie

Der DRS bettet sich in bestehende Arbeitsabläufe ein, statt sie zu ersetzen:

- **Tests existieren bereits.** Das `test_bindings`-Feld referenziert bestehende Test-Node-IDs. Kein neues Framework erforderlich.
- **JSON Schema existiert bereits.** Jeder Validator funktioniert.
- **Git existiert bereits.** Das `verified_against`-Feld ist ein Git-SHA.
- **CI existiert bereits.** Schema-Validierung läuft als CI-Schritt neben bestehenden Pipelines.

Eine Datei (`*-ai-layer.json`). Drei Felder pro Behauptung (`tier`, `test_bindings`, `verified_against`). Das sind die gesamten Integrationskosten.

### Die selbstreferentielle Eigenschaft

Der DRS ist der erste Standard, der sich selbst verifiziert. Die DRP-1-KI-Schicht enthält Behauptungen über den DRS. Diese Behauptungen tragen Falsifikationsprädikate. Diese Prädikate werden ausgewertet. Der `verified_against`-SHA stempelt die Verifikation. Der DRS ist sein eigener erster Anwender und sein eigener Machbarkeitsnachweis — selbstreferentiell in derselben Weise wie ein Compiler, der sich selbst kompiliert.

---

## 17. Ressourcen

- [DRS-Architekturspezifikation](https://github.com/thomasbrennan/Fracttalix/blob/main/paper/DRS-Architecture.md) — Vollständige Spezifikation
- [Falsifikationskernel v1.1](https://github.com/thomasbrennan/Fracttalix/blob/main/ai-layers/falsification-kernel.md) — Layer-0-Semantikspezifikation
- [KI-Schicht-Schema v3](https://github.com/thomasbrennan/Fracttalix/blob/main/ai-layers/ai-layer-schema.json) — JSON Schema
- [GVP-Spezifikation](https://github.com/thomasbrennan/Fracttalix/blob/main/docs/GVP-spec.md) — Software-Protokoll
- [Beispiel-KI-Schicht (P1)](https://github.com/thomasbrennan/Fracttalix/blob/main/ai-layers/P1-ai-layer.json) — Funktionierendes Beispiel
- [Fracttalix-Repository](https://github.com/thomasbrennan/Fracttalix)

---

**Lizenz:** CC BY 4.0 | **DOI:** [10.5281/zenodo.18859299](https://doi.org/10.5281/zenodo.18859299) | **Autor:** Thomas Brennan ([ORCID](https://orcid.org/0009-0002-6353-7115))
