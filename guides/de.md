# So implementieren Sie den Dual Reader Standard

**Ein vollständiger Leitfaden zur Erstellung maschinenverifizierbarer Behauptungen für wissenschaftliche Publikationen, Software und Wissenssysteme.**

> Siehe auch: [de.json](de.json) — JSON-LD-Strukturdatenversion dieses Leitfadens für KI-Systeme und Web-Crawler.

---

## Was ist der Dual Reader Standard?

Der **Dual Reader Standard (DRS)** ist eine Verifikationsarchitektur für Wissenssysteme. Jede Behauptung — ob in Prosa verfasst oder in Code implementiert — muss von zwei unabhängigen Leserklassen lesbar sein: **Mensch** und **Maschine**.

Der DRS umfasst zwei Protokolle:

| Protokoll | Domäne | Funktion |
|-----------|--------|----------|
| **DRP** (Dual Reader Protocol) | Text / Publikationen | Macht Prosa-Behauptungen maschinenauswertbar mittels fünfteiliger Falsifikationsprädikate |
| **GVP** (Grounded Verification Protocol) | Software / Code | Macht maschinenauswertbare Behauptungen maschinenverifiziert mittels Test-Bindungen und Commit-gepinnter Evidenz |

### Die drei Leser

| Leser | Kanal | Liest | Format |
|-------|-------|-------|--------|
| **Mensch** | Prosa | Die Publikation | Natürliche Sprache |
| **KI** | JSON | Die KI-Schicht | Strukturiertes Behauptungsregister |
| **CI / Testrunner** | Ausführbar | Test-Bindungen | Test-Knoten-IDs + Commit-SHA |

---

## Der Falsifikationskern K = (P, O, M, B)

Das gemeinsame Fundament beider Protokolle. Jede Behauptung vom Typ F (falsifizierbar) trägt ein deterministisches Prädikat, das zu genau einem von zwei Urteilen ausgewertet wird: **FALSIFIED** oder **NOT FALSIFIED**.

| Symbol | Name | JSON-Feld | Rolle |
|--------|------|-----------|-------|
| **P** | Prädikat | `FALSIFIED_IF` | Logischer Satz, der — wenn WAHR — die Behauptung falsifiziert |
| **O** | Operanden | `WHERE` | Typisierte Definitionen jeder Variable im Prädikat |
| **M** | Mechanismus | `EVALUATION` | Endliches, deterministisches Auswertungsverfahren |
| **B** | Grenzen | `BOUNDARY` + `CONTEXT` | Schwellwert-Semantik und Begründung |

### Beispielprädikat

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

Dieses Prädikat bedeutet für jedes KI-System in jeder Sprache dasselbe. Keine Übersetzung erforderlich.

### Einschränkungen für Prädikate

- **Determinismus**: Muss zu genau WAHR oder FALSCH auswerten
- **Endlichkeit**: Quantoren erstrecken sich nur über endliche Mengen
- **Kein Selbstbezug**: Keine zirkulären Prädikat-Abhängigkeiten
- **Vollständigkeit**: Jede Variable in `FALSIFIED_IF` muss in `WHERE` definiert sein und umgekehrt

---

## Die drei Behauptungstypen

| Typ | Name | Beschreibung | Prädikat erforderlich? |
|-----|------|--------------|----------------------|
| **A** | Axiom | Grundlegende Prämissen — konzeptionell unfalsifizierbar | Nein (`null`) |
| **D** | Definition | Stipulative Definitionen — nicht wahrheitsfähig | Nein (`null`) |
| **F** | Falsifizierbar | Überprüfbare Behauptungen mit deterministischen Prädikaten | Ja — vollständiger Kern K = (P, O, M, B) |

---

## Die sechs Verifikationsstufen

Jede Behauptung trägt ein `tier`-Feld, das angibt, welche Art von Evidenz sie begründet.

### Konstruktionsbedingt begründet

| Stufe | Gilt für | Bedeutung |
|-------|----------|-----------|
| `axiom` | Typ A | Grundlegend, konzeptionell unfalsifizierbar |
| `definition` | Typ D | Definitional, kein Prädikat erforderlich |

### Jetzt begründet

| Stufe | Gilt für | Bedeutung |
|-------|----------|-----------|
| `software_tested` | Typ F | Durch bestandene Tests überprüft. `test_bindings` nicht leer, `verified_against` SHA nicht null |
| `formal_proof` | Typ F | Schrittindizierte Ableitung mit `n_invalid_steps = 0` |
| `analytic` | Typ F | Verifiziert durch formale Ableitungsspur oder analytisches Argument |

### Explizit unbegründet

| Stufe | Gilt für | Bedeutung |
|-------|----------|-----------|
| `empirical_pending` | Typ F | Aktiver Platzhalter oder ausstehende Daten. Die Lücke ist konstruktionsbedingt sichtbar |

---

## Die KI-Schicht — das zentrale Artefakt

Die KI-Schicht ist ein JSON-Dokument, das jede Publikation oder jedes Softwaresystem begleitet. Sie ist das Objekt, auf dem beide Protokolle operieren.

### Erforderliche Abschnitte

| Abschnitt | Zweck |
|-----------|-------|
| `_meta` | Dokumenttyp, Schemaversion, Sitzung, Lizenz |
| `paper_id` | Eindeutiger Bezeichner |
| `paper_title` | Menschenlesbarer Titel |
| `paper_type` | `law_A`, `derivation_B`, `application_C` oder `methodology_D` |
| `phase_ready` | Phase-Gate-Urteil und Bedingungsstatus (c1–c6) |
| `claim_registry` | Array aller Behauptungen mit Typen, Prädikaten, Stufen, Bindungen |
| `placeholder_register` | Array ungelöster Abhängigkeiten |

### Schema

Das KI-Schicht-Schema ist verfügbar unter: [`ai-layers/ai-layer-schema.json`](https://github.com/thomasbrennan/Fracttalix/blob/main/ai-layers/ai-layer-schema.json)

---

## Das Sechs-Bedingungen-Phase-Gate

Eine Publikation oder Softwareversion ist **PHASE-READY**, wenn alle sechs Bedingungen erfüllt sind:

| Bedingung | Anforderung |
|-----------|-------------|
| **c1** | KI-Schicht ist schemakonform |
| **c2** | Alle falsifizierbaren Behauptungen mit Prädikaten registriert |
| **c3** | Alle Prädikate sind maschinenauswertbar |
| **c4** | Querverweise erfasst (Platzhalterregister) |
| **c5** | Verifikation ist eigenständig (KI-Schicht allein, kein Prosatext nötig) |
| **c6** | Alle Prädikate sind nicht-vakuös (Beispiel-Falsifikationsbeobachtung vorhanden) |

---

## So implementieren Sie den DRS

### Für eine Publikation (DRP)

1. Verfassen Sie die Publikation (Prosakanal für menschliche Leser)
2. Erstellen Sie die KI-Schicht-JSON-Datei (Maschinenkanal für KI-Leser)
3. Klassifizieren Sie jede Behauptung als A (Axiom), D (Definition) oder F (falsifizierbar)
4. Schreiben Sie das fünfteilige Falsifikationsprädikat für jede Behauptung vom Typ F
5. Fügen Sie eine `sample_falsification_observation` für jede Behauptung vom Typ F ein (Vakuität-Zeuge)
6. Weisen Sie jeder Behauptung die Verifikationsstufe zu
7. Validieren Sie gegen `ai-layer-schema.json`
8. Führen Sie das Phase-Gate aus (c1–c6)

### Für Software (GVP)

1. Listen Sie auf, was die Software zu leisten beansprucht
2. Klassifizieren Sie jede Behauptung als A (Annahme), D (Definition) oder F (verhaltensbezogen)
3. Schreiben Sie das Falsifikationsprädikat für jede Behauptung vom Typ F
4. Schreiben oder identifizieren Sie die Tests, die jede Behauptung überprüfen
5. Befüllen Sie `test_bindings` mit vollqualifizierten Test-Knoten-IDs
6. Führen Sie die Tests aus und tragen Sie den SHA des bestandenen Commits in `verified_against` ein
7. Weisen Sie die Stufe zu: `software_tested` wenn Tests existieren, `empirical_pending` wenn noch nicht
8. Registrieren Sie alle ungetesteten Behauptungen als Platzhalter
9. Validieren Sie gegen `ai-layer-schema.json`

### Minimale praktikable Einführung

Die kleinste nützliche DRS-Einführung ist **eine Behauptung vom Typ F mit einer Test-Bindung**:

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

Eine Behauptung. Ein Test. Ein SHA. Der DRS ist aktiv. Fügen Sie weitere Behauptungen hinzu, wenn der Nutzen die Kosten rechtfertigt.

---

## Gestaltungsprinzipien

**Poppersche Erkenntnistheorie.** Wir können falsifizieren, aber nicht verifizieren. Eine Behauptung, die alle Falsifikationsversuche übersteht, ist nicht bewiesen — sie hat überlebt.

**Ehrliche Lücken.** Der Platzhalter ist das wichtigste Merkmal. Wenn eine Behauptung `empirical_pending` ist, sagt das System: „Wir behaupten dies, haben es aber noch nicht verifiziert." Das ist strikt informativer als die Alternative, bei der unverifizierte Behauptungen nicht von verifizierten zu unterscheiden sind.

**Substratunabhängigkeit.** Der Kern K = (P, O, M, B) weiß nicht, ob er ein wissenschaftliches Theorem oder eine Softwaregarantie auswertet. Zukünftige Domänen (Recht, Regulierung, Politik) können eigene Protokolle hinzufügen, ohne den Kern zu verändern.

**Maschinenübergreifende Lingua franca.** Der Kern ist in Logik und Mathematik geschrieben, nicht in einer menschlichen Sprache. Ein Prädikat, das in Peking zu WAHR auswertet, wertet auch in Boston zu WAHR aus. JSON ist die Transportschicht. Binäre Logik ist das Substrat.

---

## Drei-Achsen-Kompatibilität

| Achse | Versprechen | Mechanismus |
|-------|-------------|-------------|
| **Vergangenheit** | Nichts bereits Geleistetes wird beschädigt | Schemaversionierung, nur-anhängende Aufzählungen, permanenter Kern |
| **Lateral** | Funktioniert über alle Domänen, Sprachen, Werkzeuge, KI-Systeme | Substratunabhängiger Kern, stringtypisierte Bindungen |
| **Zukunft** | Alles Neue kann ohne Neugestaltung hinzugefügt werden | Protokollerweiterbarkeit, Stufenerweiterbarkeit, additive Schemaentwicklung |

---

## Ressourcen

- [DRS-Architekturspezifikation](https://github.com/thomasbrennan/Fracttalix/blob/main/paper/DRS-Architecture.md) — Vollständige Spezifikation
- [Falsifikationskern v1.1](https://github.com/thomasbrennan/Fracttalix/blob/main/ai-layers/falsification-kernel.md) — Schicht-0-Semantikspezifikation
- [KI-Schicht-Schema v3](https://github.com/thomasbrennan/Fracttalix/blob/main/ai-layers/ai-layer-schema.json) — JSON-Schema
- [GVP-Spezifikation](https://github.com/thomasbrennan/Fracttalix/blob/main/docs/GVP-spec.md) — Softwareprotokoll
- [Beispiel-KI-Schicht (P1)](https://github.com/thomasbrennan/Fracttalix/blob/main/ai-layers/P1-ai-layer.json) — Funktionierendes Beispiel
- [Fracttalix-Repository](https://github.com/thomasbrennan/Fracttalix)

---

**Lizenz:** CC BY 4.0 | **DOI:** [10.5281/zenodo.18859299](https://doi.org/10.5281/zenodo.18859299) | **Autor:** Thomas Brennan ([ORCID](https://orcid.org/0009-0002-6353-7115))
