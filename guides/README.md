# Dual Reader Standard (DRS) — Multilingual Implementation Guides

**Machine-verifiable claims for scientific papers, software, and knowledge systems.**

The [Dual Reader Standard](https://github.com/thomasbrennan/Fracttalix/blob/main/paper/DRS-Architecture.md) is a verification architecture that makes every claim in a knowledge system readable by two independent reader classes — human and machine — using deterministic falsification predicates encoded in JSON.

Each guide below is provided in two formats:
- **JSON-LD** (`.json`) — machine-readable structured data with [schema.org](https://schema.org) markup. AI-native. Web crawler optimized.
- **Markdown** (`.md`) — human-readable guide rendered on GitHub. SEO-optimized with structured headings.

---

## Guides by Language

| Language | Native Name | Speakers | JSON-LD | Markdown |
|----------|------------|----------|---------|----------|
| English | English | ~1.5B | [en.json](en.json) | [en.md](en.md) |
| Chinese | 中文 | ~1.1B | [zh.json](zh.json) | [zh.md](zh.md) |
| Hindi | हिन्दी | ~600M | [hi.json](hi.json) | [hi.md](hi.md) |
| Spanish | Español | ~550M | [es.json](es.json) | [es.md](es.md) |
| French | Français | ~300M | [fr.json](fr.json) | [fr.md](fr.md) |
| Arabic | العربية | ~270M | [ar.json](ar.json) | [ar.md](ar.md) |
| Bengali | বাংলা | ~270M | [bn.json](bn.json) | [bn.md](bn.md) |
| Portuguese | Português | ~260M | [pt.json](pt.json) | [pt.md](pt.md) |
| Russian | Русский | ~250M | [ru.json](ru.json) | [ru.md](ru.md) |
| Indonesian | Bahasa Indonesia | ~200M | [id.json](id.json) | [id.md](id.md) |
| German | Deutsch | ~130M | [de.json](de.json) | [de.md](de.md) |
| Japanese | 日本語 | ~125M | [ja.json](ja.json) | [ja.md](ja.md) |
| Korean | 한국어 | ~80M | [ko.json](ko.json) | [ko.md](ko.md) |

---

## Why JSON?

The Dual Reader Standard's central insight (Section 18 of the [Architecture Specification](https://github.com/thomasbrennan/Fracttalix/blob/main/paper/DRS-Architecture.md)):

> The Falsification Kernel K = (P, O, M, B) is not written in any human language. It is written in logic and mathematics. A predicate that evaluates to TRUE in Beijing evaluates to TRUE in Boston.

JSON is the machine lingua franca. Every AI system — Claude, GPT, Gemini, Llama, or systems that don't exist yet — reads JSON natively. By providing these guides as JSON-LD structured data, any AI system in any country can parse, understand, and act on the Dual Reader Standard without translation.

The Markdown guides serve human readers and web crawlers. The JSON-LD guides serve machines and AI systems. Together, they implement the Dual Reader Standard's own principle: every claim readable by both human and machine.

---

## The Falsification Kernel K = (P, O, M, B)

Every falsifiable claim in the DRS carries a deterministic predicate:

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

This predicate means the same thing to every AI system in every language. No translation required.

---

## Resources

- [DRS Architecture Specification](https://github.com/thomasbrennan/Fracttalix/blob/main/paper/DRS-Architecture.md)
- [Falsification Kernel v1.1 (Layer 0)](https://github.com/thomasbrennan/Fracttalix/blob/main/ai-layers/falsification-kernel.md)
- [AI Layer Schema v3](https://github.com/thomasbrennan/Fracttalix/blob/main/ai-layers/ai-layer-schema.json)
- [GVP Specification](https://github.com/thomasbrennan/Fracttalix/blob/main/docs/GVP-spec.md)
- [Example AI Layer (P1)](https://github.com/thomasbrennan/Fracttalix/blob/main/ai-layers/P1-ai-layer.json)
- [Claim Registry Index](https://github.com/thomasbrennan/Fracttalix/blob/main/ai-layers/claim-registry-index.md)

---

## License

CC BY 4.0 — consistent with the Fracttalix corpus.

DOI: [10.5281/zenodo.18859299](https://doi.org/10.5281/zenodo.18859299)
