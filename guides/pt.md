# Como Implementar o Dual Reader Standard

**Um guia completo para construir afirmações verificáveis por máquina para artigos científicos, software e sistemas de conhecimento.**

> Veja também: [pt.json](pt.json) — Versão em dados estruturados JSON-LD deste guia para sistemas de IA e rastreadores web.

---

## O Que É o Dual Reader Standard?

O **Dual Reader Standard (DRS)** é uma arquitetura de verificação para sistemas de conhecimento. Toda afirmação — seja escrita em prosa ou implementada em código — deve ser legível por duas classes independentes de leitores: **humano** e **máquina**.

O DRS compreende dois protocolos:

| Protocolo | Domínio | Função |
|-----------|---------|--------|
| **DRP** (Dual Reader Protocol) | Texto / Artigos | Torna afirmações em prosa avaliáveis por máquina por meio de predicados de falsificação de 5 partes |
| **GVP** (Grounded Verification Protocol) | Software / Código | Torna afirmações avaliáveis por máquina em afirmações verificadas por máquina, por meio de vinculações de teste e evidências fixadas em commits |

### Os Três Leitores

| Leitor | Canal | Lê | Formato |
|--------|-------|-----|---------|
| **Humano** | Prosa | O artigo | Linguagem natural |
| **IA** | JSON | A camada de IA | Registro estruturado de afirmações |
| **CI / executor de testes** | Executável | Vinculações de teste | IDs de nó de teste + SHA do commit |

---

## O Núcleo de Falsificação K = (P, O, M, B)

A base compartilhada de ambos os protocolos. Toda afirmação do Tipo F (falsificável) carrega um predicado determinístico que resulta em exatamente um de dois vereditos: **FALSIFIED** ou **NOT FALSIFIED**.

| Símbolo | Nome | Campo JSON | Papel |
|---------|------|------------|-------|
| **P** | Predicado | `FALSIFIED_IF` | Sentença lógica que, se VERDADEIRA, falsifica a afirmação |
| **O** | Operandos | `WHERE` | Definições tipadas de cada variável no predicado |
| **M** | Mecanismo | `EVALUATION` | Procedimento de avaliação finito e determinístico |
| **B** | Limites | `BOUNDARY` + `CONTEXT` | Semântica de limiares e justificativa |

### Exemplo de Predicado

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

Este predicado significa a mesma coisa para todo sistema de IA em qualquer idioma. Nenhuma tradução necessária.

### Restrições do Predicado

- **Determinismo**: Deve resultar em exatamente VERDADEIRO ou FALSO
- **Finitude**: Quantificadores variam apenas sobre conjuntos finitos
- **Sem autorreferência**: Sem dependências circulares de predicados
- **Completude**: Toda variável em `FALSIFIED_IF` definida em `WHERE`, e vice-versa

---

## Os Três Tipos de Afirmação

| Tipo | Nome | Descrição | Predicado Obrigatório? |
|------|------|-----------|----------------------|
| **A** | Axioma | Premissas fundamentais — infalsificáveis por design | Não (`null`) |
| **D** | Definição | Definições estipulativas — sem valor de verdade | Não (`null`) |
| **F** | Falsificável | Afirmações testáveis com predicados determinísticos | Sim — K = (P, O, M, B) completo |

---

## Os Seis Níveis de Verificação

Toda afirmação carrega um campo `tier` que declara qual tipo de evidência a sustenta.

### Fundamentado por Construção

| Nível | Aplica-se a | Significado |
|-------|-------------|-------------|
| `axiom` | Tipo A | Fundamental, infalsificável por design |
| `definition` | Tipo D | Definicional, sem necessidade de predicado |

### Fundamentado Agora

| Nível | Aplica-se a | Significado |
|-------|-------------|-------------|
| `software_tested` | Tipo F | Exercitado por testes aprovados. `test_bindings` não vazio, `verified_against` SHA não nulo |
| `formal_proof` | Tipo F | Derivação indexada por passos com `n_invalid_steps = 0` |
| `analytic` | Tipo F | Verificado por rastreamento de derivação formal ou argumento analítico |

### Explicitamente Não Fundamentado

| Nível | Aplica-se a | Significado |
|-------|-------------|-------------|
| `empirical_pending` | Tipo F | Placeholder ativo ou dados pendentes. A lacuna é visível por design |

---

## A Camada de IA — O Artefato Central

A camada de IA é um documento JSON que acompanha cada artigo ou sistema de software. É sobre ela que ambos os protocolos operam.

### Seções Obrigatórias

| Seção | Finalidade |
|-------|-----------|
| `_meta` | Tipo de documento, versão do esquema, sessão, licença |
| `paper_id` | Identificador único |
| `paper_title` | Título legível por humanos |
| `paper_type` | `law_A`, `derivation_B`, `application_C` ou `methodology_D` |
| `phase_ready` | Veredito do phase gate e status das condições (c1–c6) |
| `claim_registry` | Array de todas as afirmações com tipos, predicados, níveis, vinculações |
| `placeholder_register` | Array de dependências não resolvidas |

### Esquema

O esquema da camada de IA está disponível em: [`ai-layers/ai-layer-schema.json`](https://github.com/thomasbrennan/Fracttalix/blob/main/ai-layers/ai-layer-schema.json)

---

## O Phase Gate de Seis Condições

Um artigo ou release de software é **PHASE-READY** quando todas as seis condições são satisfeitas:

| Condição | Requisito |
|----------|-----------|
| **c1** | A camada de IA é válida conforme o esquema |
| **c2** | Todas as afirmações falsificáveis registradas com predicados |
| **c3** | Todos os predicados são avaliáveis por máquina |
| **c4** | Referências cruzadas rastreadas (registro de placeholders) |
| **c5** | A verificação é autossuficiente (apenas a camada de IA, sem necessidade de prosa) |
| **c6** | Todos os predicados são não vacuosos (existe observação de falsificação de exemplo) |

---

## Como Implementar

### Para um Artigo (DRP)

1. Escreva o artigo (canal em prosa para leitores humanos)
2. Crie o arquivo JSON da camada de IA (canal de máquina para leitores de IA)
3. Classifique cada afirmação como A (axioma), D (definição) ou F (falsificável)
4. Escreva o predicado de falsificação de 5 partes para cada afirmação do Tipo F
5. Inclua uma `sample_falsification_observation` para cada afirmação do Tipo F (testemunha de vacuidade)
6. Atribua o nível de verificação a cada afirmação
7. Valide contra `ai-layer-schema.json`
8. Execute o phase gate (c1–c6)

### Para Software (GVP)

1. Enumere o que o software afirma fazer
2. Classifique cada afirmação como A (pressuposto), D (definição) ou F (comportamental)
3. Escreva o predicado de falsificação para cada afirmação do Tipo F
4. Escreva ou identifique os testes que exercitam cada afirmação
5. Preencha `test_bindings` com IDs de nó de teste totalmente qualificados
6. Execute os testes e registre o SHA do commit aprovado em `verified_against`
7. Atribua o nível: `software_tested` se os testes existem, `empirical_pending` se ainda não
8. Registre quaisquer afirmações não testadas como placeholders
9. Valide contra `ai-layer-schema.json`

### Adoção Mínima Viável

A menor adoção útil do DRS é **uma afirmação do Tipo F com uma vinculação de teste**:

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

Uma afirmação. Um teste. Um SHA. O DRS está ativo. Adicione mais afirmações quando o valor justificar o custo.

---

## Princípios de Design

**Epistemologia popperiana.** Podemos falsificar, mas não verificar. Uma afirmação que sobrevive a todas as tentativas de falsificação não está provada — ela sobreviveu.

**Lacunas honestas.** O placeholder é a funcionalidade mais importante. Quando uma afirmação é `empirical_pending`, o sistema diz: "afirmamos isto, mas ainda não verificamos." Isso é estritamente mais informativo do que a alternativa, na qual afirmações não verificadas são indistinguíveis das verificadas.

**Independência de substrato.** O núcleo K = (P, O, M, B) não sabe se está avaliando um teorema científico ou uma garantia de software. Domínios futuros (jurídico, regulatório, políticas públicas) podem adicionar seus próprios protocolos sem modificar o núcleo.

**Lingua franca de máquina.** O núcleo é escrito em lógica e matemática, não em nenhuma língua humana. Um predicado que resulta em TRUE em Pequim resulta em TRUE em Boston. JSON é a camada de transporte. Lógica binária é o substrato.

---

## Compatibilidade em Três Eixos

| Eixo | Promessa | Mecanismo |
|------|----------|-----------|
| **Passado** | Nada do que já foi feito quebra | Versionamento de esquema, enums somente com adição, núcleo permanente |
| **Lateral** | Funciona em todos os domínios, idiomas, ferramentas e sistemas de IA | Núcleo independente de substrato, vinculações tipadas por string |
| **Futuro** | Qualquer coisa nova pode ser adicionada sem redesenho | Extensibilidade de protocolo, extensibilidade de nível, evolução aditiva de esquema |

---

## Recursos

- [Especificação da Arquitetura DRS](https://github.com/thomasbrennan/Fracttalix/blob/main/paper/DRS-Architecture.md) — Especificação completa
- [Núcleo de Falsificação v1.1](https://github.com/thomasbrennan/Fracttalix/blob/main/ai-layers/falsification-kernel.md) — Especificação semântica da Camada 0
- [Esquema da Camada de IA v3](https://github.com/thomasbrennan/Fracttalix/blob/main/ai-layers/ai-layer-schema.json) — JSON Schema
- [Especificação GVP](https://github.com/thomasbrennan/Fracttalix/blob/main/docs/GVP-spec.md) — Protocolo de software
- [Exemplo de Camada de IA (P1)](https://github.com/thomasbrennan/Fracttalix/blob/main/ai-layers/P1-ai-layer.json) — Exemplo funcional
- [Repositório Fracttalix](https://github.com/thomasbrennan/Fracttalix)

---

**Licença:** CC BY 4.0 | **DOI:** [10.5281/zenodo.18859299](https://doi.org/10.5281/zenodo.18859299) | **Autor:** Thomas Brennan ([ORCID](https://orcid.org/0009-0002-6353-7115))
