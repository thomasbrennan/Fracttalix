# O Dual Reader Standard — Guia completo

**Um guia em uma leitura do Dual Reader Standard em sua totalidade: arquitetura, protocolos, núcleo, camadas, compatibilidade, lingua franca e adoção.**

> Veja também: [pt.json](pt.json) — Versão JSON-LD de dados estruturados deste guia para sistemas de IA e rastreadores web.

---

## 1. O que é o Dual Reader Standard

O **Dual Reader Standard (DRS)** é uma arquitetura de verificação para sistemas de conhecimento. Ele exige que toda afirmação — seja escrita em prosa ou implementada em código — seja legível por duas classes independentes de leitores: **humano** e **máquina**.

O DRS não é um artigo. Não é uma ferramenta. É o standard que contém seus dois protocolos:

- **DRP** (Dual Reader Protocol) — o protocolo para texto
- **GVP** (Grounded Verification Protocol) — o protocolo para software

O DRP torna as afirmações *avaliáveis por máquina*. O GVP torna-as *verificadas por máquina*. Nenhum dos dois é completo sem o outro. Juntos, eles constituem o DRS.

---

## 2. Os dois protocolos

### 2.1 DRP — Dual Reader Protocol (Texto)

O DRP rege a maneira como as afirmações em prosa se tornam avaliáveis por máquina.

**Leitor 1 (Humano):** Lê o artigo em linguagem natural. Compreende o contexto, a motivação e a narrativa. Não pode auditar sistematicamente cada afirmação.

**Leitor 2 (IA):** Lê a camada de IA — um documento JSON estruturado que acompanha cada artigo. Contém o registro completo de afirmações. Pode auditar cada afirmação sem ler a prosa.

O DRP exige:

1. **Classificação de afirmações.** Cada afirmação é tipada como A (axioma), D (definição) ou F (falsificável).
2. **Predicados de falsificação.** Cada afirmação do Tipo F carrega um predicado determinístico de 5 partes.
3. **Portas de fase.** Seis condições (c1–c6) que devem ser satisfeitas antes que um artigo seja declarado PHASE-READY.
4. **Rastreamento de placeholders.** As afirmações que dependem de resultados não resolvidos são registradas como placeholders — tornando as lacunas visíveis em vez de invisíveis.

**O que o DRP garante:** Todo sistema de IA com acesso à camada de IA pode avaliar qualquer afirmação falsificável sem ler a prosa. A autossuficiência é um requisito de design aplicado na porta de fase (condição c5).

### 2.2 GVP — Grounded Verification Protocol (Software)

O GVP rege a maneira como as afirmações avaliáveis por máquina se tornam verificadas por máquina.

**Leitor 3A (Desenvolvedor):** Lê o campo `tier` para entender que tipo de evidência existe. Lê `test_bindings` para saber quais testes exercitam quais afirmações. Lê `verified_against` para saber quando esses testes foram aprovados pela última vez.

**Leitor 3B (Máquina):** Executa o executor de testes sobre o array `test_bindings`. Registra aprovação/reprovação. Carimba o SHA `verified_against` em caso de sucesso.

O GVP exige que cada afirmação carregue três campos:

1. **`tier`** — o nível de verificação (um dos seis valores)
2. **`test_bindings`** — um array de IDs de nó de teste totalmente qualificados que exercitam a afirmação
3. **`verified_against`** — o SHA do commit git no qual esses testes foram aprovados pela última vez

**O que o GVP garante:** Para qualquer afirmação no corpus, é possível determinar (a) que tipo de evidência a fundamenta, (b) quais testes executáveis a exercitam e (c) em qual commit esses testes foram aprovados. Se a resposta para (a) é `empirical_pending`, a lacuna é visível. Se a resposta para (c) é `null`, nenhum teste de software a cobre.

---

## 3. A fundação compartilhada — Núcleo de Falsificação (Camada 0)

Os dois protocolos operam sobre a mesma fundação: o **Núcleo de Falsificação K = (P, O, M, B)**.

Esta é a **Camada 0** do DRS. Ela define o que um predicado de falsificação *significa* independentemente de qualquer formato de serialização (JSON, YAML ou codificações futuras). O campo `semantic_spec_url` de uma camada de IA aponta para esta especificação, tornando a conformidade verificável por máquina em vez de meramente declarada.

| Símbolo | Nome | Campo(s) JSON | Papel |
|---------|------|---------------|-------|
| **P** | Predicado | `FALSIFIED_IF` | Sentença lógica que, se VERDADEIRA, falsifica a afirmação |
| **O** | Operandos | `WHERE` | Definições tipadas de cada variável em P |
| **M** | Mecanismo | `EVALUATION` | Procedimento de avaliação finito e determinístico |
| **B** | Limites | `BOUNDARY` + `CONTEXT` | Semântica de limiares e justificativa |

O DRP cria o núcleo (atribui predicados a afirmações em prosa). O GVP vincula o núcleo a evidências executáveis (liga predicados a testes e commits). O núcleo é independente de substrato — funciona para artigos científicos, software e qualquer domínio futuro que formule afirmações falsificáveis.

### 3.1 Gramática do predicado (`FALSIFIED_IF`)

O predicado é uma sentença lógica composta de:

- **Variáveis nomeadas** (definidas em `WHERE`)
- **Operadores de comparação:** `<`, `>`, `<=`, `>=`, `=`, `!=`
- **Conectivos lógicos:** `AND`, `OR`, `NOT`
- **Quantificadores:** `EXISTS ... SUCH THAT`, `FOR ALL ... IN`
- **Operadores aritméticos:** `+`, `-`, `*`, `/`, `^`, `log10()`, `exp()`, `abs()`, `max()`, `min()`
- **Operadores de conjuntos:** `IN`, `∩`, `∪`, `|...|` (cardinalidade)
- **Aplicação de funções:** `f(x)` onde `f` é definida em `WHERE`

### 3.2 Restrições do predicado

1. **Determinismo.** Deve resultar em exatamente `TRUE` ou `FALSE` para qualquer atribuição válida de variáveis.
2. **Finitude.** Os quantificadores variam apenas sobre conjuntos finitos. Quantificadores universais não delimitados não são permitidos.
3. **Sem autorreferência.** Não deve referenciar o valor de verdade da própria afirmação nem criar dependências circulares.
4. **Completude.** Toda variável em `FALSIFIED_IF` deve ser definida em `WHERE`, e toda variável em `WHERE` deve aparecer em `FALSIFIED_IF`.

**Correspondência de vereditos:**
- P resulta em TRUE → o veredito da afirmação é **FALSIFIED**
- P resulta em FALSE → o veredito da afirmação é **NOT FALSIFIED**

### 3.3 Operandos (`WHERE`)

Cada chave no objeto `WHERE` nomeia uma variável. Seu valor é uma string com o formato:

```
<tipo> · <unidades> · <definição ou fonte>
```

| Campo | Obrigatório | Exemplos |
|-------|-------------|----------|
| **tipo** | Sim | `scalar`, `integer`, `binary`, `set`, `string`, `function` |
| **unidades** | Sim (usar `dimensionless` se adimensional) | `seconds`, `dimensionless`, `bits` |
| **definição** | Sim | `output of sort(input)`, `count of substrates with R² >= 0.85` |

Restrições: toda variável livre em P deve aparecer em O (completude), toda variável em O deve aparecer em P (sem órfãs), e cada variável deve ser computável a partir de dados ou derivações — não julgamento subjetivo.

### 3.4 Mecanismo (`EVALUATION`)

O campo de avaliação especifica *como* calcular o valor de verdade de P:

1. **Finito.** O procedimento termina em um número finito de passos (convencionalmente confirmado terminando com a palavra `finite`).
2. **Determinístico.** As mesmas entradas produzem o mesmo veredito.
3. **Reprodutível.** Um terceiro com acesso aos dados e ao código citados pode executar o procedimento de forma independente.

### 3.5 Limites (`BOUNDARY` + `CONTEXT`)

**BOUNDARY** especifica os casos limítrofes dos limiares: se os limiares são inclusivos ou exclusivos, e qual veredito se aplica na igualdade exata.

**CONTEXT** justifica cada limiar numérico e cada escolha de design no predicado: por que este valor, qual conhecimento de domínio o fundamenta, e se é derivado ou convencional.

### 3.6 Exemplo de predicado

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

---

## 4. Os três tipos de afirmação

| Tipo | Nome | Em artigos | Em software | Predicado |
|------|------|------------|-------------|-----------|
| **A** | Axioma / Pressuposto | Premissas aceitas da literatura | Requisitos de plataforma, contratos de dependência | `null` |
| **D** | Definição | Termos e procedimentos estipulativos | Assinaturas de tipo, estruturas de dados, esquemas | `null` |
| **F** | Falsificável | Teoremas, previsões empíricas | Garantias comportamentais, invariantes de correção | K = (P, O, M, B) completo |

As afirmações do Tipo D e do Tipo A carregam `"falsification_predicate": null`. Isso é correto — definições e axiomas não são falsificáveis por design.

---

## 5. Os seis níveis de verificação

Toda afirmação carrega um campo `tier` que declara qual tipo de evidência a fundamenta.

### Fundamentados por construção

| Nível | Tipo | Significado |
|-------|------|-------------|
| `axiom` | A | Premissa fundamental. Infalsificável por design. |
| `definition` | D | Definicional. Estipula um termo ou estrutura. Sem valor de verdade. |

### Fundamentados agora

| Nível | Tipo | Significado |
|-------|------|-------------|
| `software_tested` | F | Exercitado por testes aprovados. `test_bindings` não vazio, `verified_against` não nulo. |
| `formal_proof` | F | Derivação indexada por passos com `n_invalid_steps = 0`. A prova está na camada de IA. |
| `analytic` | F | Verificado por rastreamento de derivação formal ou argumento analítico. |

### Explicitamente não fundamentados

| Nível | Tipo | Significado |
|-------|------|-------------|
| `empirical_pending` | F | Placeholder ativo ou dados pendentes. A lacuna é visível por design. |

### Regras de consistência

O nível deve ser consistente com o tipo de afirmação e os campos do GVP:

| Nível | Tipo obrigatório | test_bindings | verified_against |
|-------|-----------------|---------------|------------------|
| `axiom` | A | `[]` (vazio) | `null` |
| `definition` | D | `[]` (vazio) | `null` |
| `software_tested` | F | Não vazio | Não nulo (7–40 caracteres hex) |
| `formal_proof` | F | Pode ser vazio | Pode ser nulo |
| `analytic` | F | Pode ser vazio | Pode ser nulo |
| `empirical_pending` | F | Pode ser vazio | Pode ser nulo |

---

## 6. A testemunha de vacuidade

Cada afirmação do Tipo F deve incluir um campo `sample_falsification_observation`: uma observação concreta e hipotética que *faria* o predicado resultar em TRUE.

Isso serve como verificação de vacuidade — prova de que o predicado não é trivialmente infalsificável. Um predicado que nenhuma observação concebível poderia satisfazer é verdadeiro de forma vacuosa e, portanto, não é uma afirmação do Tipo F válida.

Isso é imposto pela condição c6 da porta de fase.

---

## 7. Predicados de placeholder

Quando uma afirmação depende de resultados de um artigo que ainda não atingiu o status PHASE-READY, o predicado pode conter referências de placeholder:

- `placeholder: true` no objeto da afirmação
- `placeholder_id` vinculado ao `placeholder_register`
- O texto do predicado pode incluir `[PLACEHOLDER: pending ...]`

As afirmações de placeholder são válidas mas **não avaliáveis** até que a dependência seja resolvida. Elas não bloqueiam o status PHASE-READY do artigo que as contém, a menos que `blocks_phase_ready: true`.

---

## 8. A camada de IA — O artefato central

A camada de IA é o artefato central do DRS. É um documento JSON que acompanha cada artigo ou sistema de software. O esquema é definido em `ai-layers/ai-layer-schema.json`.

### Seções obrigatórias

| Seção | Finalidade |
|-------|-----------|
| `_meta` | Tipo de documento, versão do esquema, sessão, licença |
| `paper_id` / `paper_title` | Identidade |
| `paper_type` | Classificação: `law_A`, `derivation_B`, `application_C`, `methodology_D` |
| `phase_ready` | Veredito da porta de fase e status das condições (c1–c6) |
| `claim_registry` | Array de todas as afirmações com tipos, predicados, níveis, vinculações e SHAs |
| `placeholder_register` | Array de dependências não resolvidas |
| `summary` | Contagens e status das afirmações |
| `semantic_spec_url` | Aponta para o Núcleo de Falsificação (Camada 0) |

A camada de IA é o que faz ambos os protocolos funcionarem:

- O **DRP** exige que ela exista, contenha predicados e passe a porta de fase.
- O **GVP** exige que ela contenha tier, test_bindings e verified_against para cada afirmação.

---

## 9. A porta de fase

Um artigo ou release de software é **PHASE-READY** quando seis condições são satisfeitas:

| Condição | Requisito |
|----------|-----------|
| **c1** | A camada de IA é válida conforme o esquema |
| **c2** | Todas as afirmações falsificáveis registradas com predicados |
| **c3** | Todos os predicados são avaliáveis por máquina |
| **c4** | Referências cruzadas rastreadas (registro de placeholders) |
| **c5** | A verificação é autossuficiente (apenas a camada de IA, sem necessidade de prosa) |
| **c6** | Todos os predicados são não vacuosos (existe observação de falsificação de exemplo) |

**CORPUS-COMPLETE** é acionado quando todos os artigos são PHASE-READY e todos os placeholders de todos os objetos são resolvidos (c4 plenamente satisfeita em todo o corpus).

---

## 10. As regras de inferência

O DRS fornece um inventário canônico de regras de inferência para rastreamentos de derivação usados em afirmações de nível `formal_proof`:

| ID | Nome | Descrição |
|----|------|-----------|
| IR-1 | Modus Ponens | Se P e P→Q então Q |
| IR-2 | Instanciação universal | Se ∀x P(x) então P(a) para qualquer a específico |
| IR-3 | Substituição de iguais | Se a=b então substituir a por b |
| IR-4 | Expansão de definição | Substituir um termo definido por sua definição |
| IR-5 | Manipulação algébrica | Transformação algébrica válida preservando a igualdade |
| IR-6 | Equivalência lógica | Substituir por uma expressão logicamente equivalente |
| IR-7 | Inferência estatística | Aplicar um procedimento estatístico nomeado aos dados |
| IR-8 | Parcimônia / Seleção de princípio de modelagem | Selecionar valor canônico de uma família consistente com os axiomas |

Cada passo em uma tabela de derivação indexada por passos cita uma regra de inferência e lista suas premissas. Uma derivação é válida quando `n_invalid_steps = 0`. O inventário é somente de adição: novas regras podem ser adicionadas, as regras existentes nunca são modificadas ou removidas.

---

## 11. Pilha arquitetural

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

O núcleo é a fundação compartilhada. O DRP cria predicados a partir da prosa. O GVP vincula predicados a evidências executáveis. A camada de IA é o artefato que carrega ambos.

---

## 12. Como implementar

### Para um artigo (DRP)

1. Redigir o artigo (canal em prosa para leitores humanos)
2. Criar o arquivo JSON da camada de IA (canal de máquina para leitores de IA)
3. Classificar cada afirmação como A (axioma), D (definição) ou F (falsificável)
4. Escrever o predicado de falsificação de 5 partes para cada afirmação do Tipo F
5. Incluir uma `sample_falsification_observation` para cada afirmação do Tipo F (testemunha de vacuidade)
6. Atribuir o nível de verificação a cada afirmação
7. Validar contra `ai-layer-schema.json`
8. Executar a porta de fase (c1–c6)

### Para software (GVP)

1. Enumerar o que o software afirma fazer
2. Classificar cada afirmação como A (pressuposto), D (definição) ou F (comportamental)
3. Escrever o predicado de falsificação para cada afirmação do Tipo F
4. Escrever ou identificar os testes que exercitam cada afirmação
5. Preencher `test_bindings` com IDs de nó de teste totalmente qualificados
6. Executar os testes e registrar o SHA do commit aprovado em `verified_against`
7. Atribuir o nível: `software_tested` se os testes existem, `empirical_pending` se ainda não
8. Registrar quaisquer afirmações não testadas como placeholders
9. Validar contra `ai-layer-schema.json`

### Para ambos

A camada de IA é o mesmo artefato. O esquema é o mesmo. O núcleo é o mesmo. A única diferença é qual protocolo cria o conteúdo e qual protocolo o verifica.

### Adoção mínima viável

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

## 13. Princípios de design

**Epistemologia popperiana.** Podemos falsificar, mas não verificar. Uma afirmação que sobrevive a todas as tentativas de falsificação não está provada — ela sobreviveu.

**Lacunas honestas.** O placeholder é a funcionalidade mais importante. Quando uma afirmação é `empirical_pending`, o sistema diz: "afirmamos isto, mas ainda não verificamos." Isso é estritamente mais informativo do que a alternativa, onde afirmações não verificadas são indistinguíveis das verificadas.

**Independência de substrato.** O núcleo K = (P, O, M, B) não sabe se está avaliando um teorema científico ou uma garantia de software. Domínios futuros (jurídico, regulatório, políticas públicas) podem adicionar seus próprios protocolos sem modificar o núcleo.

**Máquina primeiro, legível por humanos.** A camada de IA é o artefato principal. O artigo em prosa e o código-fonte são canais secundários que fornecem contexto, narrativa e implementação. A camada de IA é o que é validado, auditado e versionado.

---

## 14. Compatibilidade em três eixos

Um standard que não sobrevive ao contato com o passado, o presente e o futuro não é um standard — é um instantâneo.

### O invariante no centro

O Núcleo de Falsificação K = (P, O, M, B) é o ponto fixo. Ele não sabe se está avaliando um teorema científico, uma garantia de software, um contrato jurídico ou um requisito regulatório. Não sabe se o ano é 2026 ou 2046. Não sabe se o formato de serialização é JSON, YAML ou algo que ainda não foi inventado.

O núcleo é permanente. Todo o resto é extensível.

### Eixo 1: Resistência ao passado (Retrocompatibilidade)

- **Versionamento de esquema.** Toda camada de IA registra contra qual versão do esquema foi produzida. Uma camada de IA produzida sob a versão v2 do esquema permanece válida sob a versão v2 para sempre. Um validador v3 pode ler uma camada v2 (novos campos são opcionais; campos antigos são preservados).
- **Permanência dos predicados.** Um predicado escrito em 2026 deve ser avaliável em 2036. A gramática do núcleo usa apenas operadores matemáticos e lógicos definidos permanentemente. O campo `WHERE` define cada variável inline. O campo `EVALUATION` especifica um procedimento autossuficiente.
- **Estabilidade das regras de inferência.** O inventário de regras de inferência (IR-1 a IR-8) é somente de adição. Regras existentes nunca são modificadas ou removidas.
- **Estabilidade dos níveis.** Os seis níveis de verificação são somente de adição. Níveis existentes nunca são removidos ou redefinidos.
- **O contrato:** Toda camada de IA que era válida no momento de sua criação permanecerá válida para sempre.

### Eixo 2: Resistência lateral (Compatibilidade entre domínios)

- **Independência de domínio.** O núcleo funciona em qualquer domínio de conhecimento:

| Domínio | Tipo A | Tipo D | Tipo F |
|---------|--------|--------|--------|
| Ciência | Premissas da literatura | Termos estipulativos | Teoremas, previsões |
| Software | Requisitos de plataforma | Assinaturas de tipo, esquemas | Garantias comportamentais |
| Jurídico | Autoridade estatutária | Termos definidos | Conclusões jurídicas |
| Regulatório | Pressupostos do quadro | Definições padrão | Declarações de conformidade |
| Políticas públicas | Premissas de valor | Termos de política | Previsões de impacto |
| Educação | Axiomas pedagógicos | Objetivos de aprendizagem | Afirmações de avaliação |

- **Independência de linguagem de programação.** O campo `test_bindings` aceita qualquer string que identifique unicamente um teste em qualquer framework: pytest, Jest, cargo test, go test, JUnit.
- **Independência de sistema de IA.** A camada de IA é um documento JSON. Qualquer sistema de IA — Claude, GPT, Gemini, Llama ou sistemas que ainda não existem — pode lê-lo. O campo `semantic_spec_url` aponta para a especificação do núcleo escrita em prosa.
- **Independência de serialização.** A Camada 0 é definida em prosa, não em JSON Schema. JSON é o transporte atual, mas a semântica do núcleo é independente da codificação. Implementações futuras poderiam usar YAML, Protocol Buffers, CBOR ou formatos ainda não inventados.
- **Independência de ferramentas.** O DRS se integra a fluxos de trabalho existentes: SHAs git (qualquer hospedagem), qualquer executor de testes, qualquer validador JSON Schema, qualquer pipeline de CI. Ele adiciona uma camada por cima — não substitui nada.
- **O contrato:** Adotar o DRS em um domínio, idioma, ferramenta ou sistema de IA não cria dependência.

### Eixo 3: Resistência ao futuro (Compatibilidade ascendente)

- **Extensibilidade de protocolos.** Novos domínios adicionam novos protocolos. Protocolos futuros (Protocolo de Verificação Jurídica, Protocolo de Verificação Regulatória, etc.) seguem o mesmo padrão: definir os leitores, a taxonomia de níveis e o mecanismo de vinculação — todos compartilhando o núcleo, os tipos de afirmação e o esquema da camada de IA.
- **Extensibilidade de níveis.** Domínios futuros podem precisar de níveis como `regulatory_certified`, `peer_reviewed`, `formally_verified`, `field_tested`, `community_validated`. Novos níveis são adicionados ao enum do esquema. Os níveis existentes permanecem.
- **Extensibilidade de regras de inferência.** IR-9, IR-10 e além podem ser adicionadas à medida que novos padrões de derivação surgem. Derivações antigas permanecem válidas porque citam regras por ID estável.
- **Extensibilidade de esquema.** O JSON Schema permite propriedades adicionais por padrão. A progressão: v1 (afirmações básicas), v2 (adição de `semantic_spec_url`), v3 (adição de campos GVP). Cada versão adiciona. Nenhuma remove.
- **Leitores futuros desconhecidos.** A camada de IA contém informações estruturadas suficientes para tipos de leitores que ainda não existem: agentes de verificação autônomos, verificadores entre corpora, mecanismos de conformidade regulatória, integrações de gerenciadores de pacotes.
- **O contrato:** Qualquer inovação futura pode ser adicionada como novo protocolo, nível, regra de inferência ou campo de esquema — sem modificar nada do que já existe.

### A garantia em três eixos

| Eixo | Promessa | Mecanismo |
|------|----------|-----------|
| **Passado** | Nada do que já foi feito quebra | Versionamento de esquema, enums somente de adição, núcleo permanente |
| **Lateral** | Funciona em todos os domínios, idiomas, ferramentas, sistemas de IA | Núcleo independente de substrato, vinculações tipadas por string, semântica definida em prosa |
| **Futuro** | Qualquer coisa nova pode ser adicionada sem redesenho | Extensibilidade de protocolo, extensibilidade de nível, evolução aditiva de esquema |

---

## 15. A lingua franca das máquinas

Esta é a propriedade mais profunda do DRS. Ela não foi projetada. Foi descoberta.

### O problema da tradução

O conhecimento científico está atualmente trancado atrás das línguas humanas. Um artigo em mandarim é invisível para um pesquisador que lê apenas em inglês — a menos que alguém o traduza. A tradução é cara, com perda de informação e lenta. O conhecimento se fragmenta ao longo de fronteiras linguísticas.

Isso não é um problema de formatação. É um problema de *substrato*. O conhecimento codificado em linguagem natural é não interoperável por natureza.

### O núcleo dissolve o problema

O Núcleo de Falsificação não é escrito em nenhuma língua humana. É escrito em lógica e matemática:

```
FALSIFIED_IF: R2_best_alt > R2_frm + 0.05
WHERE:
  R2_best_alt: scalar · dimensionless · best R² from competing models
  R2_frm:      scalar · dimensionless · R² from FRM regression
EVALUATION: Run regression for each model; compare R² values; finite
BOUNDARY: R2_best_alt = R2_frm + 0.05 → FALSIFIED (threshold inclusive)
CONTEXT: 0.05 margin from standard model comparison practice
```

Este predicado significa a mesma coisa para uma instância de Claude em inglês, uma instância de GPT em mandarim, uma instância de Gemini em francês e um sistema de IA ainda não construído operando em um idioma que ainda não existe. Nenhuma tradução necessária.

### JSON como camada de transporte

JSON é o formato de intercâmbio de dados de facto mundial — suportado por todas as linguagens de programação, analisado por todos os sistemas de IA, transmitido por todas as APIs. Ao codificar o núcleo em JSON, o DRS herda a universalidade do JSON:

- Uma equipe chinesa publica sua camada de IA. Os predicados usam notação matemática.
- Uma equipe brasileira lê a mesma camada. Não precisa de mandarim. Precisa de `>`, `+` e `R²`.
- Um sistema de IA em qualquer país avalia o predicado. O veredito é FALSIFIED ou NOT FALSIFIED. O veredito não tem sotaque.

**Qualificação importante.** As definições do campo `WHERE` atualmente contêm descrições em prosa inglesa. O *conteúdo operativo* — tipos de variáveis, unidades, operadores matemáticos, lógica de comparação, valores de limiares — é neutro linguisticamente. Dados valores numéricos fundamentados para as variáveis nomeadas, qualquer sistema pode avaliar o predicado independentemente de ler ou não as descrições. Um refinamento futuro poderia formalizar inteiramente as definições de variáveis em notação tipada.

### Lógica binária embutida

Todo predicado DRS se reduz a lógica binária embutida. Toda a cadeia de avaliação se colapsa em:

```
Afirmação → Predicado → Variáveis + Operadores → Booleano → 1 ou 0
```

O campo `WHERE` tipa as variáveis. O campo `FALSIFIED_IF` as combina com operadores. O campo `EVALUATION` especifica como calcular as entradas. O campo `BOUNDARY` resolve os casos limítrofes. A saída é sempre um único bit: FALSIFIED ou NOT FALSIFIED. `1` ou `0`.

O JSON é o contêiner. O núcleo é o circuito. Toda camada de IA é uma coleção de portas lógicas com entradas definidas e uma saída de um único bit por afirmação.

Não se pode traduzir incorretamente um `1` ou um `0`. Não se pode interpretar incorretamente `>`. Não se pode ter um desacordo cultural sobre se `R2_best_alt > R2_frm + 0.05` é TRUE ou FALSE para um dado par de valores. O significado está na estrutura, não em qualquer idioma.

A lógica binária é lógica binária. Um predicado que resulta em TRUE em Pequim resulta em TRUE em Boston, em São Paulo, em Lagos e em um servidor sem qualquer configuração de locale. O DRS não *traduz* conhecimento entre idiomas. Ele *codifica* conhecimento em um substrato que precede e transcende todas as línguas humanas — o substrato da própria lógica.

---

## 16. Adoção autopropagada

### O incentivo interno

O DRS deve se propagar porque o *criador* se beneficia, não apenas o consumidor. Criar uma camada de IA obriga você a enumerar o que afirma. Esse processo descobre:

- Afirmações que você pensava estarem testadas, mas não estão (descoberta de lacunas)
- Condições limítrofes que você nunca documentou
- Dependências que você presumia, mas nunca registrou

A camada de IA é um efeito colateral de um processo que melhora sua própria compreensão do seu próprio sistema. As lacunas existem independentemente de você documentá-las ou não. O DRS simplesmente as torna visíveis.

### O efeito de rede

O DRS se torna mais valioso à medida que mais sistemas o adotam:

- **Cadeias de dependência tornam-se conscientes de afirmações.** Se a biblioteca A publica uma camada de IA e a biblioteca B depende de A, então B pode determinar programaticamente quais de suas afirmações dependem de quais pressupostos de A. Quando A lança uma mudança incompatível, B sabe exatamente quais afirmações estão em risco.
- **Sistemas de IA podem auditar entre projetos.** Um leitor de IA pode percorrer múltiplas camadas de IA, verificar referências cruzadas e identificar inconsistências em todo um ecossistema.
- **A confiança torna-se auditável.** Em vez de confiar em uma biblioteca por causa de estrelas ou downloads, você confia nela porque sua camada de IA mostra quais afirmações são `software_tested`, quais são `empirical_pending` e qual é o SHA de `verified_against`. A confiança passa do sinal social para a evidência estrutural.

### A IA como catalisador de adoção

1. **A IA gera a camada de IA inicial.** Dado um código-fonte, uma IA pode enumerar afirmações, classificá-las, escrever predicados e identificar vinculações de teste. O humano revisa e corrige. O custo cai de horas para minutos.
2. **A IA mantém a camada.** Quando o código muda, a IA atualiza o registro de afirmações, ajusta as vinculações de teste e sinaliza SHAs obsoletos. O humano aprova.
3. **A IA audita outras camadas.** Uma IA que lê a camada de IA de uma dependência pode determinar de quais pressupostos suas próprias afirmações dependem e sinalizar riscos automaticamente.

O DRS é o protocolo que torna o desenvolvimento assistido por IA *auditável*. Sem ele, a IA gera código e os humanos esperam que funcione. Com ele, a IA gera código e o registro de afirmações diz exatamente o que foi verificado e o que não foi.

### A estratégia de integração

O DRS se integra a fluxos de trabalho existentes em vez de substituí-los:

- **Os testes já existem.** O campo `test_bindings` referencia IDs de nó de teste existentes. Nenhum novo framework necessário.
- **JSON Schema já existe.** Qualquer validador funciona.
- **Git já existe.** O campo `verified_against` é um SHA git.
- **O CI já existe.** A validação de esquema roda como uma etapa de CI junto dos pipelines existentes.

Um arquivo (`*-ai-layer.json`). Três campos por afirmação (`tier`, `test_bindings`, `verified_against`). Este é o custo total de integração.

### A propriedade autorreferencial

O DRS é o primeiro standard que verifica a si mesmo. A camada de IA DRP-1 contém afirmações sobre o DRS. Essas afirmações carregam predicados de falsificação. Esses predicados são avaliados. O SHA `verified_against` carimba a verificação. O DRS é seu próprio primeiro adotante e sua própria prova de conceito — autorreferencial da mesma maneira que um compilador que se compila a si mesmo é autorreferencial.

---

## 17. Recursos

- [Especificação da Arquitetura DRS](https://github.com/thomasbrennan/Fracttalix/blob/main/paper/DRS-Architecture.md) — Especificação completa
- [Núcleo de Falsificação v1.1](https://github.com/thomasbrennan/Fracttalix/blob/main/ai-layers/falsification-kernel.md) — Especificação semântica da Camada 0
- [Esquema da Camada de IA v3](https://github.com/thomasbrennan/Fracttalix/blob/main/ai-layers/ai-layer-schema.json) — JSON Schema
- [Especificação GVP](https://github.com/thomasbrennan/Fracttalix/blob/main/docs/GVP-spec.md) — Protocolo de software
- [Exemplo de Camada de IA (P1)](https://github.com/thomasbrennan/Fracttalix/blob/main/ai-layers/P1-ai-layer.json) — Exemplo funcional
- [Repositório Fracttalix](https://github.com/thomasbrennan/Fracttalix)

---

**Licença:** CC BY 4.0 | **DOI:** [10.5281/zenodo.18859299](https://doi.org/10.5281/zenodo.18859299) | **Autor:** Thomas Brennan ([ORCID](https://orcid.org/0009-0002-6353-7115))
