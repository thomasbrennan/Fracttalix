# 如何实施双重读者标准

**构建面向科学论文、软件和知识系统的机器可验证声明的完整指南。**

> 另见：[zh.json](zh.json) — 本指南的 JSON-LD 结构化数据版本，供 AI 系统和网络爬虫使用。

---

## 什么是双重读者标准？

**双重读者标准（DRS）** 是一种用于知识系统的验证架构。每一个声明——无论是以散文形式书写还是以代码形式实现——都必须能被两个独立的读者类别所读取：**人类**和**机器**。

DRS 包含两个协议：

| 协议 | 领域 | 功能 |
|------|------|------|
| **DRP**（双重读者协议） | 文本 / 论文 | 通过五部分证伪谓词使散文形式的声明可被机器评估 |
| **GVP**（锚定验证协议） | 软件 / 代码 | 通过测试绑定和提交锚定的证据，使机器可评估的声明实现机器验证 |

### 三类读者

| 读者 | 通道 | 读取内容 | 格式 |
|------|------|----------|------|
| **人类** | 散文 | 论文 | 自然语言 |
| **AI** | JSON | AI 层 | 结构化声明注册表 |
| **CI / 测试运行器** | 可执行文件 | 测试绑定 | 测试节点 ID + 提交 SHA |

---

## 证伪内核 K = (P, O, M, B)

两个协议的共同基础。每一个 F 类（可证伪）声明都携带一个确定性谓词，该谓词的求值结果恰好为两个判定之一：**已证伪（FALSIFIED）** 或 **未证伪（NOT FALSIFIED）**。

| 符号 | 名称 | JSON 字段 | 作用 |
|------|------|-----------|------|
| **P** | 谓词 | `FALSIFIED_IF` | 如果为 TRUE 则证伪该声明的逻辑语句 |
| **O** | 操作数 | `WHERE` | 谓词中每个变量的类型化定义 |
| **M** | 机制 | `EVALUATION` | 有限的、确定性的求值过程 |
| **B** | 边界 | `BOUNDARY` + `CONTEXT` | 阈值语义和依据说明 |

### 谓词示例

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

此谓词对每个语言的每个 AI 系统都具有相同的含义。无需翻译。

### 谓词约束

- **确定性**：必须恰好求值为 TRUE 或 FALSE
- **有限性**：量词仅遍历有限集合
- **无自引用**：不允许循环的谓词依赖
- **完备性**：`FALSIFIED_IF` 中的每个变量必须在 `WHERE` 中定义，反之亦然

---

## 三种声明类型

| 类型 | 名称 | 描述 | 是否需要谓词？ |
|------|------|------|----------------|
| **A** | 公理 | 基础性前提——设计上不可证伪 | 否（`null`） |
| **D** | 定义 | 约定性定义——不具有真值性 | 否（`null`） |
| **F** | 可证伪 | 带有确定性谓词的可测试声明 | 是——完整的 K = (P, O, M, B) |

---

## 六个验证层级

每个声明都携带一个 `tier` 字段，声明其所依据的证据类型。

### 构造锚定

| 层级 | 适用于 | 含义 |
|------|--------|------|
| `axiom` | A 类 | 基础性的，设计上不可证伪 |
| `definition` | D 类 | 定义性的，无需谓词 |

### 当前已锚定

| 层级 | 适用于 | 含义 |
|------|--------|------|
| `software_tested` | F 类 | 通过测试验证。`test_bindings` 非空，`verified_against` SHA 非空 |
| `formal_proof` | F 类 | 步骤索引的推导，`n_invalid_steps = 0` |
| `analytic` | F 类 | 通过形式推导轨迹或解析论证验证 |

### 显式未锚定

| 层级 | 适用于 | 含义 |
|------|--------|------|
| `empirical_pending` | F 类 | 活跃的占位符或等待外部数据。缺口是设计上可见的 |

---

## AI 层 — 核心工件

AI 层是一个伴随每篇论文或软件系统的 JSON 文档。它是两个协议共同操作的对象。

### 必需部分

| 部分 | 用途 |
|------|------|
| `_meta` | 文档类型、模式版本、会话、许可证 |
| `paper_id` | 唯一标识符 |
| `paper_title` | 人类可读的标题 |
| `paper_type` | `law_A`、`derivation_B`、`application_C` 或 `methodology_D` |
| `phase_ready` | 阶段门判定及条件状态（c1–c6） |
| `claim_registry` | 所有声明的数组，包含类型、谓词、层级、绑定 |
| `placeholder_register` | 未解决依赖的数组 |

### 模式

AI 层模式位于：[`ai-layers/ai-layer-schema.json`](https://github.com/thomasbrennan/Fracttalix/blob/main/ai-layers/ai-layer-schema.json)

---

## 六条件阶段门

当全部六个条件都满足时，论文或软件发布即为**阶段就绪（PHASE-READY）**：

| 条件 | 要求 |
|------|------|
| **c1** | AI 层通过模式验证 |
| **c2** | 所有可证伪声明均已注册并附带谓词 |
| **c3** | 所有谓词均可被机器评估 |
| **c4** | 交叉引用已追踪（占位符注册表） |
| **c5** | 验证自给自足（仅需 AI 层，无需散文文本） |
| **c6** | 所有谓词均非空泛（存在样本证伪观测值） |

---

## 如何实施

### 用于论文（DRP）

1. 撰写论文（面向人类读者的散文通道）
2. 创建 AI 层 JSON 文件（面向 AI 读者的机器通道）
3. 将每个声明分类为 A（公理）、D（定义）或 F（可证伪）
4. 为每个 F 类声明编写五部分证伪谓词
5. 为每个 F 类声明包含 `sample_falsification_observation`（空泛性见证）
6. 为每个声明指定验证层级
7. 对照 `ai-layer-schema.json` 进行验证
8. 运行阶段门检查（c1–c6）

### 用于软件（GVP）

1. 列举软件声明要做的事情
2. 将每个声明分类为 A（假设）、D（定义）或 F（行为性）
3. 为每个 F 类声明编写证伪谓词
4. 编写或确定验证每个声明的测试
5. 在 `test_bindings` 中填入完全限定的测试节点 ID
6. 运行测试并将通过的提交 SHA 记录到 `verified_against` 中
7. 指定层级：如果测试存在则为 `software_tested`，否则为 `empirical_pending`
8. 将所有未测试的声明注册为占位符
9. 对照 `ai-layer-schema.json` 进行验证

### 最小可行采纳

最小有用的 DRS 采纳方式是**一个 F 类声明加一个测试绑定**：

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

一个声明。一个测试。一个 SHA。DRS 已上线。在价值证明成本合理时再添加更多声明。

---

## 设计原则

**波普尔认识论。** 我们能证伪但不能证实。一个经受住所有证伪尝试的声明并非被证明了——它只是幸存了下来。

**诚实的缺口。** 占位符是最重要的特性。当一个声明处于 `empirical_pending` 状态时，系统在说："我们声称了这一点，但尚未验证。"这比另一种做法——未经验证的声明与已验证的声明无法区分——提供了严格更多的信息。

**基底无关性。** 内核 K = (P, O, M, B) 不知道它评估的是一个科学定理还是一个软件保证。未来的领域（法律、监管、政策）可以添加自己的协议而无需修改内核。

**机器通用语。** 内核以逻辑和数学写成，而非任何人类语言。一个在北京求值为 TRUE 的谓词在波士顿同样求值为 TRUE。JSON 是传输层。二进制逻辑是基底。

---

## 三轴兼容性

| 轴 | 承诺 | 机制 |
|----|------|------|
| **向后** | 已有的一切不会被破坏 | 模式版本控制、仅追加枚举、永久内核 |
| **横向** | 适用于所有领域、语言、工具、AI 系统 | 基底无关的内核、字符串类型绑定 |
| **向前** | 任何新内容都可以在不重新设计的情况下添加 | 协议可扩展性、层级可扩展性、增量式模式演进 |

---

## 资源

- [DRS 架构规范](https://github.com/thomasbrennan/Fracttalix/blob/main/paper/DRS-Architecture.md) — 完整规范
- [证伪内核 v1.1](https://github.com/thomasbrennan/Fracttalix/blob/main/ai-layers/falsification-kernel.md) — 第 0 层语义规范
- [AI 层模式 v3](https://github.com/thomasbrennan/Fracttalix/blob/main/ai-layers/ai-layer-schema.json) — JSON Schema
- [GVP 规范](https://github.com/thomasbrennan/Fracttalix/blob/main/docs/GVP-spec.md) — 软件协议
- [AI 层示例（P1）](https://github.com/thomasbrennan/Fracttalix/blob/main/ai-layers/P1-ai-layer.json) — 实际示例
- [Fracttalix 仓库](https://github.com/thomasbrennan/Fracttalix)

---

**许可证：** CC BY 4.0 | **DOI：** [10.5281/zenodo.18859299](https://doi.org/10.5281/zenodo.18859299) | **作者：** Thomas Brennan ([ORCID](https://orcid.org/0009-0002-6353-7115))
