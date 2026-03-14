# Dual Reader Standardの実装方法

**科学論文、ソフトウェア、知識システムのための機械検証可能な主張を構築する完全ガイド。**

> 関連資料：[ja.json](ja.json) — AIシステムおよびウェブクローラー向けの本ガイドのJSON-LD構造化データ版。

---

## Dual Reader Standardとは何か？

**Dual Reader Standard（DRS）** は、知識システムのための検証アーキテクチャです。すべての主張は——散文で書かれたものもコードで実装されたものも——**人間**と**機械**という二つの独立した読者クラスによって読解可能でなければなりません。

DRSは二つのプロトコルで構成されます：

| プロトコル | ドメイン | 機能 |
|----------|--------|------|
| **DRP**（Dual Reader Protocol） | テキスト／論文 | 5部構成の反証述語により、散文の主張を機械評価可能にする |
| **GVP**（Grounded Verification Protocol） | ソフトウェア／コード | テストバインディングとコミット固定のエビデンスにより、機械評価可能な主張を機械検証済みにする |

### 3つの読者

| 読者 | チャネル | 読むもの | 形式 |
|------|---------|---------|------|
| **人間** | 散文 | 論文 | 自然言語 |
| **AI** | JSON | AIレイヤー | 構造化された主張レジストリ |
| **CI／テストランナー** | 実行可能 | テストバインディング | テストノードID＋コミットSHA |

---

## 反証カーネル K = (P, O, M, B)

両プロトコルの共通基盤です。すべてのType F（反証可能）の主張は、**FALSIFIED**または**NOT FALSIFIED**のいずれか一方の判定に正確に評価される決定論的述語を持ちます。

| 記号 | 名称 | JSONフィールド | 役割 |
|------|------|--------------|------|
| **P** | 述語 | `FALSIFIED_IF` | TRUEの場合に主張を反証する論理文 |
| **O** | オペランド | `WHERE` | 述語内のすべての変数の型付き定義 |
| **M** | メカニズム | `EVALUATION` | 有限かつ決定論的な評価手続き |
| **B** | 境界 | `BOUNDARY` + `CONTEXT` | 閾値の意味論と正当化 |

### 述語の例

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

この述語はすべての言語のすべてのAIシステムにとって同一の意味を持ちます。翻訳は不要です。

### 述語の制約

- **決定性**：正確にTRUEまたはFALSEに評価されなければならない
- **有限性**：量化子は有限集合のみを対象とする
- **自己参照の禁止**：循環する述語依存関係は不可
- **完全性**：`FALSIFIED_IF`のすべての変数が`WHERE`で定義され、逆も同様

---

## 3つの主張タイプ

| タイプ | 名称 | 説明 | 述語が必要か？ |
|--------|------|------|--------------|
| **A** | 公理 | 基礎的前提——設計上反証不可能 | いいえ（`null`） |
| **D** | 定義 | 規定的定義——真偽を問わない | いいえ（`null`） |
| **F** | 反証可能 | 決定論的述語を持つテスト可能な主張 | はい — 完全なK = (P, O, M, B) |

---

## 6つの検証ティア

すべての主張は、どのような種類のエビデンスが根拠となるかを宣言する`tier`フィールドを持ちます。

### 構成による根拠付け

| ティア | 適用対象 | 意味 |
|--------|---------|------|
| `axiom` | Type A | 基礎的、設計上反証不可能 |
| `definition` | Type D | 定義的、述語不要 |

### 現在根拠あり

| ティア | 適用対象 | 意味 |
|--------|---------|------|
| `software_tested` | Type F | 合格テストにより検証済み。`test_bindings`が非空、`verified_against`のSHAが非null |
| `formal_proof` | Type F | ステップインデックス付き導出で`n_invalid_steps = 0` |
| `analytic` | Type F | 形式的導出トレースまたは解析的論証により検証済み |

### 明示的に未根拠

| ティア | 適用対象 | 意味 |
|--------|---------|------|
| `empirical_pending` | Type F | アクティブなプレースホルダーまたはデータ待ち。ギャップは設計上可視 |

---

## AIレイヤー — 中心的成果物

AIレイヤーは、すべての論文またはソフトウェアシステムに付随するJSONドキュメントです。両プロトコルが操作する対象です。

### 必須セクション

| セクション | 目的 |
|-----------|------|
| `_meta` | ドキュメントタイプ、スキーマバージョン、セッション、ライセンス |
| `paper_id` | 一意の識別子 |
| `paper_title` | 人間可読タイトル |
| `paper_type` | `law_A`、`derivation_B`、`application_C`、または`methodology_D` |
| `phase_ready` | フェーズゲート判定および条件ステータス（c1〜c6） |
| `claim_registry` | すべての主張のタイプ、述語、ティア、バインディングを含む配列 |
| `placeholder_register` | 未解決の依存関係の配列 |

### スキーマ

AIレイヤースキーマは以下で利用可能です：[`ai-layers/ai-layer-schema.json`](https://github.com/thomasbrennan/Fracttalix/blob/main/ai-layers/ai-layer-schema.json)

---

## 6条件フェーズゲート

論文またはソフトウェアリリースは、6つの条件がすべて満たされたとき**PHASE-READY**となります：

| 条件 | 要件 |
|------|------|
| **c1** | AIレイヤーがスキーマ有効である |
| **c2** | すべての反証可能な主張が述語とともに登録されている |
| **c3** | すべての述語が機械評価可能である |
| **c4** | 相互参照が追跡されている（プレースホルダーレジスター） |
| **c5** | 検証が自己完結的である（AIレイヤー単独で、散文は不要） |
| **c6** | すべての述語が非空虚である（反証観測サンプルが存在する） |

---

## 実装方法

### 論文の場合（DRP）

1. 論文を執筆する（人間読者のための散文チャネル）
2. AIレイヤーJSONファイルを作成する（AI読者のための機械チャネル）
3. すべての主張をA（公理）、D（定義）、F（反証可能）に分類する
4. すべてのType F主張に対して5部構成の反証述語を記述する
5. 各Type F主張に`sample_falsification_observation`を含める（空虚性の証人）
6. 各主張に検証ティアを割り当てる
7. `ai-layer-schema.json`に対して検証する
8. フェーズゲートを実行する（c1〜c6）

### ソフトウェアの場合（GVP）

1. ソフトウェアが何を主張するかを列挙する
2. 各主張をA（仮定）、D（定義）、F（動作的）に分類する
3. すべてのType F主張に対して反証述語を記述する
4. 各主張を検証するテストを記述または特定する
5. 完全修飾テストノードIDで`test_bindings`を設定する
6. テストを実行し、合格したコミットSHAを`verified_against`に記録する
7. ティアを割り当てる：テストが存在すれば`software_tested`、まだなければ`empirical_pending`
8. テストされていない主張をプレースホルダーとして登録する
9. `ai-layer-schema.json`に対して検証する

### 最小限の導入

最小限のDRS導入は、**1つのType F主張と1つのテストバインディング**です：

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

主張1つ。テスト1つ。SHA1つ。DRSは稼働中です。価値がコストを正当化するときに、主張を追加してください。

---

## 設計原則

**ポパー的認識論。** 我々は反証できるが検証はできない。すべての反証の試みを生き延びた主張は証明されたのではなく、生き延びたのである。

**誠実なギャップ。** プレースホルダーは最も重要な機能である。主張が`empirical_pending`のとき、システムは「我々はこれを主張するが、まだ検証していない」と宣言する。これは、未検証の主張が検証済みの主張と区別できない状態よりも、厳密に情報量が多い。

**基盤独立性。** カーネルK = (P, O, M, B)は、科学的定理を評価しているのかソフトウェアの保証を評価しているのかを知らない。将来のドメイン（法律、規制、政策）はカーネルを変更せずに独自のプロトコルを追加できる。

**機械の共通言語。** カーネルは論理学と数学で書かれており、いかなる人間の言語でもない。北京でTRUEと評価される述語はボストンでもTRUEと評価される。JSONはトランスポート層であり、二値論理が基盤である。

---

## 三軸互換性

| 軸 | 約束 | メカニズム |
|----|------|-----------|
| **過去** | 既存のものは壊れない | スキーマバージョニング、追加のみの列挙型、恒久的カーネル |
| **横断** | すべてのドメイン、言語、ツール、AIシステムで動作する | 基盤独立カーネル、文字列型バインディング |
| **未来** | 再設計なしに新しいものを追加できる | プロトコル拡張性、ティア拡張性、追加的スキーマ進化 |

---

## リソース

- [DRSアーキテクチャ仕様](https://github.com/thomasbrennan/Fracttalix/blob/main/paper/DRS-Architecture.md) — 完全な仕様
- [反証カーネル v1.1](https://github.com/thomasbrennan/Fracttalix/blob/main/ai-layers/falsification-kernel.md) — Layer 0意味論仕様
- [AIレイヤースキーマ v3](https://github.com/thomasbrennan/Fracttalix/blob/main/ai-layers/ai-layer-schema.json) — JSONスキーマ
- [GVP仕様](https://github.com/thomasbrennan/Fracttalix/blob/main/docs/GVP-spec.md) — ソフトウェアプロトコル
- [AIレイヤーの例（P1）](https://github.com/thomasbrennan/Fracttalix/blob/main/ai-layers/P1-ai-layer.json) — 実動例
- [Fracttalixリポジトリ](https://github.com/thomasbrennan/Fracttalix)

---

**ライセンス：** CC BY 4.0 | **DOI：** [10.5281/zenodo.18859299](https://doi.org/10.5281/zenodo.18859299) | **著者：** Thomas Brennan（[ORCID](https://orcid.org/0009-0002-6353-7115)）
