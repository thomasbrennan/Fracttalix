# 이중 판독자 표준 구현 방법

**과학 논문, 소프트웨어, 지식 체계를 위한 기계 검증 가능 주장 구축에 관한 완전한 가이드.**

> 참고: [ko.json](ko.json) — 이 가이드의 JSON-LD 구조화 데이터 버전(AI 시스템 및 웹 크롤러용).

---

## 이중 판독자 표준이란 무엇인가?

**이중 판독자 표준(DRS)**은 지식 체계를 위한 검증 아키텍처입니다. 산문으로 작성되었든 코드로 구현되었든, 모든 주장은 **인간**과 **기계**라는 두 개의 독립적인 판독자 계층에 의해 판독 가능해야 합니다.

DRS는 두 가지 프로토콜로 구성됩니다:

| 프로토콜 | 도메인 | 기능 |
|----------|--------|------|
| **DRP** (Dual Reader Protocol) | 텍스트 / 논문 | 산문 주장을 5부분 반증 술어를 통해 기계 평가 가능하게 만듦 |
| **GVP** (Grounded Verification Protocol) | 소프트웨어 / 코드 | 기계 평가 가능한 주장을 테스트 바인딩과 커밋 고정 증거를 통해 기계 검증하게 만듦 |

### 세 가지 판독자

| 판독자 | 채널 | 판독 대상 | 형식 |
|--------|------|-----------|------|
| **인간** | 산문 | 논문 | 자연어 |
| **AI** | JSON | AI 레이어 | 구조화된 주장 레지스트리 |
| **CI / 테스트 러너** | 실행 파일 | 테스트 바인딩 | 테스트 노드 ID + 커밋 SHA |

---

## 반증 커널 K = (P, O, M, B)

두 프로토콜의 공유 기반입니다. 모든 유형 F(반증 가능) 주장은 정확히 두 가지 판정 중 하나로 평가되는 결정론적 술어를 수반합니다: **FALSIFIED** 또는 **NOT FALSIFIED**.

| 기호 | 이름 | JSON 필드 | 역할 |
|------|------|-----------|------|
| **P** | 술어(Predicate) | `FALSIFIED_IF` | 참(TRUE)일 경우 해당 주장을 반증하는 논리 문장 |
| **O** | 피연산자(Operands) | `WHERE` | 술어 내 모든 변수의 유형이 지정된 정의 |
| **M** | 메커니즘(Mechanism) | `EVALUATION` | 유한하고 결정론적인 평가 절차 |
| **B** | 경계(Bounds) | `BOUNDARY` + `CONTEXT` | 임계값 의미론 및 정당화 |

### 술어 예시

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

이 술어는 모든 언어의 모든 AI 시스템에 동일한 의미를 가집니다. 번역이 필요하지 않습니다.

### 술어 제약 조건

- **결정론**: 정확히 TRUE 또는 FALSE로 평가되어야 함
- **유한성**: 양화사는 유한 집합에 대해서만 범위를 가짐
- **자기 참조 없음**: 순환적 술어 의존성 없음
- **완전성**: `FALSIFIED_IF` 내 모든 변수가 `WHERE`에 정의되어야 하며, 그 역도 성립

---

## 세 가지 주장 유형

| 유형 | 이름 | 설명 | 술어 필요 여부 |
|------|------|------|----------------|
| **A** | 공리(Axiom) | 기초 전제 — 설계상 반증 불가능 | 아니오 (`null`) |
| **D** | 정의(Definition) | 규약적 정의 — 진리 적합성 없음 | 아니오 (`null`) |
| **F** | 반증 가능(Falsifiable) | 결정론적 술어를 가진 검증 가능한 주장 | 예 — 완전한 K = (P, O, M, B) |

---

## 여섯 가지 검증 계층

모든 주장은 어떤 종류의 증거가 그것을 뒷받침하는지 선언하는 `tier` 필드를 수반합니다.

### 구성에 의한 근거

| 계층 | 적용 대상 | 의미 |
|------|-----------|------|
| `axiom` | 유형 A | 기초적, 설계상 반증 불가능 |
| `definition` | 유형 D | 정의적, 술어 불필요 |

### 현재 근거 있음

| 계층 | 적용 대상 | 의미 |
|------|-----------|------|
| `software_tested` | 유형 F | 통과하는 테스트에 의해 검증됨. `test_bindings` 비어있지 않음, `verified_against` SHA 비어있지 않음 |
| `formal_proof` | 유형 F | `n_invalid_steps = 0`인 단계 인덱스 유도 |
| `analytic` | 유형 F | 형식적 유도 추적 또는 해석적 논증에 의해 검증됨 |

### 명시적으로 근거 없음

| 계층 | 적용 대상 | 의미 |
|------|-----------|------|
| `empirical_pending` | 유형 F | 활성 플레이스홀더 또는 대기 중인 데이터. 격차가 설계상 가시적 |

---

## AI 레이어 — 핵심 산출물

AI 레이어는 모든 논문이나 소프트웨어 시스템에 수반되는 JSON 문서입니다. 두 프로토콜 모두 이것을 대상으로 작동합니다.

### 필수 섹션

| 섹션 | 목적 |
|------|------|
| `_meta` | 문서 유형, 스키마 버전, 세션, 라이선스 |
| `paper_id` | 고유 식별자 |
| `paper_title` | 사람이 읽을 수 있는 제목 |
| `paper_type` | `law_A`, `derivation_B`, `application_C`, 또는 `methodology_D` |
| `phase_ready` | 페이즈 게이트 판정 및 조건 상태 (c1–c6) |
| `claim_registry` | 유형, 술어, 계층, 바인딩이 포함된 모든 주장의 배열 |
| `placeholder_register` | 미해결 의존성 배열 |

### 스키마

AI 레이어 스키마는 다음에서 확인할 수 있습니다: [`ai-layers/ai-layer-schema.json`](https://github.com/thomasbrennan/Fracttalix/blob/main/ai-layers/ai-layer-schema.json)

---

## 6개 조건 페이즈 게이트

논문이나 소프트웨어 릴리스는 6개 조건이 모두 충족될 때 **PHASE-READY**입니다:

| 조건 | 요구사항 |
|------|----------|
| **c1** | AI 레이어가 스키마에 유효함 |
| **c2** | 모든 반증 가능 주장이 술어와 함께 등록됨 |
| **c3** | 모든 술어가 기계 평가 가능함 |
| **c4** | 교차 참조가 추적됨 (플레이스홀더 레지스터) |
| **c5** | 검증이 자족적임 (AI 레이어 단독, 산문 불필요) |
| **c6** | 모든 술어가 비공허함 (표본 반증 관측이 존재) |

---

## 구현 방법

### 논문용 (DRP)

1. 논문 작성 (인간 판독자를 위한 산문 채널)
2. AI 레이어 JSON 파일 생성 (AI 판독자를 위한 기계 채널)
3. 모든 주장을 A(공리), D(정의), F(반증 가능)로 분류
4. 모든 유형 F 주장에 대해 5부분 반증 술어 작성
5. 각 유형 F 주장에 대해 `sample_falsification_observation` 포함 (공허성 증인)
6. 각 주장에 검증 계층 할당
7. `ai-layer-schema.json`에 대해 검증
8. 페이즈 게이트 실행 (c1–c6)

### 소프트웨어용 (GVP)

1. 소프트웨어가 수행한다고 주장하는 것을 열거
2. 각 주장을 A(가정), D(정의), F(동작적)로 분류
3. 모든 유형 F 주장에 대해 반증 술어 작성
4. 각 주장을 검증하는 테스트 작성 또는 식별
5. 정규화된 테스트 노드 ID로 `test_bindings` 채우기
6. 테스트를 실행하고 통과한 커밋 SHA를 `verified_against`에 기록
7. 계층 할당: 테스트가 있으면 `software_tested`, 아직 없으면 `empirical_pending`
8. 테스트되지 않은 주장을 플레이스홀더로 등록
9. `ai-layer-schema.json`에 대해 검증

### 최소 실행 가능 도입

가장 작은 유용한 DRS 도입은 **하나의 유형 F 주장과 하나의 테스트 바인딩**입니다:

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

주장 하나. 테스트 하나. SHA 하나. DRS가 작동합니다. 가치가 비용을 정당화할 때 주장을 추가하십시오.

---

## 설계 원칙

**포퍼적 인식론.** 우리는 반증할 수 있지만 검증할 수는 없습니다. 모든 반증 시도에서 살아남은 주장은 증명된 것이 아니라 — 살아남은 것입니다.

**정직한 격차.** 플레이스홀더는 가장 중요한 기능입니다. 주장이 `empirical_pending`일 때, 시스템은 "우리는 이것을 주장하지만 아직 검증하지 않았다"고 말합니다. 이는 미검증 주장이 검증된 주장과 구별 불가능한 대안보다 엄밀히 더 유익합니다.

**기질 독립성.** 커널 K = (P, O, M, B)는 과학 정리를 평가하는지 소프트웨어 보증을 평가하는지 알지 못합니다. 미래 도메인(법률, 규제, 정책)은 커널을 수정하지 않고 자체 프로토콜을 추가할 수 있습니다.

**기계 공용어.** 커널은 어떤 인간 언어가 아닌 논리와 수학으로 작성됩니다. 베이징에서 TRUE로 평가되는 술어는 보스턴에서도 TRUE로 평가됩니다. JSON은 전송 계층입니다. 이진 논리가 기질입니다.

---

## 3축 호환성

| 축 | 약속 | 메커니즘 |
|----|------|----------|
| **과거** | 이미 수행된 것은 깨지지 않음 | 스키마 버전 관리, 추가 전용 열거형, 영구 커널 |
| **횡적** | 모든 도메인, 언어, 도구, AI 시스템에서 작동 | 기질 독립적 커널, 문자열 유형 바인딩 |
| **전방** | 재설계 없이 새로운 것을 추가할 수 있음 | 프로토콜 확장성, 계층 확장성, 추가적 스키마 진화 |

---

## 참고 자료

- [DRS 아키텍처 명세](https://github.com/thomasbrennan/Fracttalix/blob/main/paper/DRS-Architecture.md) — 전체 명세
- [반증 커널 v1.1](https://github.com/thomasbrennan/Fracttalix/blob/main/ai-layers/falsification-kernel.md) — 레이어 0 의미론 명세
- [AI 레이어 스키마 v3](https://github.com/thomasbrennan/Fracttalix/blob/main/ai-layers/ai-layer-schema.json) — JSON 스키마
- [GVP 명세](https://github.com/thomasbrennan/Fracttalix/blob/main/docs/GVP-spec.md) — 소프트웨어 프로토콜
- [AI 레이어 예시 (P1)](https://github.com/thomasbrennan/Fracttalix/blob/main/ai-layers/P1-ai-layer.json) — 작동하는 예시
- [Fracttalix 저장소](https://github.com/thomasbrennan/Fracttalix)

---

**라이선스:** CC BY 4.0 | **DOI:** [10.5281/zenodo.18859299](https://doi.org/10.5281/zenodo.18859299) | **저자:** Thomas Brennan ([ORCID](https://orcid.org/0009-0002-6353-7115))
