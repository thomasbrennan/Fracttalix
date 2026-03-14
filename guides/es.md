# Cómo Implementar el Dual Reader Standard

**Guía completa para construir afirmaciones verificables por máquina para artículos científicos, software y sistemas de conocimiento.**

> Véase también: [es.json](es.json) — Versión en datos estructurados JSON-LD de esta guía para sistemas de IA y rastreadores web.

---

## ¿Qué es el Dual Reader Standard?

El **Dual Reader Standard (DRS)** es una arquitectura de verificación para sistemas de conocimiento. Cada afirmación — ya sea escrita en prosa o implementada en código — debe ser legible por dos clases independientes de lectores: **humano** y **máquina**.

El DRS comprende dos protocolos:

| Protocolo | Dominio | Función |
|-----------|---------|---------|
| **DRP** (Dual Reader Protocol) | Texto / Artículos | Hace que las afirmaciones en prosa sean evaluables por máquina mediante predicados de falsación de 5 partes |
| **GVP** (Grounded Verification Protocol) | Software / Código | Hace que las afirmaciones evaluables por máquina sean verificadas por máquina mediante vinculaciones de pruebas y evidencia anclada a commits |

### Los Tres Lectores

| Lector | Canal | Lee | Formato |
|--------|-------|-----|---------|
| **Humano** | Prosa | El artículo | Lenguaje natural |
| **IA** | JSON | La capa de IA | Registro estructurado de afirmaciones |
| **CI / ejecutor de pruebas** | Ejecutable | Vinculaciones de pruebas | IDs de nodo de prueba + SHA de commit |

---

## El Núcleo de Falsación K = (P, O, M, B)

La base compartida de ambos protocolos. Cada afirmación de Tipo F (falsable) lleva un predicado determinista que se evalúa a exactamente uno de dos veredictos: **FALSIFIED** o **NOT FALSIFIED**.

| Símbolo | Nombre | Campo JSON | Rol |
|---------|--------|------------|-----|
| **P** | Predicado | `FALSIFIED_IF` | Sentencia lógica que, si es VERDADERA, falsa la afirmación |
| **O** | Operandos | `WHERE` | Definiciones tipadas de cada variable en el predicado |
| **M** | Mecanismo | `EVALUATION` | Procedimiento de evaluación finito y determinista |
| **B** | Límites | `BOUNDARY` + `CONTEXT` | Semántica de umbrales y justificación |

### Ejemplo de Predicado

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

Este predicado significa lo mismo para cualquier sistema de IA en cualquier idioma. No requiere traducción.

### Restricciones del Predicado

- **Determinismo**: Debe evaluarse a exactamente VERDADERO o FALSO
- **Finitud**: Los cuantificadores abarcan solo conjuntos finitos
- **Sin autorreferencia**: Sin dependencias circulares entre predicados
- **Completitud**: Cada variable en `FALSIFIED_IF` definida en `WHERE`, y viceversa

---

## Los Tres Tipos de Afirmación

| Tipo | Nombre | Descripción | ¿Requiere Predicado? |
|------|--------|-------------|----------------------|
| **A** | Axioma | Premisas fundacionales — no falsables por diseño | No (`null`) |
| **D** | Definición | Definiciones estipulativas — no son aptas para la verdad | No (`null`) |
| **F** | Falsable | Afirmaciones comprobables con predicados deterministas | Sí — K = (P, O, M, B) completo |

---

## Los Seis Niveles de Verificación

Cada afirmación lleva un campo `tier` que declara qué tipo de evidencia la sustenta.

### Fundamentados por Construcción

| Nivel | Se Aplica A | Significado |
|-------|-------------|-------------|
| `axiom` | Tipo A | Fundacional, no falsable por diseño |
| `definition` | Tipo D | Definicional, no requiere predicado |

### Fundamentados Ahora

| Nivel | Se Aplica A | Significado |
|-------|-------------|-------------|
| `software_tested` | Tipo F | Ejercitado mediante pruebas aprobadas. `test_bindings` no vacío, `verified_against` SHA no nulo |
| `formal_proof` | Tipo F | Derivación indexada por pasos con `n_invalid_steps = 0` |
| `analytic` | Tipo F | Verificado por traza de derivación formal o argumento analítico |

### Explícitamente No Fundamentados

| Nivel | Se Aplica A | Significado |
|-------|-------------|-------------|
| `empirical_pending` | Tipo F | Marcador activo o datos pendientes. La brecha es visible por diseño |

---

## La Capa de IA — El Artefacto Central

La capa de IA es un documento JSON que acompaña a cada artículo o sistema de software. Es sobre lo que operan ambos protocolos.

### Secciones Requeridas

| Sección | Propósito |
|---------|-----------|
| `_meta` | Tipo de documento, versión del esquema, sesión, licencia |
| `paper_id` | Identificador único |
| `paper_title` | Título legible por humanos |
| `paper_type` | `law_A`, `derivation_B`, `application_C` o `methodology_D` |
| `phase_ready` | Veredicto de la puerta de fase y estado de condiciones (c1–c6) |
| `claim_registry` | Arreglo de todas las afirmaciones con tipos, predicados, niveles, vinculaciones |
| `placeholder_register` | Arreglo de dependencias no resueltas |

### Esquema

El esquema de la capa de IA está disponible en: [`ai-layers/ai-layer-schema.json`](https://github.com/thomasbrennan/Fracttalix/blob/main/ai-layers/ai-layer-schema.json)

---

## La Puerta de Fase de Seis Condiciones

Un artículo o lanzamiento de software está **LISTO PARA FASE** cuando se cumplen las seis condiciones:

| Condición | Requisito |
|-----------|-----------|
| **c1** | La capa de IA es válida según el esquema |
| **c2** | Todas las afirmaciones falsables registradas con predicados |
| **c3** | Todos los predicados son evaluables por máquina |
| **c4** | Referencias cruzadas rastreadas (registro de marcadores) |
| **c5** | La verificación es autosuficiente (solo la capa de IA, sin necesidad de prosa) |
| **c6** | Todos los predicados son no vacuos (existe una observación de falsación de muestra) |

---

## Cómo Implementar

### Para un Artículo (DRP)

1. Escribir el artículo (canal de prosa para lectores humanos)
2. Crear el archivo JSON de la capa de IA (canal de máquina para lectores IA)
3. Clasificar cada afirmación como A (axioma), D (definición) o F (falsable)
4. Escribir el predicado de falsación de 5 partes para cada afirmación de Tipo F
5. Incluir una `sample_falsification_observation` para cada afirmación de Tipo F (testigo de vacuidad)
6. Asignar el nivel de verificación a cada afirmación
7. Validar contra `ai-layer-schema.json`
8. Ejecutar la puerta de fase (c1–c6)

### Para Software (GVP)

1. Enumerar lo que el software afirma hacer
2. Clasificar cada afirmación como A (supuesto), D (definición) o F (comportamental)
3. Escribir el predicado de falsación para cada afirmación de Tipo F
4. Escribir o identificar las pruebas que ejercitan cada afirmación
5. Rellenar `test_bindings` con los IDs de nodo de prueba completamente cualificados
6. Ejecutar las pruebas y registrar el SHA del commit aprobado en `verified_against`
7. Asignar el nivel: `software_tested` si existen pruebas, `empirical_pending` si aún no
8. Registrar cualquier afirmación no probada como marcador
9. Validar contra `ai-layer-schema.json`

### Adopción Mínima Viable

La adopción más pequeña y útil del DRS es **una afirmación de Tipo F con una vinculación de prueba**:

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

Una afirmación. Una prueba. Un SHA. El DRS está activo. Añada más afirmaciones cuando el valor justifique el costo.

---

## Principios de Diseño

**Epistemología popperiana.** Podemos falsar pero no verificar. Una afirmación que sobrevive a todos los intentos de falsación no está probada — ha sobrevivido.

**Brechas honestas.** El marcador de posición es la característica más importante. Cuando una afirmación es `empirical_pending`, el sistema dice: "afirmamos esto pero aún no lo hemos verificado." Esto es estrictamente más informativo que la alternativa, donde las afirmaciones no verificadas son indistinguibles de las verificadas.

**Independencia de sustrato.** El núcleo K = (P, O, M, B) no sabe si está evaluando un teorema científico o una garantía de software. Dominios futuros (legal, regulatorio, político) pueden añadir sus propios protocolos sin modificar el núcleo.

**Lengua franca de máquina.** El núcleo está escrito en lógica y matemáticas, no en ningún idioma humano. Un predicado que se evalúa como VERDADERO en Pekín se evalúa como VERDADERO en Boston. JSON es la capa de transporte. La lógica binaria es el sustrato.

---

## Compatibilidad en Tres Ejes

| Eje | Promesa | Mecanismo |
|-----|---------|-----------|
| **Pasado** | Nada de lo ya hecho se rompe | Versionado de esquemas, enumeraciones de solo adición, núcleo permanente |
| **Lateral** | Funciona en todos los dominios, idiomas, herramientas, sistemas de IA | Núcleo independiente del sustrato, vinculaciones tipadas como cadenas |
| **Futuro** | Cualquier novedad puede añadirse sin rediseño | Extensibilidad de protocolos, extensibilidad de niveles, evolución aditiva de esquemas |

---

## Recursos

- [Especificación de la Arquitectura DRS](https://github.com/thomasbrennan/Fracttalix/blob/main/paper/DRS-Architecture.md) — Especificación completa
- [Núcleo de Falsación v1.1](https://github.com/thomasbrennan/Fracttalix/blob/main/ai-layers/falsification-kernel.md) — Especificación semántica de Capa 0
- [Esquema de Capa de IA v3](https://github.com/thomasbrennan/Fracttalix/blob/main/ai-layers/ai-layer-schema.json) — JSON Schema
- [Especificación GVP](https://github.com/thomasbrennan/Fracttalix/blob/main/docs/GVP-spec.md) — Protocolo de software
- [Ejemplo de Capa de IA (P1)](https://github.com/thomasbrennan/Fracttalix/blob/main/ai-layers/P1-ai-layer.json) — Ejemplo funcional
- [Repositorio Fracttalix](https://github.com/thomasbrennan/Fracttalix)

---

**Licencia:** CC BY 4.0 | **DOI:** [10.5281/zenodo.18859299](https://doi.org/10.5281/zenodo.18859299) | **Autor:** Thomas Brennan ([ORCID](https://orcid.org/0009-0002-6353-7115))
