# Credit Scoring PD — Resumen Fase 1 (EDA) y contexto para Fase 2

> Documento de contexto para retomar el proyecto en un chat nuevo. Contiene todo lo
> trabajado y decidido en la Fase 1 (EDA), y una guía de cómo abordar la Fase 2
> (Feature Engineering / WOE-IV).

---

## Contexto del proyecto

Proyecto de portfolio de **Credit Scoring (PD — Probability of Default)** con el
dataset "Give Me Some Credit" (Kaggle), enfoque de banca de riesgos. Target:
`SeriousDlqin2yrs`. El documento de referencia completo del proyecto es
`PROYECTO_CREDIT_SCORING.md` (roadmap de las 6 fases + dashboard Streamlit).

**Nivel del usuario:** aprendiendo activamente pandas/Python y conceptos de riesgo
crediticio. Prefiere que se le explique la lógica y se le den pistas, no el código
resuelto directamente — construye el código él mismo, con revisión y corrección
de criterio en cada paso.

**Entorno:** notebook `01_eda.ipynb` en VS Code, dataset en
`../data/raw/training-data.csv` (150,000 filas originales, renombrado desde
`cs-training.csv`). Repo en GitHub (`Credit-Risk-Scoring`), rama de trabajo `EDA`.

---

## 1. Variable objetivo — `SeriousDlqin2yrs`
- Confirmado el desbalance de clases: ~6-7% de mora (positivos).
- Implicación: no usar accuracy como métrica principal en fases futuras; estratificar
  splits train/test.

---

## 2. Tabla resumen de calidad de datos

| Variable | Problema | Filas afectadas | Decisión | Justificación |
|---|---|---|---|---|
| `MonthlyIncome` | Missing | 29,731 (19.8%) | No imputar. Categoría propia en binning WOE | Grupo con missing tiene menor mora (5.6% vs 6.9%) y mayor edad (56 vs 51 años). Asociado a jubilados con ingreso no capturado como salario, no informalidad de riesgo |
| `NumberOfDependents` | Missing | 3,924 (2.6%) | No imputar. Categoría propia en binning WOE | Mismo patrón: menor mora (4.6% vs 6.7%) y mayor edad (59.6 vs 52.1). Hijos ya independientes |
| `age` | Valor imposible (age=0) | 1 | Eliminar la fila | Caso aislado, resto de variables de la fila normales. Error de captura, no sistemático |
| `RevolvingUtilizationOfUnsecuredLines` | Valores extremos (máx. 50,708) | 241 casos >10 | Capar en 1.0 + flag `_anomalo` | Media (6.05) muy por encima de mediana (0.15). Ratio de utilización no debería superar ~1 en la práctica |
| `DebtRatio` | Valores extremos (máx. 329,664) | Concentrado en filas con `MonthlyIncome` nulo (31,215 filas >2, más que las 29,731 con ingreso nulo) | Capar en 2.0 + flag `_anomalo` | Percentil 75 = 2,382 en filas con ingreso nulo vs 0.48 con ingreso reportado. Confirma que el missing de ingreso "rompe" el cálculo del ratio (división por cero/casi cero) |
| `NumberOfTime30-59...` / `60-89...` / `NumberOfTimes90DaysLate` | Códigos de sistema 96/98 mezclados con conteos reales | 264 (código 98) + 5 (código 96) | No eliminar. Categoría/bin propio (`_codigo_sistema`) | Tasa de mora del grupo 96/98 es 54% vs 6.6% del resto (~8x superior). Es señal de alto riesgo, no ruido — eliminarlo perdería la info más predictiva del dataset |

---

## 3. Análisis bivariado — conclusiones por variable

### `age`
- Relación **monotónica decreciente** en los 10 deciles, sin excepciones: 11.4% de
  mora en el decil más joven (21-33 años) → 2.2% en el decil de mayor edad (72-109).
- No se observa estabilización ni repunte en edades avanzadas (contrario a la
  hipótesis inicial planteada en el documento del proyecto).
- Variable con fuerte poder predictivo, ideal para binning WOE sin ajustes especiales.

### `DebtRatio`
- Tras capar en 2.0, el `qcut` en 10 bins muestra una relación mayormente creciente
  pero con un punto de masa importante: 69% del bin superior corresponde al propio
  valor del cap (31,215 de 44,999 filas), mezclando señal real de apalancamiento
  alto con artefacto matemático del missing de ingreso.
- Fase 2 deberá usar el flag `DebtRatio_anomalo` para separar ambas señales en el
  binning definitivo (`optbinning`), en vez de dejarlas mezcladas.

### `RevolvingUtilizationOfUnsecuredLines`
- Tras capar en 1.0 (100% utilización), 22% del bin superior es el punto de masa
  del cap (3,338 de 15,000 filas) — menos dominante que en `DebtRatio`.
- Patrón mayormente creciente (1.4% → 23.2% de mora), coherente con la lógica de
  riesgo (mayor uso relativo de crédito = mayor riesgo).
- **Excepción real detectada:** el bin de utilización casi nula (0-0.3%) tiene mora
  ligeramente mayor que el siguiente bin (2.5% vs 1.4%), rompiendo la monotonicidad
  estricta. Se investigó y se confirmó: ese grupo tiene edad promedio mayor e
  ingreso reportado menor que el resto — consistente con el mismo perfil de
  jubilados/ingresos no convencionales visto en `MonthlyIncome`.
- Fase 2 deberá decidir si fuerza monotonicidad total (fusionando ese primer bin
  con el siguiente) o permite esta excepción justificada por evidencia de negocio.

### `NumberOfOpenCreditLinesAndLoans`
- Confirma relación en **forma de U parcial**: mora muy alta con 0 líneas (25.6%),
  cae hasta un mínimo ~4.8% alrededor de 8 líneas (poco historial = riesgo, no
  seguridad). La zona media (9-25 líneas) se mantiene relativamente estable/plana,
  sin tendencia creciente clara todavía.
- A partir de 29 líneas (114 casos y menos), el volumen se vuelve poco confiable
  (oscilaciones erráticas 0%-50%).
- Se capa en 29 + flag `_anomalo`.

### `NumberOfTimes90DaysLate` (excluyendo códigos 96/98)
- La relación más fuerte de todo el EDA: mora sube de 4.6% (0 atrasos) a 33.7%
  (1 atraso), y sigue creciendo hasta estabilizarse ~60-80% a partir de 5+ atrasos
  (con volumen cada vez más bajo).
- Se capa en 5 (131 casos, último punto con volumen mínimamente estable).
- Tratamiento en 3 capas para no mezclar poblaciones distintas:
  - `NumberOfTimes90DaysLate_codigo_sistema`: flag para 96/98.
  - `NumberOfTimes90DaysLate_anomalo`: flag para atrasos reales ≥5 (excluyendo 96/98).
  - `NumberOfTimes90DaysLate_capped`: valor original si <5; 5.0 si atraso real ≥5;
    `NaN` si es código de sistema (para no mezclarlo numéricamente con los atrasos
    reales altos).
- Este mismo patrón de tratamiento (3 flags/columnas) es candidato a replicarse en
  `NumberOfTime30-59DaysPastDueNotWorse` y `NumberOfTime60-89DaysPastDueNotWorse`,
  que comparten los mismos códigos 96/98 pero **no se ha hecho aún su bivariado
  individual ni su capping** (ver pendientes).

---

## 4. Columnas nuevas creadas en el notebook hasta ahora

| Columna | Tipo | Descripción |
|---|---|---|
| `DebtRatio_anomalo` | flag (0/1) | 1 si `DebtRatio` original > 2 |
| `DebtRatio_capped` | numérica | `DebtRatio` capado en 2.0 |
| `RevolvingUtilizationOfUnsecuredLines_anomalo` | flag (0/1) | 1 si original > 1 |
| `RevolvingUtilizationOfUnsecuredLines_capped` | numérica | capado en 1.0 |
| `NumberOfOpenCreditLinesAndLoans_anomalo` | flag (0/1) | 1 si original ≥ 29 |
| `NumberOfOpenCreditLinesAndLoans_capped` | numérica | capado en 29 |
| `NumberOfTimes90DaysLate_codigo_sistema` | flag (0/1) | 1 si original es 96 o 98 |
| `NumberOfTimes90DaysLate_anomalo` | flag (0/1) | 1 si atraso real ≥ 5 (excluye 96/98) |
| `NumberOfTimes90DaysLate_capped` | numérica | capado en 5; `NaN` si es código de sistema |

**Nota técnica importante aprendida en sesión:** nunca sobreescribir la columna
original al hacer binning/transformaciones (`df['col'] = pd.qcut(...)` rompe el
dtype a `category` y pierde los datos numéricos originales) — siempre crear
columna nueva. Ya ocurrió una vez con `RevolvingUtilizationOfUnsecuredLines` y se
corrigió recargando esa columna desde el CSV original alineando por índice.

Además, la fila con `age=0` (índice 65695) **ya fue eliminada** del dataframe
principal (`df = df[df['age'] != 0]`), por lo que el dataframe de trabajo tiene
149,999 filas desde ese punto en adelante.

---

## 5. Pendientes antes de considerar la Fase 1 100% cerrada (opcional)

- No se ha hecho el mismo tratamiento (capping + flags) para
  `NumberOfTime30-59DaysPastDueNotWorse` ni `NumberOfTime60-89DaysPastDueNotWorse`
  individualmente — solo se detectaron sus códigos 96/98 al principio. Se puede
  replicar el mismo patrón de 3 columnas usado en `NumberOfTimes90DaysLate`.
- No se ha explorado bivariado de `NumberRealEstateLoansOrLines` ni de la columna
  índice `Unnamed: 0` (pendiente decidir si se descarta, ya se intuía que no aporta).
- No se ha guardado/exportado aún ninguna versión "limpia" del dataset a
  `data/processed/` (sugerido por la estructura de repo del documento del proyecto).

Estos son opcionales — se puede pasar a Fase 2 con lo que ya hay, y resolver estos
puntos sueltos en paralelo si aparece la necesidad.

---

## 6. Cómo proceder en la Fase 2 (Feature Engineering — WOE/IV)

Según el documento de referencia del proyecto (`PROYECTO_CREDIT_SCORING.md`,
sección Fase 2), los pasos son:

1. **Explicar conceptos antes de programar** (mismo estilo de trabajo que en Fase 1):
   qué es WOE (Weight of Evidence) y qué es IV (Information Value), por qué se usan
   en banca en vez de solo correlación, y las reglas de bolsillo de IV
   (< 0.02 descartar, 0.02-0.1 débil, 0.1-0.3 media, 0.3-0.5 fuerte, > 0.5 sospechosa).

2. **Binning supervisado** con la librería `optbinning`, forzando monotonicidad de
   la tasa de mora por bin (salvo excepciones ya justificadas como la de
   `RevolvingUtilizationOfUnsecuredLines`). Es un requisito típico de validación de
   modelos en banca real.

3. **Reutilizar directamente el trabajo ya hecho en la Fase 1:**
   - Los missing de `MonthlyIncome` y `NumberOfDependents` deben entrar al binning
     como su propia categoría, no imputarse.
   - Los flags `_anomalo` y `_codigo_sistema` ya creados deben aprovecharse para
     que el binning no mezcle poblaciones distintas.
   - Las variables `_capped` ya creadas son las que deberían alimentar el binning
     numérico (no las originales sin tratar).

4. **Entregable de la fase:** `src/woe_binning.py` con funciones reutilizables +
   tabla de IV por variable (ranking), que servirá de base para la selección de
   variables del modelo baseline (Fase 3, regresión logística sobre WOE).

**Primer paso recomendado al abrir el chat nuevo:** pedir que explique WOE e IV
con el mismo enfoque didáctico (conceptos antes de código), y luego aplicar el
binning empezando por `age` (la variable más "limpia" y con mejor relación
monotónica de toda la Fase 1), para practicar el flujo completo con `optbinning`
antes de extenderlo al resto de variables.
