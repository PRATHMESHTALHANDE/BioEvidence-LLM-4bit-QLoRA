# Task Specification & Taxonomy: BioEvidence-LLM

> **Status:** Specification Document  
> **Master Reference:** [AGENTS.md](file:///c:/Users/roari/Downloads/AI%20Projects/Fine%20Tunning%20LLM/AGENTS.md) | [Plan.md](file:///c:/Users/roari/Downloads/AI%20Projects/Fine%20Tunning%20LLM/Plan.md)  
> **Target Output Format:** Validated Structured JSON

---

## 1. System Objective & Safety Scope

BioEvidence-LLM is an NLP language model system designed to analyze biomedical scientific literature. It consumes **Evidence Context** (such as a PubMed abstract or clinical trial results section) and an optional **Biomedical Question**, producing a strictly grounded, structured JSON response.

### Non-Diagnostic Boundary
- BioEvidence-LLM is an NLP research and literature extraction tool.
- It is **not** a clinical diagnostic system or medical provider substitute.
- All claims must be cited from the provided context. If context is insufficient, the system must answer with uncertainty (`maybe`).

---

## 2. Task Taxonomy

The system supports seven specific biomedical instruction tasks:

| # | Task Name (`task`) | Primary Input | Core Objective | Expected Output Fields |
|---|---|---|---|---|
| 1 | `evidence_qa` | Question + Abstract | Synthesize answer directly grounded in study findings. | `decision`, `answer`, `evidence`, `uncertainty`, `limitations` |
| 2 | `evidence_classification` | Question + Abstract | Classify finding as `yes`, `no`, or `maybe`. | `decision`, `answer`, `evidence` |
| 3 | `pico_extraction` | Abstract | Extract Population, Intervention, Comparison, Outcome. | `pico` breakdown or structured extraction |
| 4 | `evidence_summary` | Abstract | Summarize key findings without extrapolating beyond text. | `answer` (summary), `evidence` |
| 5 | `uncertainty_extraction` | Abstract | Extract hedge words, sample size caveats, confidence intervals. | `uncertainty`, `evidence` |
| 6 | `limitation_extraction` | Abstract / Discussion | Extract author-acknowledged weaknesses and constraints. | `limitations`, `evidence` |
| 7 | `medical_explanation` | Technical Excerpt | Explain complex technical jargon in clear, plain language. | `answer`, `evidence` |

---

## 3. Canonical Output JSON Schema

Every inference response must strictly validate against this JSON structure:

```json
{
  "decision": "yes | no | maybe",
  "answer": "String synthesizing the finding.",
  "evidence": [
    "Verbatim cited sentence from the input text."
  ],
  "uncertainty": "String documenting caveats, mixed results, or null.",
  "limitations": [
    "String documenting methodological limitation 1",
    "String documenting methodological limitation 2"
  ]
}
```

### Field Definitions:
- **`decision`**: Mandatory for QA and classification tasks. Must be lowercase `yes`, `no`, or `maybe`.
- **`answer`**: Concise, grammatical synthesis. Must not contain ungrounded extrapolate claims.
- **`evidence`**: List of exact verbatim string excerpts from the provided `context`.
- **`uncertainty`**: Explicit notes on statistical power, sample diversity, or conflicting endpoints.
- **`limitations`**: Documented study constraints (e.g. single-center, retrospective, short follow-up).
