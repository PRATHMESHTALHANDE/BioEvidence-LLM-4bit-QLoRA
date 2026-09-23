# Human Review Rubric & Quality Control Standard: BioEvidence-LLM

> **Status:** Standard Operating Procedure (SOP)  
> **Master Reference:** [AGENTS.md](file:///c:/Users/roari/Downloads/AI%20Projects/Fine%20Tunning%20LLM/AGENTS.md) | [Plan.md](file:///c:/Users/roari/Downloads/AI%20Projects/Fine%20Tunning%20LLM/Plan.md)  
> **Applicability:** Pilot Dataset (`BioEvidence-SFT-v0.1`), Scaled SFT, and Golden Benchmark (`BioEvidence-Eval-v0.1`)

---

## 1. Quality Control Workflow

No synthetic or transformed dataset record may be used in training or benchmarking without undergoing automated heuristic checks followed by human-in-the-loop sample review.

```text
Ingested Candidate ──► Automated Schema Validation ──► Heuristic Checks ──► Human Review Rubric ──► SFT / Eval Dataset
```

---

## 2. Review Criteria & Scoring Matrix

Each audited sample is evaluated against 6 binary criteria. A single critical failure results in an overall verdict of `FAIL`.

| Criterion | Description | Pass Requirement | Critical Failure Condition |
|---|---|---|---|
| **C1: Factual Grounding** | All claims in `answer` must be traceable to the source context. | $100\%$ claims backed by context. | Generates claim absent from source text. |
| **C2: Verbatim Evidence** | Items in `evidence` array must be substrings of the context. | Exact substring match. | Fabricated quotation or hallucinated phrase. |
| **C3: Zero Fact Fabrication** | No invented numerical figures, sample counts ($n$), or $p$-values. | Statistics preserved exactly. | Any fabricated statistic (e.g. invented $p < 0.05$). |
| **C4: Uncertainty Preservation** | Nuance, hedge phrases, and non-definitive findings must not be forced. | Preserved when evidence is inconclusive. | Forcing `YES`/`NO` when study was inconclusive or non-significant. |
| **C5: Limitations Accuracy** | Extracted limitations must reflect true study methodology. | Real study constraints captured. | Fabricated methodology weaknesses. |
| **C6: JSON Schema Conformance** | Output strictly parses according to `StructuredModelOutput`. | Valid JSON with all required keys. | Syntax error, truncated keys, markdown wrappers. |

---

## 3. Rating Scale & Decision Protocol

* **`PASS`**: All 6 criteria (C1 through C6) pass without defect. Approved for training/eval pool.
* **`NEEDS_REVIEW`**: Ambiguous biomedical interpretation or minor stylistic issue; requires secondary domain expert review.
* **`FAIL`**: Any violation of C1, C2, C3, or C6. Sample is permanently discarded or flagged for schema repair.
