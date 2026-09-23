"""Preset NCBI Entrez queries for controlled biomedical evidence retrieval."""

NCBI_SEARCH_QUERIES = {
    "oncology_trials": '("clinical trial"[pt] OR "randomized controlled trial"[pt]) AND neoplasm AND survival[tiab]',
    "cardiovascular_trials": '("clinical trial"[pt] OR "randomized controlled trial"[pt]) AND ("myocardial infarction" OR "heart failure") AND mortality[tiab]',
    "pharmacology_rct": '("randomized controlled trial"[pt]) AND (efficacy OR safety) AND drug therapy[sh]',
    "biomarkers": 'biomarker AND ("predictive value" OR sensitivity OR specificity) AND "clinical trial"[pt]',
}
