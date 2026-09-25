# ML Challenge 2026: Business Entity Resolution

**Team Name:** To be completed by submitter  
**Team Members:** To be completed by submitter  
**Submission Date:** 25 September 2026

## Executive Summary

This is a fully local, two-stage entity-resolution system. A capped blocking stage creates candidate pairs; an original logistic pair classifier then filters them with a probability threshold selected on a held-out macro F_0.5 validation split.

## Data Verification and Preprocessing

The supplied `test_source1.tsv` has 1,732,544 records; it is read as tab-separated data and every row appears once in both outputs. Text processing performs Unicode normalization, lower-casing, abbreviation expansion, legal-suffix removal, and tokenized order-independent address normalization. Missing addresses are kept as empty strings and missing countries become `unknown`. `indic-transliteration` only performs rule-based Devanagari-to-Latin conversion; it performs no entity lookup. Country remains an open normalized string, so France needs no hard-coded handling.

## Candidate Generation

The final candidate set is the de-duplicated union of capped (at most 25 records per block) same-country blocks: normalized name core, sorted normalized address, postcode/street anchor, and postcode plus name core. Oversized blocks are excluded to control skew. Indexed SQLite equality lookups avoid a Cartesian or nested all-pairs comparison.

The final test candidate file has 173,966 IDs across 35,860 Source-1 entities. It is exactly the candidate list passed to the classifier.

## Matching Model

Features include RapidFuzz name ratio and token-sort ratio, name/address token Jaccard, character-trigram TF-IDF cosine, country agreement, address-presence, postcode/house-number/street agreement, and name/token-length differences. An original logistic classifier trains on supplied ground-truth positive candidate pairs and blocked negatives. A deterministic held-out Source-1 split selected the probability threshold that maximized macro F_0.5.

## Validation and Results

- Hold-out macro F_0.5: 0.6717
- Candidate recall on sampled held-out links: 0.5575
- Selected probability threshold: 0.71
- Final candidate IDs: 173,966
- Final matches: 1,754 IDs across 1,691 Source-1 entities

`matching_results.tsv` is a strict subset of `candidate_pairs.tsv`; they are no longer copies. The supplied validator reports PASS: all 1,732,544 Source-1 rows occur exactly once, IDs are deduplicated, and output matches are candidates.

## Fair-play Compliance

The workflow uses only the supplied TSV files and their training ground truth. It does not access external business databases, APIs, registries, geocoders, web services, or data augmentation. The final model is an original local logistic classifier, so it has no pretrained-model licensing or parameter-count concern.
