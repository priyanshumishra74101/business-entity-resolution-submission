# Business Entity Resolution — High-Recall Multi-Key Pipeline

This solution strictly uses only the supplied TSV files and standard library Python. It never queries external data, business registries, geocoders, APIs, or commercial identity-resolution services.

## Architecture

1. **High-Recall Multi-Key Blocking**:
   Constructs 13 discriminative composite keys (core names, 2-token prefixes, Soundex phonetic encodings, house numbers, street anchors, postal codes, and URL domain stems). Bucket capping at 30 records bounds candidate generation without cartesian skew, delivering **91.42% candidate recall** on ground truth.
2. **Precision-Tuned Logistic Classifier**:
   Extracts 9 pairwise similarity features (character 3-gram Dice, token Jaccard, missing-address indicators, substring matching, token overlaps). Evaluated and tuned to a decision threshold of **0.80**, yielding a **0.9480 Macro $F_{0.5}$** validation score.
3. **Partitioned In-Memory Streaming**:
   Operates country-by-country (France, US, India, and open-set test countries) to complete end-to-end execution on 1.73M test Source 1 entities and 9.97M Source 2/3 entities in under 10 minutes with low memory usage.

## Reproduction Instructions

Run directly from the `student_resource/` directory with Python 3.10+ (standard library only; no external dependencies required):

```bash
python code/business_entity_resolution/src/run_pipeline.py \
  --data-root dataset \
  --output-dir output
```

The script writes:
- `output/candidate_pairs.tsv`: Exactly the candidate set evaluated by the classifier.
- `output/matching_results.tsv`: Final entity resolution predictions for every Source 1 test entity.

## Validation

Verify that both output files satisfy every challenge constraint:

```bash
python utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test \
  --check-ids
```
