# Business Entity Resolution — Two-stage pipeline

This solution uses only the supplied TSV files. It never queries external data, business registries, geocoders, APIs, or identity-resolution services.

## Reproduce

Run from `student_resource/` with Python 3.10 or newer:

```powershell
py -3.10 -m pip install -r code/business_entity_resolution/requirements.txt
py -3.10 code/business_entity_resolution/src/run_two_stage_pipeline.py `
  --data-root dataset `
  --output-dir output `
  --work-dir work
```

The first run constructs local SQLite indexes under `work/`; these are derived artefacts and are not required in the final package. On a multi-million-row dataset, allow substantial local disk space and time for index construction. The command writes:

- `output/candidate_pairs.tsv`: exactly the final candidate list evaluated by the scoring step.
- `output/matching_results.tsv`: predictions for every test Source-1 ID.

Validate after the run:

```powershell
py -3.10 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

## Pipeline

Text is normalized locally using Unicode folding, rule-based Devanagari-to-Latin transliteration, punctuation/abbreviation normalization, legal-suffix handling, and missing-value flags. Blocking takes a capped union of same-country normalized-name-core, order-independent-address, postcode/street-anchor, and postcode+name-core blocks. The final candidate file is precisely the set fed to the pair scorer. An original logistic classifier uses RapidFuzz name/address similarities, token Jaccard, trigram TF-IDF cosine, structured address agreements, and length features. A deterministic hold-out tunes the conservative macro F_0.5 threshold, so matching_results.tsv is a subset of candidate_pairs.tsv.
