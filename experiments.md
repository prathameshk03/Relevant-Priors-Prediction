# Experiments

## Final Approach

The final submission uses a deterministic rule-based scorer. It compares each
current study with each prior study using body-part match, modality match,
recency, and keyword overlap. Embeddings were removed from the deployed API
because model loading and inference latency caused timeout risk on the free-tier
endpoint.

The final scoring weights are:

| Signal | Weight |
| --- | ---: |
| Body-part match | 0.45 |
| Modality match | 0.20 |
| Recency | 0.20 |
| Keyword overlap | 0.15 |

## Results

| Experiment | Result | Latency | Notes |
| --- | ---: | ---: | --- |
| Embedding API attempt | Timed out / 524 | N/A | Sentence-transformer loading and request-time embedding work were too slow and memory-heavy for Render free tier. |
| Rule-only local public evaluation | 24,156 / 27,614 correct, 87.48% accuracy | Local batch run | `python main.py` on `relevant_priors_public.json`; 666 false positives and 2,792 false negatives. |
| Rule-only quick API check | 164 / 173 correct, 94.80% accuracy | 586 ms | Fixed public smoke test of 10 cases with full prior-study coverage. |
| Final evaluation | 77.00 / 100 | Not reported | Strong endpoint accuracy and coverage, but hidden/full evaluation exposed remaining rule-based edge cases. |

## Error Analysis

The quick API check showed that the rule-based endpoint can return complete
predictions quickly and accurately on a small public smoke set. The lower final
score suggests that broader evaluation cases include description patterns that
the handcrafted body-part and keyword mappings do not fully cover. Likely misses
include uncommon abbreviations, studies with vague descriptions, and priors that
are clinically relevant despite only partial text overlap.

## Radiologist Workflow

For a radiologist, missing a relevant prior can be worse than surfacing a few
extra irrelevant priors. A missed prior can remove important context about disease
progression, stability, or prior interventions. Extra priors add review burden,
but they are often easier to dismiss if the user interface keeps them organized.

Because of that tradeoff, thresholding should generally favor recall when the UI
can tolerate some false positives. The current threshold keeps the endpoint fast
and stable while trying to avoid obvious irrelevant priors, but future work should
measure false negatives carefully and tune the threshold around clinical review
cost rather than accuracy alone.

## Next Steps

Useful follow-up experiments would include expanding the abbreviation and body
part dictionaries, tracking false negatives separately from false positives, and
testing a lightweight cached semantic feature only if deployment latency remains
safe.
