# Evaluation Notes

## Final Approach

The production service uses a deterministic feature-based scorer. It compares
each current study with each prior study using body-part match, modality match,
recency, and keyword overlap. Embeddings are not loaded in the production API so
request handling remains fast and predictable on lightweight deployment
environments.

The scoring weights are:

| Signal | Weight |
| --- | ---: |
| Body-part match | 0.45 |
| Modality match | 0.20 |
| Recency | 0.20 |
| Keyword overlap | 0.15 |

## Evaluation Summary

| Version | Accuracy | Notes |
| --- | ---: | --- |
| Baseline Rules | ~0.83 | Initial heuristic model using coarse study matching. |
| Feature-Based Scoring | ~0.88 | Added text normalization, recency, and overlap scoring. |
| Embedding-Augmented | ~0.88+ | Improved semantic matching but introduced higher latency and deployment risk. |
| Final Production Model | ~0.88 | Optimized for stable low-latency inference. |

## Design Tradeoffs

Embedding-based matching can help when two study descriptions are semantically
related but share few exact tokens. In production, however, the extra model
loading and inference work increased latency and reliability risk for a compact
API service. The deterministic scorer was preferred because it has no warm-up
dependency, covers every prior study, and is straightforward to test.

## Error Analysis

The rule-based endpoint performs best when descriptions contain recognizable
body parts, modalities, and dates. Likely misses include uncommon abbreviations,
vague study descriptions, and priors that are clinically relevant despite only
partial text overlap.

## Clinical Workflow Notes

For a radiology workflow, missing a relevant prior can be more costly than
surfacing an extra irrelevant prior. A missed prior can remove context about
disease progression, stability, or prior interventions. Extra priors add review
burden, but they are easier to dismiss when the interface keeps results
organized.

Threshold tuning should consider that asymmetry alongside response latency. The
current threshold keeps the service fast and stable while avoiding obvious
irrelevant priors; future tuning should measure false negatives and false
positives separately rather than optimizing accuracy alone.

## Next Steps

Useful follow-up work would include expanding abbreviation coverage, reviewing
false negatives by body region and modality, and exploring cached lightweight
semantic features only if deployment latency remains stable.
