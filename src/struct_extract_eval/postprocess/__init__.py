"""Post-processing hooks for field results.

These run after scoring and batch dispatch, before metrics are computed.
Pass any of them (or your own) to ``evaluate(post_process=...)``.

Built-in post-processors:
- ``reclassify_nulls`` -- treat null/empty values as absent (omission/hallucination/skip)
"""

from struct_extract_eval.postprocess.null_handling import NullHandling, reclassify_nulls

__all__ = [
    "NullHandling",
    "reclassify_nulls",
]
