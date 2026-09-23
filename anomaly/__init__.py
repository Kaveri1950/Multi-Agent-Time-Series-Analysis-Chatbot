"""Domain-neutral anomaly detection.

One scoring contract, many detectors, several unrelated kinds of time series.

The claim this package exists to test is deliberately modest: not that a
model trained on stars transfers to stocks (that would be transfer learning,
and it is a much harder thing to defend), but that the same METHOD, refitted
independently on each dataset, works on both. Method generalisation, not
transfer.

Nothing in `core` or `detectors` may reference a domain. If a detector needs
to know that flux sits near 1.0, or that returns are centred on zero, it has
stopped being a general method and the experiment stops meaning anything.
Domain knowledge lives in `labels.py` and nowhere else.
"""
