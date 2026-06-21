# lunaris — Lunar Subsurface-Ice Detection & Mission-Planning Platform

`lunaris` is an end-to-end platform for the ISRO BAH 2026 Problem Statement 8.
It detects subsurface water-ice in a lunar South-Pole *doubly-shadowed crater*
(Faustini PSR) from Chandrayaan-2 DFSAR radar polarimetry — applying the refined
`CPR > 1 ∧ DOP < 0.13` criterion and fusing it with 30+ multi-sensor datasets —
then plans a scientifically viable landing site, an energy-aware rover traverse,
and a top-5 m ice-volume estimate with uncertainty.

Quick start:

```bash
python -m venv .venv && .venv/bin/pip install -e .
lunaris demo            # generate the synthetic Faustini scene + ice summary
.venv/bin/python -m pytest tests/test_foundation.py -q
```

See `ARCHITECTURE.md` (WIP) for the full design, module map, and pipeline
stages.
