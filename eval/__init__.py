"""Reference-free evaluation harness for the MangaTVL pipeline.

Measures stages 3 (Translation), 4 (Inpainting) and 5 (Rendering) on raw
manga pages — no ground-truth labels / reference translations required. All
signals are derived either from the pipeline's own intermediate outputs or
from the bubble *segmentation polygon* (so "did inpaint clean inside the
bubble" / "did rendered text stay inside the bubble" become measurable).

See eval/README.md for the metric definitions.
"""
