"""LongBench evaluation harness for the IterRet/IMR model.

This package tests the *same* IterRet model (CTC graph + closed-loop retrieval
controller + experience bank) and the *same* surprise boundary module on
LongBench instead of LoCoMo. It reuses the core package (``iterret``) wholesale
and only adds what LongBench specifically needs:

- ``longbench_data``   -- load LongBench QA tasks (document + question).
- ``doc_segmenter``    -- surprise segmentation for *continuous documents*
                          (reuses ``iterret.surprise`` via
                          ``iterret.episode_segmenter``; no turn-snapping).
- ``doc_memory_builder`` -- build a CTC graph from document event-spans
                          (reuses ``iterret.memory_builder``'s semantic/topic
                          distillation).
- ``metrics``          -- LongBench's official qa_f1 metric.
- ``run_longbench_eval`` -- the runner (mirrors ``run_locomo_eval.py``).

The dialogue/LoCoMo path in ``iterret`` is untouched.
"""
