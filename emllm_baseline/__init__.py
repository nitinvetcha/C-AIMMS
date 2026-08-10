"""Vanilla long-context baseline on LongBench for the EM-LLM paper's backbones.

This is the *comparison baseline* that both the EM-LLM method (arXiv:2407.09450)
and our IterRet+surprise method are measured against: feed the model the
middle-truncated document + question directly (no memory system), using
LongBench's official per-task prompts and generation lengths, and score with the
same qa_f1 metric used by the `longbench/` harness -- so all three sets of
numbers are directly comparable.

Backbones (EM-LLM's exact set): LLaMA-3-8B-Instruct, Mistral-7B-Instruct-v0.2,
Phi-3.5-mini-instruct. It reuses `longbench.longbench_data` (data) and
`longbench.metrics` (scoring); only the prompting + truncation are new.
"""
