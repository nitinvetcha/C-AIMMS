"""Official LongBench per-task prompt templates, generation lengths, and the
EM-LLM backbone set. Values taken from THUDM/LongBench (config/dataset2prompt.json
and dataset2maxlen.json) so this baseline matches the benchmark's canonical setup.
"""

from __future__ import annotations

# Prompt templates for the QA subtasks (verbatim from LongBench).
DATASET2PROMPT = {
    "narrativeqa": (
        "You are given a story, which can be either a novel or a movie script, and a question. "
        "Answer the question as concisely as you can, using a single phrase if possible. Do not "
        "provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story "
        "as concisely as you can, using a single phrase if possible. Do not provide any explanation."
        "\n\nQuestion: {input}\n\nAnswer:"
    ),
    "qasper": (
        "You are given a scientific article and a question. Answer the question as concisely as you "
        "can, using a single phrase or sentence if possible. If the question cannot be answered based "
        "on the information in the article, write \"unanswerable\". If the question is a yes/no "
        "question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\n"
        "Article: {context}\n\n Answer the question based on the above article as concisely as you can, "
        "using a single phrase or sentence if possible. If the question cannot be answered based on the "
        "information in the article, write \"unanswerable\". If the question is a yes/no question, answer "
        "\"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:"
    ),
    "multifieldqa_en": (
        "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question "
        "based on the above text, only give me the answer and do not output any other words.\n\n"
        "Question: {input}\nAnswer:"
    ),
    "hotpotqa": (
        "Answer the question based on the given passages. Only give me the answer and do not output any "
        "other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the "
        "given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "2wikimqa": (
        "Answer the question based on the given passages. Only give me the answer and do not output any "
        "other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the "
        "given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "musique": (
        "Answer the question based on the given passages. Only give me the answer and do not output any "
        "other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the "
        "given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:"
    ),
}

# Max NEW tokens to generate per task (LongBench dataset2maxlen.json).
DATASET2MAXGEN = {
    "narrativeqa": 128, "qasper": 128, "multifieldqa_en": 64,
    "hotpotqa": 32, "2wikimqa": 32, "musique": 32,
}

# EM-LLM's backbones (short alias -> HF id). LLaMA and Mistral are gated on HF:
# you must `huggingface-cli login` (accept their licenses) before downloading.
EMLLM_MODELS = {
    "llama3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",
    "phi3.5-mini": "microsoft/Phi-3.5-mini-instruct",
}

# Default context budget for middle-truncation, per backbone (its usable window
# minus generation headroom). The vanilla baseline truncates the prompt to this;
# this is exactly the limited-window setting EM-LLM's method is designed to beat.
MODEL2MAXLEN = {
    "meta-llama/Meta-Llama-3-8B-Instruct": 7500,      # 8k window
    "mistralai/Mistral-7B-Instruct-v0.2": 31500,      # 32k window
    "microsoft/Phi-3.5-mini-instruct": 127000,        # 128k window
}
