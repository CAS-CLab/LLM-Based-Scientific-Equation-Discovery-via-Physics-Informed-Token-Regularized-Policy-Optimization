#!/usr/bin/env python3
"""One-GPU smoke test for the shared PiT-PO train -> generate transition."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pitpo.grpo_trainer import GRPOTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--training_state", required=True)
    parser.add_argument("--num_epochs", type=int, default=4)
    parser.add_argument("--detect_anomaly", action="store_true")
    parser.add_argument(
        "--gradient_checkpointing",
        choices=("unsloth", "standard", "off"),
        default="unsloth",
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    args = parser.parse_args()

    with open(args.training_state, encoding="utf-8") as handle:
        state = json.load(handle)
    experiences = state.get("training_buffer") or []
    if len(experiences) < 16:
        raise RuntimeError("Smoke test requires at least 16 saved experiences")

    trainer = GRPOTrainer(
        model_name=args.model_path,
        learning_rate=1e-6,
        batch_size=4,
        reward_scaling=10.0,
        min_mse_threshold=1e-15,
        buffer_size=100,
        max_seq_length=4096,
        load_in_4bit=True,
        fast_inference=True,
        vllm_gpu_memory_utilization=0.6,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    trainer.training_buffer = experiences
    trainer.scores_since_finetune = [
        float(experience["reward"]) for experience in experiences
    ]

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    stats = trainer.train_step(num_epochs=args.num_epochs)
    if stats.get("model_version") != 1:
        raise RuntimeError(f"Unexpected smoke-test model version: {stats}")

    outputs = trainer.generate(
        prompt=experiences[-1]["prompt"],
        num_return_sequences=4,
        max_new_tokens=256,
        temperature=0.6,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.1,
    )
    collapsed = [
        bool(text.strip()) and set(text.strip()) <= {"!"} for text in outputs
    ]
    print("SMOKE_STATS", stats, flush=True)
    for index, output in enumerate(outputs, start=1):
        print(f"SMOKE_OUTPUT_{index}", repr(output[:300]), flush=True)
    print("SMOKE_TRANSITION_SECONDS", trainer.consume_train_to_generation_seconds(), flush=True)

    if any(collapsed):
        raise RuntimeError("Post-training generation collapsed to exclamation marks")
    if not all(output.strip() for output in outputs):
        raise RuntimeError("Post-training generation returned an empty completion")


if __name__ == "__main__":
    main()
