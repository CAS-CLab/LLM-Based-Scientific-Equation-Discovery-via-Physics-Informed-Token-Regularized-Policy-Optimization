#!/usr/bin/env python3
"""Diagnose model-backward numerics with one model load and no optimizer step."""

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
    args = parser.parse_args()

    with open(args.training_state, encoding="utf-8") as handle:
        experiences = json.load(handle)["training_buffer"]

    runtime = GRPOTrainer(
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
        lora_dropout=0.0,
    )
    runtime.enter_training_mode()
    model = runtime.model
    tokenizer = runtime.tokenizer
    device = next(model.parameters()).device

    # Use the exact four rewards seen in the first failing Trainer micro-batch.
    target_rewards = (75.58385467529297, 72.51982879638672,
                      112.20693969726562, 72.934814453125)
    chosen = [
        min(experiences, key=lambda item: abs(item["reward"] - target))
        for target in target_rewards
    ]
    texts = [item["prompt"] + item["response"] for item in chosen]
    encoded = tokenizer(
        texts,
        truncation=True,
        # Reproduce prepare_training_data's global length for the 16 samples
        # used by the failing update, rather than padding only to this batch.
        padding="max_length",
        max_length=431,
        return_tensors="pt",
    )
    original_ids = encoded["input_ids"].to(device)
    original_mask = encoded["attention_mask"].to(device)
    original_rewards = torch.tensor(target_rewards, device=device)

    def report_gradients(case_name: str) -> None:
        bad_names = []
        tensors_with_grad = 0
        finite_abs_max = 0.0
        for name, parameter in model.named_parameters():
            gradient = parameter.grad
            if not parameter.requires_grad or gradient is None:
                continue
            tensors_with_grad += 1
            finite = torch.isfinite(gradient)
            if not bool(finite.all()):
                bad_names.append(name)
            if bool(finite.any()):
                finite_abs_max = max(
                    finite_abs_max,
                    float(gradient[finite].abs().max().item()),
                )
        print(
            f"DIAG_CASE {case_name} tensors={tensors_with_grad} "
            f"bad={len(bad_names)} finite_abs_max={finite_abs_max:.6e}",
            flush=True,
        )
        if bad_names:
            print("DIAG_BAD", bad_names[:8], flush=True)

    def policy_backward(
        case_name: str,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        rewards: torch.Tensor,
        token_penalty_weights: torch.Tensor | None = None,
    ) -> None:
        model.zero_grad(set_to_none=True)
        labels = input_ids.clone()
        labels.masked_fill_(attention_mask.eq(0), -100)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits
            loss = runtime.compute_grpo_loss(
                logits=logits,
                labels=labels,
                rewards=rewards,
                attention_mask=attention_mask,
                token_penalty_weights=token_penalty_weights,
            )
            # Match the Trainer's four-way gradient accumulation scaling.
            loss = loss / 4.0
        loss.backward()
        report_gradients(case_name)

    policy_backward(
        "original_right_padding",
        original_ids,
        original_mask,
        original_rewards,
    )

    # Reconstruct the exact per-token coefficient penalty used by train_step.
    penalty_dataset = runtime.prepare_training_data(chosen)
    penalty_rows = [
        torch.tensor(row, dtype=torch.float32, device=device)
        for row in penalty_dataset["token_penalty_weights"]
    ]
    penalty_weights = torch.nn.utils.rnn.pad_sequence(
        penalty_rows,
        batch_first=True,
        padding_value=0.0,
    )
    if penalty_weights.shape[1] < original_ids.shape[1]:
        penalty_weights = torch.nn.functional.pad(
            penalty_weights,
            (0, original_ids.shape[1] - penalty_weights.shape[1]),
        )
    policy_backward(
        "original_with_token_penalty",
        original_ids,
        original_mask,
        original_rewards,
        penalty_weights,
    )
    policy_backward(
        "original_with_tenth_token_penalty",
        original_ids,
        original_mask,
        original_rewards,
        penalty_weights * 0.1,
    )

    # Duplicate one sequence so every row has the same non-padded length while
    # retaining non-zero group-normalized advantages.
    source_length = int(original_mask[0].sum().item())
    no_pad_ids = original_ids[0, :source_length].unsqueeze(0).repeat(4, 1)
    no_pad_mask = torch.ones_like(no_pad_ids)
    contrast_rewards = torch.tensor([0.0, 1.0, 2.0, 3.0], device=device)
    policy_backward(
        "duplicate_without_padding",
        no_pad_ids,
        no_pad_mask,
        contrast_rewards,
    )

    # Add right padding back to the duplicated sequence.  This changes only
    # the padded suffix; all valid tokens and their policy advantages match.
    padding_length = max(1, original_ids.shape[1] - source_length)
    pad_ids = torch.full(
        (4, padding_length), tokenizer.pad_token_id,
        dtype=no_pad_ids.dtype, device=device,
    )
    duplicate_padded_ids = torch.cat((no_pad_ids, pad_ids), dim=1)
    duplicate_padded_mask = torch.cat(
        (no_pad_mask, torch.zeros_like(pad_ids)), dim=1
    )
    policy_backward(
        "duplicate_with_right_padding",
        duplicate_padded_ids,
        duplicate_padded_mask,
        contrast_rewards,
    )

    # Reproduce every batch from train_step, including its selection order,
    # global padding, token penalties, and deterministic shuffle.
    threshold = float(torch.tensor(
        [item["reward"] for item in experiences], dtype=torch.float64
    ).quantile(0.3).item())
    selected = [
        item for item in experiences
        if item.get("mse") is not None
        and float(item["mse"]) > 0.0
        and float(item["reward"]) >= threshold
    ][-16:]
    selected_dataset = runtime.prepare_training_data(selected)
    selected_ids = torch.tensor(
        selected_dataset["input_ids"], dtype=torch.long, device=device
    )
    selected_mask = torch.tensor(
        selected_dataset["attention_mask"], dtype=torch.long, device=device
    )
    selected_penalties = torch.tensor(
        selected_dataset["token_penalty_weights"],
        dtype=torch.float32,
        device=device,
    )
    selected_rewards = torch.tensor(
        selected_dataset["rewards"], dtype=torch.float32, device=device
    )
    order = torch.randperm(16, generator=torch.Generator().manual_seed(3408))
    for batch_number, indices in enumerate(order.split(4), start=1):
        indices = indices.to(device)
        policy_backward(
            f"train_step_batch_{batch_number}",
            selected_ids[indices],
            selected_mask[indices],
            selected_rewards[indices],
            selected_penalties[indices],
        )

    # Baseline using Unsloth's own masked language-model loss.
    model.zero_grad(set_to_none=True)
    sft_labels = original_ids.clone()
    sft_labels.masked_fill_(original_mask.eq(0), -100)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        sft_loss = model(
            input_ids=original_ids,
            attention_mask=original_mask,
            labels=sft_labels,
        ).loss / 4.0
    sft_loss.backward()
    report_gradients("builtin_sft_loss")


if __name__ == "__main__":
    main()
