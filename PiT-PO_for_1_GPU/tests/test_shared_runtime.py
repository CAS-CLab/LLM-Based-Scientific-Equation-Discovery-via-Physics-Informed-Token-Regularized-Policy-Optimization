import pathlib
import sys
import types
import unittest
from unittest import mock

import torch

from pitpo import config
from pitpo import evaluator
from pitpo import grpo_trainer
from pitpo import sampler


class _FakeSharedTrainer:
    def __init__(self):
        self.calls = []
        self.transition_seconds = 0.125

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return [
            "def equation(x, params):\n    return x + params[0]\n",
            "def equation(x, params):\n    return x * params[0]\n",
        ]

    def consume_train_to_generation_seconds(self):
        value = self.transition_seconds
        self.transition_seconds = None
        return value


class _FakeModel:
    def __init__(self):
        self.config = type("Config", (), {"use_cache": None})()
        self.training = None

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


class _FastGenerationModel(_FakeModel):
    def __init__(self):
        super().__init__()
        self.loaded_lora = None

    def load_lora(self, path, load_tensors):
        self.loaded_lora = (path, load_tensors)
        return "live-lora-request"

    def fast_generate(self, prompts, sampling_params, lora_request, use_tqdm):
        self.fast_generate_call = {
            "prompts": prompts,
            "sampling_params": sampling_params,
            "lora_request": lora_request,
            "use_tqdm": use_tqdm,
        }
        outputs = [
            types.SimpleNamespace(text="first"),
            types.SimpleNamespace(text="second"),
        ]
        return [types.SimpleNamespace(outputs=outputs)]


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, _text, **_kwargs):
        return {"input_ids": [10, 11, 12]}

    def decode(self, token_ids, skip_special_tokens):
        assert token_ids == [10, 11, 12]
        assert not skip_special_tokens
        return "truncated prompt"


class _FakeFastLanguageModel:
    inference_calls = []
    training_calls = []

    @classmethod
    def for_inference(cls, model):
        cls.inference_calls.append(model)

    @classmethod
    def for_training(cls, model):
        cls.training_calls.append(model)


class SharedSamplerTests(unittest.TestCase):
    def test_local_sampler_uses_attached_runtime_without_http(self):
        runtime = _FakeSharedTrainer()
        llm = sampler.LocalLLM(samples_per_prompt=2)
        llm.set_grpo_trainer(runtime)

        samples = llm.draw_samples(
            "search prompt",
            config.Config(use_api=False, max_new_tokens=384),
        )

        self.assertEqual(len(samples), 2)
        self.assertEqual(runtime.calls[0]["prompt"], "search prompt")
        self.assertEqual(runtime.calls[0]["num_return_sequences"], 2)
        self.assertEqual(runtime.calls[0]["max_new_tokens"], 384)
        self.assertTrue(all("return" in sample for sample in samples))
        self.assertEqual(llm.consume_train_to_generation_seconds(), 0.125)
        self.assertIsNone(llm.consume_train_to_generation_seconds())

    def test_missing_shared_runtime_fails_fast(self):
        llm = sampler.LocalLLM(samples_per_prompt=1)
        with self.assertRaisesRegex(RuntimeError, "Shared model runtime"):
            llm.draw_samples("search prompt", config.Config(use_api=False))


class RuntimeModeTests(unittest.TestCase):
    def setUp(self):
        _FakeFastLanguageModel.inference_calls.clear()
        _FakeFastLanguageModel.training_calls.clear()
        self.runtime = grpo_trainer.GRPOTrainer.__new__(grpo_trainer.GRPOTrainer)
        self.runtime.model = _FakeModel()
        self.runtime._fast_language_model = _FakeFastLanguageModel
        self.runtime._runtime_mode = "initializing"

    def test_training_and_inference_reuse_the_identical_model(self):
        model_identity = id(self.runtime.model)

        self.runtime.enter_training_mode()
        self.assertEqual(self.runtime._runtime_mode, "training")
        self.assertTrue(self.runtime.model.training)
        self.assertFalse(self.runtime.model.config.use_cache)

        self.runtime.enter_inference_mode()
        self.assertEqual(self.runtime._runtime_mode, "inference")
        self.assertFalse(self.runtime.model.training)
        self.assertTrue(self.runtime.model.config.use_cache)
        self.assertEqual(id(self.runtime.model), model_identity)
        self.assertIs(_FakeFastLanguageModel.training_calls[-1], self.runtime.model)
        self.assertIs(_FakeFastLanguageModel.inference_calls[-1], self.runtime.model)

    def test_transition_metric_is_consumed_once(self):
        self.runtime._last_train_to_generation_seconds = 0.25
        self.assertEqual(
            self.runtime.consume_train_to_generation_seconds(), 0.25
        )
        self.assertIsNone(self.runtime.consume_train_to_generation_seconds())

    def test_fast_generation_loads_current_lora_tensors_in_memory(self):
        self.runtime.model = _FastGenerationModel()
        self.runtime.tokenizer = _FakeTokenizer()
        self.runtime.fast_inference = True
        self.runtime.use_lora = True
        self.runtime.max_seq_length = 4096
        self.runtime._runtime_lora_dir = "/tmp/pitpo-test-runtime-lora"
        self.runtime._last_training_completed_at = None
        self.runtime._last_train_to_generation_seconds = None

        class SamplingParams:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        fake_vllm = types.ModuleType("vllm")
        fake_vllm.SamplingParams = SamplingParams
        with mock.patch.dict(sys.modules, {"vllm": fake_vllm}):
            completions = self.runtime.generate(
                "original prompt", num_return_sequences=2
            )

        self.assertEqual(completions, ["first", "second"])
        self.assertEqual(
            self.runtime.model.loaded_lora,
            ("/tmp/pitpo-test-runtime-lora", True),
        )
        self.assertEqual(
            self.runtime.model.fast_generate_call["lora_request"],
            "live-lora-request",
        )
        self.assertEqual(
            self.runtime.model.fast_generate_call["prompts"],
            ["truncated prompt"],
        )


class RemovedDeploymentPathTests(unittest.TestCase):
    def test_local_code_has_no_server_restart_path(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        source = "\n".join(
            (root / relative).read_text(encoding="utf-8")
            for relative in (
                "pitpo/sampler.py",
                "pitpo/evaluator.py",
                "pitpo/pipeline.py",
            )
        )
        for forbidden in (
            "start_vllm_server",
            "restart_vllm_with_model",
            "subprocess.Popen",
            "_vllm_pid",
        ):
            self.assertNotIn(forbidden, source)


class _FailingTrainer:
    def __init__(self):
        self.training_buffer = []
        self.scores_since_finetune = []

    def add_experience(self, **kwargs):
        self.training_buffer.append({"mse": kwargs["mse"]})
        self.scores_since_finetune.append(1.0)

    def _is_valid_experience(self, _experience):
        return True

    def should_train(self):
        return True

    def train_step(self, num_epochs):
        raise RuntimeError(f"training failed after {num_epochs} epoch")


class FailurePolicyTests(unittest.TestCase):
    def test_training_failure_is_not_swallowed(self):
        instance = evaluator.Evaluator.__new__(evaluator.Evaluator)
        instance._grpo_trainer = _FailingTrainer()
        instance._training_counter = 0
        instance._train_every_n_valid = 1
        instance._strategy_config = None

        with self.assertRaisesRegex(RuntimeError, "training failed"):
            instance._handle_grpo_training(
                "    return x\n",
                {"data": -1.0},
                original_prompt="prompt",
                global_sample_nums=10,
            )


class LossStabilityTests(unittest.TestCase):
    def setUp(self):
        self.runtime = grpo_trainer.GRPOTrainer.__new__(grpo_trainer.GRPOTrainer)

    def test_padding_with_negative_infinite_logits_has_finite_gradients(self):
        logits = torch.randn(2, 4, 8)
        # These rows predict padding targets and deliberately mimic a model
        # masking its dedicated pad token with -inf logits.
        logits[1, 1:, :] = float("-inf")
        logits.requires_grad_()
        labels = torch.tensor(
            [[1, 2, 3, 4], [1, 2, -100, -100]], dtype=torch.long
        )
        attention_mask = torch.tensor(
            [[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.long
        )

        loss = self.runtime.compute_grpo_loss(
            logits=logits,
            labels=labels,
            rewards=torch.tensor([1.0, 2.0]),
            attention_mask=attention_mask,
            token_penalty_weights=torch.zeros(2, 4),
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_training_dataset_masks_padding_labels(self):
        dataset = [{
            "input_ids": [10, 11, 99, 99],
            "attention_mask": [1, 1, 0, 0],
            "rewards": 1.0,
            "token_penalty_weights": [0.0, 0.0, 0.0, 0.0],
        }]

        item = grpo_trainer.GRPOTrainer.GRPOTrainingDataset(dataset)[0]

        self.assertEqual(item["labels"].tolist(), [10, 11, -100, -100])

    def test_nonfinite_rewards_fail_before_backward(self):
        with self.assertRaisesRegex(FloatingPointError, "rewards"):
            self.runtime.compute_grpo_loss(
                logits=torch.randn(2, 3, 8),
                labels=torch.tensor([[1, 2, 3], [1, 2, 3]]),
                rewards=torch.tensor([1.0, float("nan")]),
                attention_mask=torch.ones(2, 3, dtype=torch.long),
            )

    def test_right_padding_is_unmasked_only_for_model_forward(self):
        self.runtime.tokenizer = types.SimpleNamespace(padding_side="right")
        policy_mask = torch.tensor([[1, 1, 0], [1, 0, 0]])

        model_mask = self.runtime._causal_training_attention_mask(policy_mask)

        self.assertEqual(model_mask.tolist(), [[1, 1, 1], [1, 1, 1]])
        self.assertEqual(policy_mask.tolist(), [[1, 1, 0], [1, 0, 0]])

    def test_left_padding_is_rejected_for_causal_unmasking(self):
        self.runtime.tokenizer = types.SimpleNamespace(padding_side="left")
        with self.assertRaisesRegex(RuntimeError, "right-side padding"):
            self.runtime._causal_training_attention_mask(torch.ones(1, 2))


if __name__ == "__main__":
    unittest.main()
