
import os
# Avoid the tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import numpy as np
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import Dataset
import math
import re
import csv
import json

# Equation-analysis reward
from .equation_functions import (
    extract_equation_functions_from_text,
    analyze_equation_function,
)
from .coef_penalty import compute_coefficient_penalty

# PEFT / LoRA support
try:
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
except Exception as _peft_err:
    LoraConfig = None
    get_peft_model = None
    TaskType = None
    PeftModel = tuple()  # fallback for isinstance checks


logger = logging.getLogger(__name__)

# EvoTune-aligned: matches `percentile: 70` in `configs/config.yaml`; threshold =
#   np.percentile(scores_since_tuning, 100 - percentile)
# Intentionally hard-coded; not exposed as a configurable option.
_DPO_STYLE_PERCENTILE = 70


class GRPOTrainer:
    """
    Group Robust Policy Optimization trainer for fine-tuning language models.
    Uses MSE as reward signal where lower MSE results in higher reward.
    """
    
    def __init__(
        self,
        model_name: str,
        learning_rate: float = 1e-5,
        batch_size: int = 4,
        max_length: int = 512,
        reward_scaling: float = 10.0,
        min_mse_threshold: float = 1e-15,
        buffer_size: int = 1000,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        required_train_samples: int = 16,
        # LoRA-related configuration (enabled by default; only LoRA params are trained)
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
        # Tunable training hyperparameters: per-device batch and gradient accumulation
        per_device_train_batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        # Warmup / skip-first-N-generations control
        warmup_generations: int = 4,
        samples_per_prompt: Optional[int] = None,
        warmup_min_global_samples: Optional[int] = None,
        # PiT-PO Dual-Constraint configs (Paper Section 3.1 & 3.2)
        # -- AST complexity penalty P_cplx (Eq. 8)
        ast_complexity_weight: float = 0.05,
        # -- Gated physical penalty P_phy (Eq. 9)
        enable_physical_penalty: bool = True,
        phy_penalty_dim: float = 2.0,
        phy_penalty_diff: float = 1.5,
        phy_gate_mse_threshold: float = 1.0,
        phy_units: Optional[Dict[str, str]] = None,
        # -- Coefficient redundancy penalty via Support Exclusion Theorem (Eq. 5-6)
        enable_coef_penalty: bool = True,
        coef_ratio_threshold: float = 0.05,
        coef_penalty_weight: float = 1.0,
    ):
        """
        Initialize GRPO trainer.
        
        Args:
            model_name: Hugging Face model name or path
            learning_rate: Learning rate for optimization
            batch_size: Batch size for training
            max_length: Maximum sequence length
            reward_scaling: Scaling factor for log reward transformation
            min_mse_threshold: Minimum MSE threshold to prevent log(0)
            buffer_size: Maximum size of training buffer
            device: Device to run on (cuda/cpu)
            required_train_samples: Number of valid experiences required per fine-tuning step
            use_lora: Whether to use LoRA and train only LoRA adapters
            lora_r/lora_alpha/lora_dropout: LoRA hyperparameters
            lora_target_modules: Target modules for LoRA injection; if None will be resolved for common LLaMA blocks
            per_device_train_batch_size: training batch size per device
            gradient_accumulation_steps: gradient-accumulation steps (effective batch = per_device * accum)
            warmup_generations: number of generations to skip GRPO fine-tuning for (default: 4)
            samples_per_prompt: samples per prompt (used to infer the global sample count from generations)
            warmup_min_global_samples: explicit minimum global_sample_nums threshold to trigger training; overrides the two above when provided
            ast_complexity_weight: lambda_len for P_cplx = lambda_len * AST_node_count (Paper Eq. 8)
            enable_physical_penalty: enable gated physical penalty P_phy (Paper Eq. 9)
            phy_penalty_dim: penalty for dimensional inconsistency
            phy_penalty_diff: penalty for non-differentiability
            phy_gate_mse_threshold: delta_gate; physical penalties only activate when MSE < this
            phy_units: unit mapping for dimensional analysis
            enable_coef_penalty: enable Support Exclusion Theorem penalty (Paper Eq. 5-6)
            coef_ratio_threshold: rho threshold on tau_i = |b_i|/(sum|b_j|+eps)
            coef_penalty_weight: scaling coefficient p in P_tok = p * max(0, -ln(|b_i|+eps))
        """
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_length = max_length
        self.reward_scaling = reward_scaling
        self.min_mse_threshold = min_mse_threshold
        self.buffer_size = buffer_size
        self.device = device
        # Allow reading single-GPU id from env var or attribute (default 0)
        self.single_device_id = os.environ.get('LLMSR_DEVICE_ID', '0')
        self.required_train_samples = required_train_samples
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules
        # Save tunable training hyperparameters
        self.per_device_train_batch_size = per_device_train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Warmup-related configuration
        self.warmup_generations = warmup_generations
        self.samples_per_prompt = samples_per_prompt
        # Compat: if an explicit threshold is given, use it; otherwise derive from generations and samples_per_prompt; otherwise degrade to warmup_generations+1
        if warmup_min_global_samples is not None:
            self.warmup_min_global_samples = int(warmup_min_global_samples)
        elif self.samples_per_prompt is not None:
            # global_sample_nums starts at 1; skipping N gens => threshold = N*samples_per_prompt + 1
            self.warmup_min_global_samples = int(self.warmup_generations * self.samples_per_prompt + 1)
        else:
            # When samples-per-generation is unknown, approximate as 1 per gen — still skips the first N reliably
            self.warmup_min_global_samples = int(self.warmup_generations + 1)
        
        # AST complexity penalty (Paper Eq. 8)
        self.ast_complexity_weight = float(ast_complexity_weight)

        # Gated physical penalty (Paper Eq. 9)
        self.enable_physical_penalty = bool(enable_physical_penalty)
        self.phy_penalty_dim = float(phy_penalty_dim)
        self.phy_penalty_diff = float(phy_penalty_diff)
        self.phy_gate_mse_threshold = float(phy_gate_mse_threshold)
        self.phy_units = dict(phy_units) if phy_units else None

        # Coefficient redundancy penalty (Paper Eq. 5-6)
        self.enable_coef_penalty = bool(enable_coef_penalty)
        self.coef_ratio_threshold = float(coef_ratio_threshold)
        self.coef_penalty_weight = float(coef_penalty_weight)
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Force device_map setting
        force_map = None
        if self.device == 'cuda' and torch.cuda.is_available():
            force_map = {"": int(self.single_device_id)}  # single-GPU placement
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=force_map if force_map is not None else ("auto" if device == "cuda" else None)
        )
        try:
            print(f"[INF] Loaded model with hf_device_map={getattr(self.model, 'hf_device_map', None)} on device={self.device}")
        except Exception:
            pass
        # Apply LoRA (only LoRA parameters are trained)
        if self.use_lora:
            if LoraConfig is None or get_peft_model is None:
                raise RuntimeError(
                    "PEFT/LoRA is not installed; install with: pip install peft"
                )
            # Disable cache for compatibility with gradient checkpointing / PEFT training
            if hasattr(self.model, "config"):
                try:
                    self.model.config.use_cache = False
                except Exception:
                    pass
            # Resolve target modules (defaults aimed at LLaMA)
            target_modules = self.lora_target_modules or [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
            lora_cfg = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=target_modules,
                task_type=TaskType.CAUSAL_LM,
            )
            self.model = get_peft_model(self.model, lora_cfg)
            try:
                self.model.print_trainable_parameters()
            except Exception:
                pass
        
        # Training data buffer (long-running, analogous to EvoTune's train_dataset)
        self.training_buffer = []
        # Reward of each new experience since the last successful train_step (EvoTune's scores_since_finetune)
        self.scores_since_finetune: list[float] = []
        
        logger.info(f"Initialized GRPO trainer with model {model_name} | LoRA: {self.use_lora} | warmup_min_global_samples={self.warmup_min_global_samples}")

    def _is_valid_experience(self, exp: Dict[str, Any]) -> bool:
        """Check if an experience has a valid (non-None) score/MSE.
        Additional rule: if metadata contains a score field that is positive (usually meaning MSE was negated), treat the entry as invalid.
        """
        try:
            mse = exp.get("mse", None)
            if mse is None:
                return False
            # MSE must be a finite positive number
            if not (isinstance(mse, (int, float)) and np.isfinite(mse) and mse > 0):
                return False
            # If a positive score is provided (meaning -MSE > 0, i.e. MSE < 0), treat as invalid
            meta = exp.get("metadata", {}) or {}
            score_val = None
            for key in ("score", "raw_score", "eval_score"):
                val = meta.get(key, None)
                if val is None:
                    continue
                try:
                    score_val = float(val)
                    break
                except Exception:
                    continue
            if score_val is not None and score_val > 0:
                return False
            return True
        except Exception:
            return False

    def _evotune_reward_threshold(self) -> float:
        """Matches EvoTune `calculate_dataset_statistics`: take the (100 - percentile) quantile of scores_since_finetune."""
        if not self.scores_since_finetune:
            return float("-inf")
        arr = np.asarray(self.scores_since_finetune, dtype=np.float64)
        return float(np.percentile(arr, 100 - _DPO_STYLE_PERCENTILE))

    def mse_to_reward(self, mse: float) -> float:
        """
        Convert MSE to reward using enhanced log transformation.
        Lower MSE results in higher reward, very high MSE becomes penalty.
        
        Args:
            mse: Mean squared error value
            
        Returns:
            Reward value (positive for good performance, negative for poor performance)
        """
        # Ensure MSE is at least min_threshold to prevent log(0)
        mse = max(mse, self.min_mse_threshold)
        
        # Enhanced reward function based on log transformation
        # reward = -log(MSE) * scaling, but with penalty for very poor performance
        base_reward = -np.log(mse) * self.reward_scaling
        
        # Add penalty for very poor performance (MSE > 10)
        # This creates a stronger negative signal for bad solutions
        if mse > 10.0:
            penalty_factor = np.log(mse / 10.0) * self.reward_scaling * 0.5
            base_reward -= penalty_factor
        
        # Bonus for excellent performance (MSE < 0.01)
        elif mse < 0.01:
            bonus_factor = np.log(0.01 / mse) * self.reward_scaling * 0.3
            base_reward += bonus_factor
            
        return base_reward

    def add_experience(
        self,
        prompt: str,
        response: str,
        mse: float,
        **metadata
    ):
        """
        Add training experience to buffer with enhanced management strategy.
        
        Args:
            prompt: Input prompt
            response: Model response
            mse: Mean squared error for this response
            **metadata: Additional metadata
        """
        # First, filter for input validity and positive-score entries (score>0 => drop)
        if mse is None or not np.isfinite(mse) or mse <= 0:
            logger.debug(f"Discarding experience due to invalid MSE: {mse}")
            return
        meta_score = None
        for key in ("score", "raw_score", "eval_score"):
            if key in metadata and metadata[key] is not None:
                try:
                    meta_score = float(metadata[key])
                    break
                except Exception:
                    continue
        if meta_score is not None and meta_score > 0:
            logger.debug(f"Discarding experience due to positive score (implies negative MSE): score={meta_score}, MSE={mse}")
            return
        
        # === R_fit: Fitting accuracy reward (Paper Eq. 7) ===
        r_fit = self.mse_to_reward(mse)

        # === P_cplx: AST complexity penalty (Paper Eq. 8) ===
        p_cplx = 0.0
        ast_count = 0
        try:
            fns = extract_equation_functions_from_text(response)
            if fns:
                analysis = analyze_equation_function(fns[0].source, symbol_units=self.phy_units or {})
                ast_count = int(analysis.get("ast_node_count", 0))
                p_cplx = self.ast_complexity_weight * ast_count
        except Exception:
            pass

        # === P_phy: Gated physical penalty (Paper Eq. 9) ===
        p_phy = 0.0
        dim_ok = True
        diff_ok = True
        if self.enable_physical_penalty and mse < self.phy_gate_mse_threshold:
            p_phy = self._compute_physical_penalty(response, mse)

        # === R_global = R_fit - P_cplx - P_phy (Paper Eq. 10) ===
        reward = r_fit - p_cplx - p_phy

        # === Coefficient redundancy penalty for token-level (Paper Eq. 5-6) ===
        coef_penalty_val = 0.0
        coef_details = {}
        if self.enable_coef_penalty:
            try:
                optimized_params_by_test = metadata.get("optimized_params_by_test", None)
            except Exception:
                optimized_params_by_test = None
            if optimized_params_by_test:
                coef_penalty_val, coef_details = compute_coefficient_penalty(
                    response_text=response,
                    optimized_params_by_test=optimized_params_by_test,
                    ratio_threshold=self.coef_ratio_threshold,
                    penalty_weight=self.coef_penalty_weight,
                )
                if coef_penalty_val > 0:
                    reward -= coef_penalty_val
                    logger.debug(f"Applied coefficient penalty: {coef_penalty_val:.4f}; reward now {reward:.2f}")
        
        experience = {
            "prompt": prompt,
            "response": response,
            "mse": mse,
            "reward": reward,
            "metadata": metadata,
            "timestamp": time.time()  # Add timestamp for experience aging
        }
        
        # Record reward decomposition for debugging
        try:
            experience["metadata"]["reward_decomposition"] = {
                "r_fit": float(r_fit),
                "p_cplx": float(p_cplx),
                "ast_node_count": int(ast_count),
                "p_phy": float(p_phy),
                "coef_penalty": float(coef_penalty_val),
                "total_reward": float(reward),
            }
            if coef_details:
                experience["metadata"]["coef_penalty_details"] = coef_details
        except Exception:
            pass

        self.training_buffer.append(experience)
        
        # Enhanced buffer management
        if len(self.training_buffer) > self.buffer_size:
            # Strategy: Keep best experiences + some diversity
            self.training_buffer.sort(key=lambda x: x["reward"], reverse=True)
            
            # Keep top 70% best experiences
            top_keep = int(self.buffer_size * 0.7)
            top_experiences = self.training_buffer[:top_keep]
            
            # From remaining, keep 30% for diversity (varied MSE)
            remaining = self.training_buffer[top_keep:]
            remaining_keep = self.buffer_size - top_keep
            
            if len(remaining) > remaining_keep:
                # Sort remaining by MSE diversity (keep spread of MSE values)
                mse_values = [exp["mse"] for exp in remaining]
                mse_sorted_indices = np.argsort(mse_values)
                
                # Take evenly spaced samples for diversity
                step = len(mse_sorted_indices) // remaining_keep if remaining_keep > 0 else len(mse_sorted_indices)
                diverse_indices = mse_sorted_indices[::step][:remaining_keep]
                diverse_experiences = [remaining[i] for i in diverse_indices]
            else:
                diverse_experiences = remaining
                
            self.training_buffer = top_experiences + diverse_experiences
            
        # EvoTune: every experience successfully written to the buffer has an associated score for the next pre-fine-tune quantile threshold
        self.scores_since_finetune.append(float(reward))
        
        logger.debug(f"Added experience: MSE={mse:.2e}, Reward={reward:.2f}, Buffer size={len(self.training_buffer)}")
        
        # Log buffer statistics periodically
        if len(self.training_buffer) % 100 == 0:
            mse_values = [exp["mse"] for exp in self.training_buffer]
            logger.info(f"Buffer stats - Size: {len(self.training_buffer)}, "
                       f"MSE range: [{np.min(mse_values):.2e}, {np.max(mse_values):.2e}], "
                       f"MSE mean: {np.mean(mse_values):.2e}")
    
    def prepare_training_data(self, experiences: Optional[List[Dict[str, Any]]] = None) -> Dataset:
        """
        Prepare training dataset from buffer experiences.
        If `experiences` is provided, only those are used; otherwise use the entire buffer.
        
        Returns:
            Hugging Face Dataset for training
        """
        exps = experiences if experiences is not None else self.training_buffer
        if not exps:
            raise ValueError("No training data in buffer")
        
        # Prepare input texts and rewards
        texts: List[str] = []
        rewards: List[float] = []
        
        for exp in exps:
            # Combine prompt and response for training
            full_text = exp["prompt"] + exp["response"]
            texts.append(full_text)
            rewards.append(exp["reward"])
        
        # Tokenize texts
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Build per-token penalty weights (true token-level weighting)
        # 1) Compute offset mappings aligned with above tokenization
        try:
            enc_with_offsets = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_offsets_mapping=True,
            )
            offsets_batch = enc_with_offsets.get("offset_mapping", None)
        except Exception:
            offsets_batch = None

        token_penalty_weights: List[List[float]] = []
        for exp_idx, exp in enumerate(exps):
            seq_len = int(encodings["input_ids"][exp_idx].shape[0]) if hasattr(encodings["input_ids"][exp_idx], 'shape') else len(encodings["input_ids"][exp_idx])
            weights_vec = [0.0] * seq_len

            # Recover prompt/response for this sample
            prompt_text = exp.get("prompt", "")
            response_text = exp.get("response", "")
            full_text = (prompt_text or "") + (response_text or "")
            prompt_char_len = len(prompt_text or "")

            # Small-index penalties per index based on optimized params
            penalty_by_index: Dict[int, float] = {}
            small_indices: List[int] = []
            try:
                from .coef_penalty import compute_coefficient_penalty
                opt_params = (exp.get("metadata", {}) or {}).get("optimized_params_by_test", None)
                if opt_params:
                    _penalty_total, details = compute_coefficient_penalty(
                        response_text=response_text,
                        optimized_params_by_test=opt_params,
                        ratio_threshold=getattr(self, 'coef_ratio_threshold', 0.05),
                        penalty_weight=getattr(self, 'coef_penalty_weight', 1.0),
                    )
                    penalty_by_index = details.get("penalty_by_index", {}) if isinstance(details, dict) else {}
                    small_indices = details.get("redundant_indices", details.get("small_indices", [])) if isinstance(details, dict) else []
            except Exception:
                penalty_by_index = {}
                small_indices = []

            # Build char spans for occurrences of params[i] in response
            spans: List[Tuple[int, int, float]] = []  # (global_start, global_end, weight)
            if response_text and small_indices:
                for i in small_indices:
                    if i in penalty_by_index and penalty_by_index[i] > 0:
                        pattern = rf"params\s*\[\s*{i}\s*\]"
                        for m in re.finditer(pattern, response_text):
                            s, e = m.span()
                            # map to full_text global char positions
                            spans.append((prompt_char_len + s, prompt_char_len + e, float(penalty_by_index[i])))

            # Project spans onto token offsets (if available)
            if offsets_batch is not None and isinstance(offsets_batch, list) and exp_idx < len(offsets_batch):
                offsets = offsets_batch[exp_idx]
                for t_idx, off in enumerate(offsets):
                    try:
                        tok_s, tok_e = int(off[0]), int(off[1])
                    except Exception:
                        continue
                    if tok_e <= tok_s:
                        continue
                    # overlap with any span
                    w_sum = 0.0
                    for (gs, ge, w) in spans:
                        if not (tok_e <= gs or tok_s >= ge):
                            w_sum += float(w)
                    if w_sum > 0:
                        weights_vec[t_idx] = w_sum
            # If no offsets, leave weights as zeros (fallback to sample-level penalty only)
            token_penalty_weights.append(weights_vec)

        # Create dataset
        dataset_dict = {
            "input_ids": encodings["input_ids"].tolist(),
            "attention_mask": encodings["attention_mask"].tolist(),
            "rewards": rewards,
            "token_penalty_weights": token_penalty_weights,
        }
        
        dataset = Dataset.from_dict(dataset_dict)
        return dataset

    def compute_grpo_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        rewards: torch.Tensor,
        attention_mask: torch.Tensor,
        token_penalty_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute PiT-PO loss with token-aware advantage estimation (Paper Eq. 11-12).

        Token-aware advantage:
            A_hat_{i,k} = (R_global(o_i) - mu_group) / sigma_group  -  P_{i,k}

        The policy gradient:
            nabla J  proportional to  sum_{i,k} A_hat_{i,k} * nabla log pi(t_{i,k})
        """
        # Shift logits and labels for language modeling
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_attention_mask = attention_mask[..., 1:].contiguous()

        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        gathered_log_probs = log_probs.gather(
            dim=-1,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        # === Group normalization of rewards (Paper Eq. 4) ===
        mu = rewards.mean()
        sigma = rewards.std().clamp_min(1e-8)
        normalized_advantages = (rewards - mu) / sigma  # [batch_size]

        # === Token-aware advantage (Paper Eq. 11) ===
        # Expand normalized advantage to per-token: [batch_size, seq_len-1]
        seq_advantages = normalized_advantages.unsqueeze(-1).expand_as(gathered_log_probs)

        if token_penalty_weights is not None:
            shift_penalty = token_penalty_weights[..., 1:].contiguous()
            seq_advantages = seq_advantages - shift_penalty

        # === Policy gradient loss (Paper Eq. 12) ===
        weighted_log_probs = seq_advantages * gathered_log_probs * shift_attention_mask
        per_seq_loss = weighted_log_probs.sum(dim=-1) / shift_attention_mask.sum(dim=-1).clamp_min(1.0)
        grpo_loss = -per_seq_loss.mean()

        return grpo_loss

    class GRPOTrainingDataset(torch.utils.data.Dataset):
        """Custom dataset for GRPO training with proper tensor handling."""
        def __init__(self, dataset):
            self.dataset = dataset
            
        def __len__(self):
            return len(self.dataset)
            
        def __getitem__(self, idx):
            item = self.dataset[idx]
            result = {
                "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
                "labels": torch.tensor(item["input_ids"], dtype=torch.long),  # For language modeling
                "grpo_rewards": item["rewards"],  # use a different name to avoid being dropped
                "token_penalty_weights": torch.tensor(item.get("token_penalty_weights", [0.0]*len(item["input_ids"])), dtype=torch.float32),
            }
            # Debug info
            if idx == 0:  # print debug info only for the first sample
                logger.info(f"Dataset __getitem__ returning keys: {list(result.keys())}")
                logger.info(f"Rewards value: {result['grpo_rewards']}")
            return result
    
    def train_step(self, num_epochs: int = 1) -> Dict[str, float]:
        """
        Perform one training step using experiences in buffer with enhanced strategy.
        
        Args:
            num_epochs: Number of training epochs
            
        Returns:
            Training statistics
        """
        # Use only samples with a valid score (MSE positive and finite)
        valid_exps = [exp for exp in self.training_buffer if self._is_valid_experience(exp)]
        if len(valid_exps) == 0:
            logger.warning("No valid data for training (all scores invalid). Skipping.")
            return {}
        # EvoTune-style: reward >= np.percentile(scores_since_finetune, 100 - _DPO_STYLE_PERCENTILE)
        reward_thr = self._evotune_reward_threshold()
        filtered = [e for e in valid_exps if e.get("reward", float("-inf")) >= reward_thr]
        if len(filtered) == 0:
            logger.warning(
                "EvoTune-style filter removed all valid samples; falling back to unfiltered valid_exps."
            )
            filtered = valid_exps
        take_count = min(len(filtered), self.required_train_samples)
        selected_exps = filtered[-take_count:]
        
        logger.info("Starting GRPO reinforcement-learning fine-tune...")
        logger.info(
            f"📚 EvoTune-style: magic percentile={_DPO_STYLE_PERCENTILE} → reward threshold={reward_thr:.6f} "
            f"| kept {len(filtered)}/{len(valid_exps)} valid in buffer | train batch {len(selected_exps)}"
        )
        logger.info(f"Training data: using {len(selected_exps)} valid samples")
        logger.info(f"Training epochs: {num_epochs}")
        
        # Prepare training data
        dataset = self.prepare_training_data(experiences=selected_exps)
        train_dataset = self.GRPOTrainingDataset(dataset)
        
        # Calculate buffer statistics for adaptive learning rate
        mse_values = [exp["mse"] for exp in selected_exps]
        avg_mse = np.mean(mse_values)
        mse_std = np.std(mse_values)
        
        # Adaptive learning rate based on MSE quality
        adaptive_lr = self.learning_rate
        if avg_mse < 0.1:  # Very good results - use higher LR to exploit
            adaptive_lr = self.learning_rate * 1.5
        elif avg_mse > 10.0:  # Poor results - use lower LR for stability
            adaptive_lr = self.learning_rate * 0.5
        
        # Adaptive batch size (at least 1, at most half the samples or batch_size)
        adaptive_batch_size = min(self.batch_size, max(1, len(selected_exps) // 2))
        adaptive_batch_size = max(adaptive_batch_size, 1)
        
        # Ensure model is on the designated GPU
        target_device = f"cuda:{self.single_device_id}"
        if torch.cuda.is_available():
            cur_dev = str(next(self.model.parameters()).device)
            if cur_dev != target_device:
                logger.info(f"Moving model from {cur_dev} to {target_device}")
                self.model.to(target_device)

        # Setup training arguments
        training_args = TrainingArguments(
            output_dir="./grpo_training",
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            num_train_epochs=num_epochs,
            learning_rate=adaptive_lr,
            logging_steps=max(1, len(train_dataset) // max(1, (self.per_device_train_batch_size * self.gradient_accumulation_steps))),
            save_strategy="no",
            report_to=None,
            warmup_steps=0,
            weight_decay=0.01,
            dataloader_drop_last=False,
            # Must be disabled: otherwise the Trainer drops grpo_rewards / token_penalty_weights and the collator only sees defaults of 0.0
            remove_unused_columns=False,
        )
        
        # Custom trainer class for GRPO loss
        class CustomTrainer(Trainer):
            def __init__(self, grpo_trainer_instance, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.grpo_trainer = grpo_trainer_instance
                
            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
                """Compute GRPO loss with flexible parameter handling for transformers 4.55.0+"""
                # Extract rewards and labels from inputs
                rewards = inputs.pop("rewards")
                labels = inputs.pop("labels")
                # Print current batch rewards
                try:
                    if isinstance(rewards, torch.Tensor):
                        print(f"[GRPO] Batch rewards: {rewards.detach().cpu().tolist()}")
                    else:
                        print(f"[GRPO] Batch rewards: {rewards}")
                except Exception:
                    pass
                # Extract optional token penalty weights to avoid passing to model.forward
                token_penalty_weights = inputs.pop("token_penalty_weights", None)
                
                # Forward pass
                outputs = model(**inputs)
                logits = outputs.logits
                
                # Compute GRPO loss
                loss = self.grpo_trainer.compute_grpo_loss(
                    logits=logits,
                    labels=labels,
                    rewards=rewards,
                    attention_mask=inputs["attention_mask"],
                    token_penalty_weights=token_penalty_weights,
                )
                
                return (loss, outputs) if return_outputs else loss
                
            def _prepare_inputs(self, inputs):
                """Override to preserve custom fields like rewards"""
                inputs = super()._prepare_inputs(inputs)
                return inputs
        
        # Define a custom data collator class
        class GRPODataCollator:
            def __init__(self, tokenizer):
                self.tokenizer = tokenizer
                
            def __call__(self, features):
                """Custom data collator that preserves rewards field."""
                # Debug info
                logger.info(f"Data collator received {len(features)} features")
                if features:
                    logger.info(f"First feature keys: {list(features[0].keys())}")
                
                # Extract rewards (using the new field name)
                rewards = [f.pop("grpo_rewards", 0.0) for f in features]  # pop to avoid conflicts
                
                # Let transformers handle other fields with its default behavior
                batch = {}
                
                # Manually handle each field
                if features:
                    batch["input_ids"] = torch.stack([f["input_ids"] for f in features])
                    batch["attention_mask"] = torch.stack([f["attention_mask"] for f in features])  
                    batch["labels"] = torch.stack([f["labels"] for f in features])
                    batch["rewards"] = torch.tensor(rewards, dtype=torch.float32)  # still called rewards downstream
                    # token-level penalty weights
                    if "token_penalty_weights" in features[0]:
                        batch["token_penalty_weights"] = torch.stack([f["token_penalty_weights"] for f in features])
                
                return batch

        # Initialize data collator
        data_collator = GRPODataCollator(self.tokenizer)

        # Initialize trainer
        trainer = CustomTrainer(
            grpo_trainer_instance=self,
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator  # use the custom data collator
        )
        
        # Train model
        try:
            logger.info(f"Starting training; dataset size: {len(train_dataset)}")
            logger.info(f"Per-device batch: {self.per_device_train_batch_size}, gradient-accum: {self.gradient_accumulation_steps} (effective per-device batch ~= {self.per_device_train_batch_size * self.gradient_accumulation_steps})")
            logger.info(f"Adaptive learning rate: {adaptive_lr:.2e}")
            
            train_result = trainer.train()
            training_loss = train_result.training_loss
            
            logger.info(f"Training finished successfully; final loss: {training_loss:.4f}")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Dataset size: {len(train_dataset) if train_dataset else 'None'}")
            logger.error(f"Model device: {next(self.model.parameters()).device if hasattr(self.model, 'parameters') else 'Unknown'}")
            
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            return {}
        
        # Compute comprehensive statistics
        avg_reward = np.mean([exp["reward"] for exp in selected_exps])
        min_mse = np.min(mse_values)
        max_mse = np.max(mse_values)
        
        stats = {
            "train_loss": training_loss,
            "avg_reward": avg_reward,
            "avg_mse": avg_mse,
            "min_mse": min_mse,
            "max_mse": max_mse,
            "mse_std": mse_std,
            "buffer_size": len(self.training_buffer),
            "used_samples": len(selected_exps),
            "adaptive_lr": adaptive_lr,
            "adaptive_batch_size": adaptive_batch_size,
            "num_epochs": num_epochs,
            "evotune_magic_percentile": _DPO_STYLE_PERCENTILE,
            "evotune_reward_threshold": reward_thr,
            "evotune_filtered_valid": len(filtered),
        }
        # Curve logging
        try:
            self._append_curve_record(stats)
        except Exception:
            pass
        # EvoTune: scores_since_finetune is cleared after prepare_dpo_chats; here we clear after a successful training step
        self.scores_since_finetune = []
        logger.info(
            "EvoTune-style: cleared scores_since_finetune after successful train_step "
            f"(next round accumulates from scratch; training_buffer size={len(self.training_buffer)})"
        )
        return stats
    
    def _compute_physical_penalty(self, text: str, mse: float) -> float:
        """Compute gated physical penalty P_phy (Paper Eq. 9).

        Penalties are applied only when MSE < phy_gate_mse_threshold (gating).
        - Dimensional inconsistency -> +phy_penalty_dim
        - Non-differentiability     -> +phy_penalty_diff
        Satisfying a constraint incurs zero penalty for that constraint.
        """
        if not self.enable_physical_penalty or not text:
            return 0.0
        if mse >= self.phy_gate_mse_threshold:
            return 0.0
        try:
            fns = extract_equation_functions_from_text(text)
            if not fns:
                return 0.0
            analysis = analyze_equation_function(fns[0].source, symbol_units=self.phy_units or {})
            penalty = 0.0
            if not analysis.get("dimensionally_consistent", True):
                penalty += self.phy_penalty_dim
            if not analysis.get("differentiable", True):
                penalty += self.phy_penalty_diff
            return float(penalty)
        except Exception:
            return 0.0

    def should_train(self) -> bool:
        """
        Determine if training should be triggered based on buffer state.
        Start only after warmup: require at least warmup_min_global_samples observed,
        and at least 1 valid experience in buffer.
        """
        # Require at least 1 valid experience
        valid_count = sum(1 for exp in self.training_buffer if self._is_valid_experience(exp))
        if valid_count < 1:
            logger.debug("should_train=False (no valid experiences)")
            return False
        
        # Resolve warmup threshold defensively (handle None or wrong types)
        threshold = getattr(self, "warmup_min_global_samples", None)
        if threshold is None:
            spp = getattr(self, "samples_per_prompt", None)
            try:
                if spp is not None:
                    threshold = int(self.warmup_generations * int(spp) + 1)
                else:
                    threshold = int(self.warmup_generations + 1)
            except Exception:
                threshold = 1  # safest minimal default
            # Cache back to the instance to avoid repeated computation
            self.warmup_min_global_samples = int(threshold)
        else:
            try:
                threshold = int(threshold)
            except Exception:
                threshold = 1
                self.warmup_min_global_samples = threshold
        
        # Determine the maximum global sample index observed so far (from metadata) — robust to None/str/float
        globals_list = []
        for exp in self.training_buffer:
            meta = exp.get("metadata", {}) or {}
            for key in ("global_sample_nums", "global_sample_index"):
                val = meta.get(key, None)
                if val is None:
                    continue
                try:
                    if isinstance(val, (int, float)) and np.isfinite(val):
                        globals_list.append(int(val))
                    elif isinstance(val, str) and val.strip() != "":
                        globals_list.append(int(val.strip()))
                except Exception:
                    # Ignore unparsable values
                    continue
        max_global = max(globals_list) if globals_list else 0
        
        ok = max_global >= threshold
        logger.debug(
            f"should_train={ok} | valid_count={valid_count} | max_global={max_global} | warmup_min_global_samples={threshold}"
        )
        return ok
    
    def force_enable_inference_model(self):
        """Force-mark that callers can immediately swap to the freshly fine-tuned model for sampling."""
        setattr(self, "_force_inference_ready", True)

    def is_inference_ready(self) -> bool:
        return getattr(self, "_force_inference_ready", False) or self.should_train()

    def merge_lora_and_unload(self):
        """If using LoRA, merge the adapter into the base weights and detach the LoRA structure for inference."""
        try:
            from peft import PeftModel
        except Exception:
            return False
        if isinstance(self.model, PeftModel):
            try:
                self.model = self.model.merge_and_unload()
                if hasattr(self.model, 'config'):
                    self.model.config.use_cache = True
                return True
            except Exception:
                return False
        return False

    def save_merged_model(self, save_dir: str) -> bool:
        """Save LoRA adapter, then merge into a base-model copy on CPU and write to *save_dir*.

        The current self.model (with LoRA layers) is NOT modified so training
        can continue afterwards.  The original base model is never overwritten.

        Returns True on success.
        """
        try:
            from peft import PeftModel
        except ImportError:
            logger.warning("peft not installed – cannot save merged model")
            return False

        if not isinstance(self.model, PeftModel):
            logger.info("Model has no LoRA adapters; saving base model directly")
            os.makedirs(save_dir, exist_ok=True)
            self.model.save_pretrained(save_dir)
            self.tokenizer.save_pretrained(save_dir)
            return True

        adapter_dir = os.path.join(save_dir, "_lora_adapter")
        os.makedirs(adapter_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)

        try:
            logger.info(f"Saving LoRA adapter to {adapter_dir} ...")
            self.model.save_pretrained(adapter_dir)
            self.tokenizer.save_pretrained(adapter_dir)

            logger.info("Loading fresh base model on CPU for merge ...")
            base = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map={"": "cpu"},
            )
            merged = PeftModel.from_pretrained(base, adapter_dir)
            merged = merged.merge_and_unload()

            logger.info(f"Saving merged model to {save_dir} ...")
            merged.save_pretrained(save_dir)
            self.tokenizer.save_pretrained(save_dir)

            del merged, base
            torch.cuda.empty_cache()
            logger.info("Merged model saved successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to save merged model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def get_model_for_inference(self):
        model = self.model
        try:
            if hasattr(model, 'config'):
                model.config.use_cache = True
        except Exception:
            pass
        return model

    def set_single_device(self, device_id: str | int = 0):
        """Force the model onto a single GPU (disabling device_map auto offload)."""
        if isinstance(device_id, int):
            device_id = str(device_id)
        try:
            # Only when the model is a transformers model and supports .to()
            model = self.model
            if hasattr(model, 'to'):
                model.to(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
            # Remove any lingering device_map attribute that could interfere
            if hasattr(model, 'hf_device_map'):
                try:
                    delattr(model, 'hf_device_map')
                except Exception:
                    pass
        except Exception:
            pass

    def get_inference_stats(self, prompt_token_count: int, generated_token_count: int, total_time: float) -> dict:
        prefill_tokens = prompt_token_count
        decode_tokens = max(generated_token_count, 1)
        prefill_ms_per_tok = (total_time * 1000.0) / (prefill_tokens + decode_tokens)
        decode_ms_per_tok = (total_time * 1000.0) / decode_tokens
        return {
            'prompt_tokens': prefill_tokens,
            'generated_tokens': decode_tokens,
            'total_time_s': total_time,
            'prefill_ms_per_token_est': round(prefill_ms_per_tok, 3),
            'decode_ms_per_token': round(decode_ms_per_tok, 3),
            'use_cache': getattr(getattr(self.model, 'config', None), 'use_cache', None)
        }

    # ================== Curve & metrics logging ==================
    def set_reward_curves_dir(self, path: str):
        """Configure the output directory for training metrics and curves.
        Creates the directory and initializes the CSV header; also prepares a JSONL for flexible later use.
        Degradation strategy: mark logging disabled if the directory is unwritable."""
        try:
            self.reward_curves_dir = path
            os.makedirs(self.reward_curves_dir, exist_ok=True)
        except Exception as e:
            logger.warning(f"Failed to create reward-curves directory {path}: {e}; curve logging will be disabled")
            self.reward_curves_dir = None
            return

        # CSV file path and JSON Lines path
        self.curve_csv_path = os.path.join(self.reward_curves_dir, "training_curve.csv")
        self.curve_jsonl_path = os.path.join(self.reward_curves_dir, "training_curve.jsonl")
        self.curve_png_path = os.path.join(self.reward_curves_dir, "curves.png")
        # Counters
        self._curve_step = 0

        if not os.path.exists(self.curve_csv_path):
            try:
                with open(self.curve_csv_path, "w", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "step","timestamp","train_loss","avg_reward","avg_mse","min_mse","max_mse","mse_std",
                        "buffer_size","used_samples","adaptive_lr","adaptive_batch_size","num_epochs"
                    ])
            except Exception as e:
                logger.warning(f"Failed to initialize curve CSV file: {e}")
        logger.info(f"Reward curves and metrics will be written to: {self.reward_curves_dir}")

    def _append_curve_record(self, stats: dict):
        """Append a training-statistics record and try to update the curve plots.
        Fields in `stats` come from the train_step summary.
        If matplotlib/pandas are unavailable, only CSV/JSONL are written."""
        if not hasattr(self, 'reward_curves_dir') or not self.reward_curves_dir:
            return  # output directory not configured
        self._curve_step = getattr(self, '_curve_step', 0) + 1
        now_ts = int(time.time())

        row = [
            self._curve_step,
            now_ts,
            stats.get('train_loss',''),
            stats.get('avg_reward',''),
            stats.get('avg_mse',''),
            stats.get('min_mse',''),
            stats.get('max_mse',''),
            stats.get('mse_std',''),
            stats.get('buffer_size',''),
            stats.get('used_samples',''),
            stats.get('adaptive_lr',''),
            stats.get('adaptive_batch_size',''),
            stats.get('num_epochs',''),
        ]
        # Write CSV
        try:
            with open(self.curve_csv_path, 'a', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
        except Exception as e:
            logger.warning(f"Failed to write training-curve CSV: {e}")
        # Write JSONL (easier to post-process later)
        try:
            json_obj = {
                'step': self._curve_step,
                'timestamp': now_ts,
                **stats
            }
            with open(self.curve_jsonl_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.warning(f"Failed to write training-curve JSONL: {e}")

        # Attempt to plot
        self._maybe_plot_curves()

    def _maybe_plot_curves(self):
        """Try to plot reward and MSE curves with pandas+matplotlib.
        Skip if dependencies are missing or there is not enough data."""
        try:
            import pandas as pd
            import matplotlib.pyplot as plt
        except Exception:
            return
        # Read the CSV
        try:
            df = pd.read_csv(self.curve_csv_path)
        except Exception:
            return
        if df.empty:
            return
        # Need at least two records to be meaningful
        if len(df) < 2:
            return
        try:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            # Reward curve
            axes[0].plot(df['step'], df['avg_reward'], label='avg_reward', color='tab:blue')
            axes[0].set_xlabel('step')
            axes[0].set_ylabel('avg_reward')
            axes[0].set_title('Reward Curve')
            axes[0].grid(alpha=0.3)
            # MSE curve (log scale)
            axes[1].plot(df['step'], df['avg_mse'], label='avg_mse', color='tab:red')
            axes[1].set_xlabel('step')
            axes[1].set_ylabel('avg_mse (log)')
            try:
                axes[1].set_yscale('log')
            except Exception:
                pass
            axes[1].set_title('MSE Curve')
            axes[1].grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(self.curve_png_path)
            plt.close(fig)
        except Exception as e:
            logger.debug(f"Plotting curve failed (ignored): {e}")
