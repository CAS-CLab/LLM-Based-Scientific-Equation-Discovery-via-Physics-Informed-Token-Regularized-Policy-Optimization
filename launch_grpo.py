#!/usr/bin/env python3
"""
PiT-PO launcher script.
Runs GRPO-enhanced symbolic regression with continuous learning.
"""

import os
# Avoid the tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import argparse
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import re

# Add the pitpo package directory to the path
sys.path.append('./pitpo')
from pitpo import pipeline, config, sampler, evaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='PiT-PO: GRPO-enhanced symbolic regression (continuous learning)')

    # Basic arguments
    parser.add_argument('--problem', type=str, default="oscillator1",
                       help='Problem name (default: oscillator1)')
    parser.add_argument('--max_samples', type=int, default=3000,
                       help='Maximum number of samples (default: 3000)')
    parser.add_argument('--port', type=int, default=5000,
                       help='LLM server port (default: 5000)')

    # GRPO arguments
    parser.add_argument('--grpo_lr', type=float, default=1e-6,
                       help='GRPO learning rate (default: 1e-6)')
    parser.add_argument('--grpo_batch_size', type=int, default=4,
                       help='GRPO batch size (default: 4)')
    parser.add_argument('--buffer_size', type=int, default=500,
                       help='Experience-buffer size (default: 500)')
    parser.add_argument('--reward_scaling', type=float, default=10.0,
                       help='Reward scaling factor (default: 10.0)')
    parser.add_argument('--training_strategy', type=str, default='continuous',
                       choices=['conservative', 'adaptive', 'aggressive', 'continuous'],
                       help='Training-strategy preset (default: continuous)')
    # Trigger a GRPO fine-tune step every N valid samples
    parser.add_argument('--grpo_train_every', type=int, default=64,
                       help='Trigger one GRPO fine-tune step every N valid outputs (default: 64)')

    # Control arguments
    parser.add_argument('--disable_grpo', action='store_true',
                       help='Disable GRPO training (use only the base model)')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--device_id', type=str, default='0', help='Single GPU device ID used for GRPO fine-tuning (default: 0)')
    parser.add_argument('--vllm_gpu', type=str, default=None,
                       help='vLLM server GPU ID (physical). If set, enables EvoTune-style vLLM restart after training')

    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Build timestamped log directory (anchored at this script's directory so cwd changes don't break paths)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(base_dir, 'logs', f"grpo_{args.problem}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    # Reward-curve output directory (matches the log timestamp)
    reward_curves_dir = os.path.join(base_dir, 'reward_curves', f"grpo_{args.problem}_{timestamp}")
    try:
        os.makedirs(reward_curves_dir, exist_ok=True)
    except Exception as e:
        logger.warning(f"Failed to create reward-curve directory {reward_curves_dir}: {e}")

    # Attach a file handler (best-effort: degrade to console-only on failure)
    try:
        log_file = os.path.join(log_dir, 'training.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)
    except FileNotFoundError:
        # Directory may have been removed by an external process; recreate and retry
        try:
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, 'training.log')
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            logging.getLogger().addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Could not create file log, falling back to console only: {e}")
    except Exception as e:
        logger.warning(f"Could not create file log, falling back to console only: {e}")

    logger.info("="*80)
    logger.info("Starting PiT-PO (GRPO-enhanced symbolic regression)")
    logger.info("="*80)
    logger.info(f"Problem: {args.problem}")
    logger.info(f"Max samples: {args.max_samples}")
    logger.info(f"GRPO enabled: {not args.disable_grpo}")
    logger.info(f"Learning rate: {args.grpo_lr}")
    logger.info(f"Batch size: {args.grpo_batch_size}")
    logger.info(f"Buffer size: {args.buffer_size}")
    logger.info(f"Reward scaling: {args.reward_scaling}")
    logger.info(f"Training strategy: {args.training_strategy}")
    logger.info(f"Train every N valid outputs: {args.grpo_train_every}")
    logger.info(f"Log directory: {log_dir}")
    logger.info(f"Reward-curve directory: {reward_curves_dir}")
    logger.info(f"Device ID (device_id): {args.device_id}")

    try:
        # Locate required files
        # Spec file: if the problem name ends with a digit (e.g. CRK0..CRK35), prefer the exact-match spec;
        # otherwise fall back to the generic spec with the trailing digits stripped.
        base_problem_for_spec = re.sub(r'\d+$', '', args.problem)
        spec_path_exact = f"./specs/specification_{args.problem}_numpy.txt"
        spec_path_base = f"./specs/specification_{base_problem_for_spec}_numpy.txt"
        spec_path = spec_path_exact if os.path.exists(spec_path_exact) else spec_path_base

        data_path = f"./data/{args.problem}/train.csv"

        if not os.path.exists(spec_path):
            raise FileNotFoundError(f"Spec file not found: {spec_path} (tried: {spec_path_exact} and {spec_path_base})")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        logger.info(f"Spec file: {spec_path}")
        logger.info(f"Data file: {data_path}")

        # Prepare config
        class_config = config.ClassConfig(llm_class=sampler.LocalLLM, sandbox_class=evaluator.LocalSandbox)
        config_obj = config.Config(use_api=False, api_model="", llm_server_url=f"http://127.0.0.1:{args.port}/completions")

        # Load spec
        with open(spec_path, encoding="utf-8") as f:
            specification = f.read()

        # Load data
        df = pd.read_csv(data_path)
        data = np.array(df)
        X = data[:, :-1]
        y = data[:, -1].reshape(-1)

        if 'torch' in spec_path:
            X = torch.Tensor(X)
            y = torch.Tensor(y)

        data_dict = {'inputs': X, 'outputs': y}
        dataset = {'data': data_dict}

        logger.info(f"Data shape: inputs {X.shape}, outputs {y.shape}")

        # Prepare config (duplicate-safe rebuild)
        class_config = config.ClassConfig(llm_class=sampler.LocalLLM, sandbox_class=evaluator.LocalSandbox)
        config_obj = config.Config(use_api=False, api_model="", llm_server_url=f"http://127.0.0.1:{args.port}/completions")

        logger.info("Starting main training loop...")

        # Run GRPO-enhanced symbolic regression
        pipeline.main(
            specification=specification,
            inputs=dataset,
            config=config_obj,
            max_sample_nums=args.max_samples,
            class_config=class_config,
            enable_grpo=not args.disable_grpo,
            grpo_model_name="/mnt/finder/wangboxiao/LLM-Research/Meta-Llama-3.1-8B-Instruct/LLM-Research/Meta-Llama-3.1-8B-Instruct",
            grpo_learning_rate=args.grpo_lr,
            grpo_batch_size=args.grpo_batch_size,
            grpo_reward_scaling=args.reward_scaling,
            grpo_min_mse_threshold=1e-15,
            grpo_buffer_size=args.buffer_size,
            training_strategy=args.training_strategy,
            log_dir=log_dir,
            grpo_train_every_n_valid=args.grpo_train_every,
            grpo_device_id=args.device_id,
            reward_curves_dir=reward_curves_dir,
            vllm_gpu=args.vllm_gpu,
            vllm_port=args.port,
        )

    except KeyboardInterrupt:
        logger.info("Interrupt received; shutting down gracefully...")

    except Exception as e:
        logger.error(f"Runtime error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

    finally:
        logger.info("PiT-PO training session finished")
        logger.info("="*80)

if __name__ == "__main__":
    print("PiT-PO: GRPO-enhanced symbolic regression")
    print("- Continuous-learning mode")
    print("- Log-scaled MSE reward")
    print("- Automatic training triggers")
    print("- Integrated local LLM backend")
    print("="*50)
    main()
