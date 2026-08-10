#!/usr/bin/env python3
"""PiT-PO-for-1-GPU launcher.

Training, vLLM generation, and in-memory LoRA synchronization are colocated on
one physical GPU. No external model server is required.
"""

import os
# Avoid the tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import argparse
import logging
from datetime import datetime
import re
import signal

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description=(
            'PiT-PO-for-1-GPU: colocated search and online GRPO fine-tuning'
        )
    )

    # Basic arguments
    parser.add_argument('--problem', type=str, default="oscillator1",
                       help='Problem name (default: oscillator1)')
    parser.add_argument('--max_samples', type=int, default=3000,
                       help='Maximum number of samples (default: 3000)')
    parser.add_argument('--gpu', type=str, default='0',
                       help='Physical GPU used by the colocated runtime (default: 0)')
    parser.add_argument(
        '--model_path',
        type=str,
        default='/data/home/zdhs0037/zdhs0037_data/meta-llama/Llama-3.1-8B-Instruct/LLM-Research/Meta-Llama-3___1-8B-Instruct',
        help='Local or Hugging Face model path',
    )
    parser.add_argument(
        '--max_seq_length', type=int, default=4096,
        help='Shared model context length (default: 4096)',
    )
    parser.add_argument(
        '--samples_per_prompt', type=int, default=4,
        help='Number of candidates generated together for each prompt (default: 4)',
    )
    parser.add_argument(
        '--max_new_tokens', type=int, default=1024,
        help='Maximum generated tokens per candidate (default: 1024)',
    )
    parser.add_argument(
        '--load_in_4bit',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Load the shared model in 4-bit mode (default: enabled)',
    )
    parser.add_argument(
        '--fast_inference',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Enable the CALM-style Unsloth fast-inference runtime',
    )
    parser.add_argument(
        '--vllm_gpu_memory_utilization', type=float, default=0.6,
        help='Fraction reserved by Unsloth fast inference (default: 0.6)',
    )

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
    parser.add_argument('--checkpoint_every', type=int, default=100,
                       help='Save a recovery LoRA adapter every N model updates (default: 100)')
    parser.add_argument('--export_merged_model', action='store_true',
                       help='Export merged weights once, after the search finishes')

    # Control arguments
    parser.add_argument('--disable_grpo', action='store_true',
                       help='Disable GRPO training (use only the base model)')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    args = parser.parse_args()

    if args.max_samples <= 0:
        parser.error('--max_samples must be positive')
    if args.samples_per_prompt <= 0:
        parser.error('--samples_per_prompt must be positive')
    if args.max_new_tokens <= 0:
        parser.error('--max_new_tokens must be positive')
    if args.max_new_tokens >= args.max_seq_length:
        parser.error('--max_new_tokens must be smaller than --max_seq_length')
    if not 0.0 < args.vllm_gpu_memory_utilization < 1.0:
        parser.error('--vllm_gpu_memory_utilization must be between 0 and 1')

    def _handle_termination(signum, _frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGTERM, _handle_termination)

    # Restrict visibility before importing torch/Unsloth and loading any model.
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    base_dir = os.path.dirname(os.path.abspath(__file__))
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)

    import numpy as np
    import pandas as pd
    import torch

    from pitpo import pipeline, config, sampler, evaluator

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Build timestamped log directory (anchored at this script's directory so cwd changes don't break paths)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
    logger.info("Starting PiT-PO-for-1-GPU")
    logger.info("="*80)
    logger.info(f"Problem: {args.problem}")
    logger.info(f"Max samples: {args.max_samples}")
    logger.info(f"Samples per prompt: {args.samples_per_prompt}")
    logger.info(f"Max new tokens: {args.max_new_tokens}")
    logger.info(f"GRPO enabled: {not args.disable_grpo}")
    logger.info(f"Learning rate: {args.grpo_lr}")
    logger.info(f"Batch size: {args.grpo_batch_size}")
    logger.info(f"Buffer size: {args.buffer_size}")
    logger.info(f"Reward scaling: {args.reward_scaling}")
    logger.info(f"Training strategy: {args.training_strategy}")
    logger.info(f"Train every N valid outputs: {args.grpo_train_every}")
    logger.info(f"Shared-runtime GPU: {args.gpu}")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"4-bit loading: {args.load_in_4bit}")
    logger.info(f"Fast inference: {args.fast_inference}")
    logger.info(f"vLLM memory utilization: {args.vllm_gpu_memory_utilization}")
    logger.info(f"Log directory: {log_dir}")
    logger.info(f"Reward-curve directory: {reward_curves_dir}")

    try:
        # Locate required files
        # Spec file: if the problem name ends with a digit (e.g. CRK0..CRK35), prefer the exact-match spec;
        # otherwise fall back to the generic spec with the trailing digits stripped.
        base_problem_for_spec = re.sub(r'\d+$', '', args.problem)
        spec_path_exact = os.path.join(
            base_dir, 'specs', f"specification_{args.problem}_numpy.txt"
        )
        spec_path_base = os.path.join(
            base_dir, 'specs', f"specification_{base_problem_for_spec}_numpy.txt"
        )
        spec_path = spec_path_exact if os.path.exists(spec_path_exact) else spec_path_base

        data_path = os.path.join(base_dir, 'data', args.problem, 'train.csv')

        if not os.path.exists(spec_path):
            raise FileNotFoundError(f"Spec file not found: {spec_path} (tried: {spec_path_exact} and {spec_path_base})")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        logger.info(f"Spec file: {spec_path}")
        logger.info(f"Data file: {data_path}")

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

        class_config = config.ClassConfig(llm_class=sampler.LocalLLM, sandbox_class=evaluator.LocalSandbox)
        config_obj = config.Config(
            use_api=False,
            api_model="",
            samples_per_prompt=args.samples_per_prompt,
            max_new_tokens=args.max_new_tokens,
        )

        logger.info("Starting main training loop...")

        # Run GRPO-enhanced symbolic regression
        pipeline.main(
            specification=specification,
            inputs=dataset,
            config=config_obj,
            max_sample_nums=args.max_samples,
            class_config=class_config,
            enable_grpo=not args.disable_grpo,
            grpo_model_name=args.model_path,
            grpo_learning_rate=args.grpo_lr,
            grpo_batch_size=args.grpo_batch_size,
            grpo_reward_scaling=args.reward_scaling,
            grpo_min_mse_threshold=1e-15,
            grpo_buffer_size=args.buffer_size,
            training_strategy=args.training_strategy,
            log_dir=log_dir,
            grpo_train_every_n_valid=args.grpo_train_every,
            reward_curves_dir=reward_curves_dir,
            grpo_max_seq_length=args.max_seq_length,
            grpo_load_in_4bit=args.load_in_4bit,
            grpo_fast_inference=args.fast_inference,
            grpo_vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            checkpoint_every=args.checkpoint_every,
            export_merged_model=args.export_merged_model,
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
    print("PiT-PO-for-1-GPU")
    print("- One colocated training and vLLM inference runtime")
    print("- In-memory LoRA synchronization after online GRPO updates")
    print("- No model-server deployment or restart")
    print("="*50)
    main()
