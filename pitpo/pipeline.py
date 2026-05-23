from __future__ import annotations

import os
# Avoid the tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# from collections.abc import Sequence
from typing import Any, Tuple, Sequence
import time
import numpy as np

from pitpo import code_manipulation
from pitpo import config as config_lib
from pitpo import evaluator
from pitpo import buffer
from pitpo import sampler
from pitpo import profile
from pitpo import grpo_trainer


def _extract_function_names(specification: str) -> Tuple[str, str]:
    """ Return the name of the function to evolve and of the function to run.

    The so-called specification refers to the boilerplate code template for a task.
    The template MUST have two important functions decorated with '@evaluate.run', '@equation.evolve' respectively.
    The function labeled with '@evaluate.run' is going to evaluate the generated code (like data-diven fitness evaluation).
    The function labeled with '@equation.evolve' is the function to be searched (like 'equation' structure).
    """
    run_functions = list(code_manipulation.yield_decorated(specification, 'evaluate', 'run'))
    evolve_functions = list(code_manipulation.yield_decorated(specification, 'equation', 'evolve'))
    
    if len(evolve_functions) != 1:
        raise ValueError('Expected 1 function decorated with `@equation.evolve`.')
    
    # If the spec does not define evaluate.run, return the placeholder name 'evaluate'; main() will inject a default impl
    function_to_run = run_functions[0] if len(run_functions) == 1 else 'evaluate'
    return evolve_functions[0], function_to_run



def main(
        specification: str,
        inputs: Sequence[Any],
        config: config_lib.Config,
        max_sample_nums: int | None,
        class_config: config_lib.ClassConfig,
        enable_grpo: bool = False,
        grpo_model_name: str = "/mnt/finder/wangboxiao/LLM-Research/Meta-Llama-3.1-8B-Instruct/LLM-Research/Meta-Llama-3.1-8B-Instruct",
        grpo_learning_rate: float = 1e-5,
        grpo_batch_size: int = 2,
        grpo_reward_scaling: float = 10.0,
        grpo_min_mse_threshold: float = 1e-15,
        grpo_buffer_size: int = 1000,
        training_strategy: str = 'adaptive',
        grpo_train_every_n_valid: int | None = None,
        grpo_device_id: str | None = None,
        vllm_gpu: str | None = None,
        vllm_port: int | None = None,
        **kwargs
):
    """ Launch a PiT-PO experiment.
    Args:
        specification: the boilerplate code for the problem.
        inputs       : the data instances for the problem.
        config       : config file.
        max_sample_nums: the maximum samples nums from LLM. 'None' refers to no stop.
        class_config: Configuration for LLM and sandbox classes.
        enable_grpo  : Whether to enable GRPO fine-tuning.
        grpo_model_name: Local path to the model for GRPO training.
        grpo_learning_rate: Learning rate for GRPO training.
        grpo_batch_size: Batch size for GRPO training.
        grpo_reward_scaling: Reward scaling factor for log transformation.
        grpo_min_mse_threshold: Minimum MSE threshold to prevent log(0).
    """
    # If the specification lacks @evaluate.run, inject a default implementation
    run_functions_check = list(code_manipulation.yield_decorated(specification, 'evaluate', 'run'))
    if len(run_functions_check) != 1:
        default_eval = r'''
@evaluate.run
def evaluate(data: dict) -> float:
    """Evaluate the equation on data observations."""
    import numpy as _np
    from scipy.optimize import minimize as _minimize

    inputs, outputs = data['inputs'], data['outputs']
    X = inputs

    def _loss(params):
        y_pred = equation(*X.T, params)
        return _np.mean((y_pred - outputs) ** 2)

    result = _minimize(_loss, [1.0]*10, method='BFGS')
    optimized_params = getattr(result, 'x', None)
    loss_val = float(getattr(result, 'fun', _np.inf))

    if not _np.isfinite(loss_val):
        return None
    return -loss_val
'''
        specification = specification + "\n\n" + default_eval
    
    function_to_evolve, function_to_run = _extract_function_names(specification)
    template = code_manipulation.text_to_program(specification)
    database = buffer.ExperienceBuffer(config.experience_buffer, template, function_to_evolve)

    # get log_dir and create profiler
    log_dir = kwargs.get('log_dir', None)
    if log_dir is None:
        profiler = None
    else:
        profiler = profile.Profiler(log_dir)
    # Reward-curve directory
    reward_curves_dir = kwargs.get('reward_curves_dir', None)

    # Initialize GRPO trainer if enabled
    grpo_trainer_instance = None
    if enable_grpo:
        print(f"Initializing GRPO trainer with model: {grpo_model_name}")
        # Pass the device ID through (two channels: environment variable + instance method)
        if grpo_device_id is not None:
            os.environ['LLMSR_DEVICE_ID'] = str(grpo_device_id)
        grpo_trainer_instance = grpo_trainer.GRPOTrainer(
            model_name=grpo_model_name,
            learning_rate=grpo_learning_rate,
            batch_size=grpo_batch_size,
            reward_scaling=grpo_reward_scaling,
            min_mse_threshold=grpo_min_mse_threshold,
            buffer_size=grpo_buffer_size,
        )
        # Pass the reward-curve directory
        if reward_curves_dir is not None and hasattr(grpo_trainer_instance, 'set_reward_curves_dir'):
            try:
                grpo_trainer_instance.set_reward_curves_dir(reward_curves_dir)
            except Exception:
                pass
        # Reassert that the model lives on the specified device
        if grpo_device_id is not None and hasattr(grpo_trainer_instance, 'set_single_device'):
            grpo_trainer_instance.set_single_device(grpo_device_id)
        print("GRPO trainer initialized successfully")

    evaluators = []
    for _ in range(config.num_evaluators):
        evaluators.append(evaluator.Evaluator(
            database,
            template,
            function_to_evolve,
            function_to_run,
            inputs,
            timeout_seconds=config.evaluate_timeout_seconds,
            sandbox_class=class_config.sandbox_class,
            grpo_trainer=grpo_trainer_instance,
            enable_grpo=enable_grpo,
            training_strategy=training_strategy
        ))


    # Configure per-evaluator training cadence
    try:
        if enable_grpo and grpo_trainer_instance is not None and grpo_train_every_n_valid:
            for ev in evaluators:
                if hasattr(ev, 'set_train_every_n_valid'):
                    ev.set_train_every_n_valid(int(grpo_train_every_n_valid))
    except Exception:
        pass

    initial = template.get_function(function_to_evolve).body
    evaluators[0].analyse(initial, island_id=None, version_generated=None, profiler=profiler)

    # Determine which LLM class to use
    if enable_grpo and grpo_trainer_instance is not None:
        llm_class = sampler.GRPOLocalLLM
        print("Using GRPO-enhanced LocalLLM")
    else:
        llm_class = class_config.llm_class
        print(f"Using standard LLM: {llm_class.__name__}")

    # EvoTune-style: compute merged_model_dir
    merged_model_dir = None
    if enable_grpo and log_dir:
        merged_model_dir = os.path.join(log_dir, 'merged_model')

    # Build samplers
    samplers_list = []
    for _ in range(config.num_samplers):
        sampler_instance = sampler.Sampler(
            database, evaluators,
            config.samples_per_prompt,
            max_sample_nums=max_sample_nums,
            llm_class=llm_class,
            config=config
        )
        if enable_grpo and grpo_trainer_instance is not None:
            if hasattr(sampler_instance._llm, 'set_grpo_trainer'):
                sampler_instance._llm.set_grpo_trainer(grpo_trainer_instance)
        samplers_list.append(sampler_instance)

    # ---- EvoTune-style: Python launches vLLM and owns the PID ----
    llm_inst = samplers_list[0]._llm if samplers_list else None
    vllm_managed = False
    if enable_grpo and vllm_gpu is not None and vllm_port is not None and llm_inst is not None:
        print(f"[vLLM] Python will manage vLLM lifecycle: GPU={vllm_gpu} port={vllm_port}")
        ok = llm_inst.start_vllm_server(grpo_model_name, gpu=vllm_gpu, port=vllm_port)
        if not ok:
            raise RuntimeError("[vLLM] Failed to start vLLM server — aborting")
        vllm_managed = True

        # Wire merged_model_dir and LLM instance into evaluators
        for ev in evaluators:
            ev._merged_model_dir = merged_model_dir
            ev._llm_instance = llm_inst

    try:
        for i, s in enumerate(samplers_list):
            s.sample(profiler=profiler)
    finally:
        # Cleanup: kill vLLM on exit
        if vllm_managed and llm_inst is not None:
            print("[vLLM] Cleaning up vLLM server...")
            llm_inst.cleanup_vllm()

    if profiler:
        try:
            profiler.write_time_summary()
        except Exception:
            pass
