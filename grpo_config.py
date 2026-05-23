"""
GRPO training strategy configuration.
Defines different training frequencies and strategies.
"""

# Training-frequency strategies
TRAINING_STRATEGIES = {
    'conservative': {
        'description': 'Conservative: fewer training triggers, more stable',
        'evaluator_trigger_intervals': {
            'very_good_mse': 30,    # train every 30 samples when MSE < 0.1
            'good_mse': 50,         # train every 50 samples when MSE < 1.0
            'regular_mse': 100      # train every 100 samples otherwise
        },
        'buffer_conditions': {
            'diversity_threshold': 0.5,     # coefficient-of-variation threshold
            'good_performance_threshold': 1.0,  # MSE threshold for "good"
            'min_buffer_ratio': 0.6         # minimum buffer fill ratio
        },
        'adaptive_params': {
            'lr_boost_factor': 1.2,         # learning-rate boost on good results
            'lr_reduce_factor': 0.8,        # learning-rate reduction on bad results
            'epoch_boost_good': 1,          # extra epochs on good results
            'epoch_boost_excellent': 2      # extra epochs on excellent results
        }
    },

    'adaptive': {
        'description': 'Adaptive: dynamically adjusts training frequency based on performance',
        'evaluator_trigger_intervals': {
            'very_good_mse': 20,    # train every 20 samples when MSE < 0.1
            'good_mse': 30,         # train every 30 samples when MSE < 1.0
            'regular_mse': 50       # train every 50 samples otherwise
        },
        'buffer_conditions': {
            'diversity_threshold': 0.3,     # lower coefficient-of-variation threshold
            'good_performance_threshold': 1.0,
            'min_buffer_ratio': 0.4         # lower buffer fill ratio
        },
        'adaptive_params': {
            'lr_boost_factor': 1.5,
            'lr_reduce_factor': 0.6,
            'epoch_boost_good': 2,
            'epoch_boost_excellent': 3
        }
    },

    'aggressive': {
        'description': 'Aggressive: frequent training for fast adaptation',
        'evaluator_trigger_intervals': {
            'very_good_mse': 10,    # train every 10 samples when MSE < 0.1
            'good_mse': 20,         # train every 20 samples when MSE < 1.0
            'regular_mse': 30       # train every 30 samples otherwise
        },
        'buffer_conditions': {
            'diversity_threshold': 0.2,     # very low coefficient-of-variation threshold
            'good_performance_threshold': 2.0,  # looser threshold for "good"
            'min_buffer_ratio': 0.25        # very low buffer fill ratio
        },
        'adaptive_params': {
            'lr_boost_factor': 2.0,
            'lr_reduce_factor': 0.5,
            'epoch_boost_good': 3,
            'epoch_boost_excellent': 4
        }
    },

    'continuous': {
        'description': 'Continuous: trains on every evaluation, maximizing learning frequency',
        'evaluator_trigger_intervals': {
            'very_good_mse': 1,     # train on every sample when MSE < 0.1
            'good_mse': 1,          # train on every sample when MSE < 1.0
            'regular_mse': 1        # train on every sample otherwise
        },
        'buffer_conditions': {
            'diversity_threshold': 0.1,     # extremely low coefficient-of-variation threshold
            'good_performance_threshold': 5.0,  # very loose "good" threshold
            'min_buffer_ratio': 0.1         # extremely low buffer fill ratio
        },
        'adaptive_params': {
            'lr_boost_factor': 1.8,
            'lr_reduce_factor': 0.7,
            'epoch_boost_good': 2,
            'epoch_boost_excellent': 3
        }
    }
}

# Model-switching strategies
MODEL_SWITCH_STRATEGIES = {
    'conservative': {
        'switch_after_samples_ratio': 0.4,      # switch after 40% of samples
        'min_training_episodes': 5,             # require at least 5 training episodes
        'performance_threshold': 0.5,           # switch only if average MSE < 0.5
    },
    'adaptive': {
        'switch_after_samples_ratio': 0.3,      # switch after 30% of samples
        'min_training_episodes': 3,             # require at least 3 training episodes
        'performance_threshold': 1.0,           # switch only if average MSE < 1.0
    },
    'aggressive': {
        'switch_after_samples_ratio': 0.2,      # switch after 20% of samples
        'min_training_episodes': 2,             # require at least 2 training episodes
        'performance_threshold': 2.0,           # switch only if average MSE < 2.0
    },
    'continuous': {
        'switch_after_samples_ratio': 0.1,      # may switch after just 10% of samples
        'min_training_episodes': 1,             # require at least 1 training episode
        'performance_threshold': 5.0,           # switch only if average MSE < 5.0 (loose)
    }
}

def get_strategy_config(strategy_name: str):
    """
    Get the configuration for the named strategy.

    Args:
        strategy_name: one of 'conservative', 'adaptive', 'aggressive', 'continuous'

    Returns:
        Dictionary with training and model-switch configs.
    """
    if strategy_name not in TRAINING_STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(TRAINING_STRATEGIES.keys())}")

    return {
        'training': TRAINING_STRATEGIES[strategy_name],
        'model_switch': MODEL_SWITCH_STRATEGIES[strategy_name]
    }
