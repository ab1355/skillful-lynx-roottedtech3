import numpy as np
from typing import Dict, List, Any

def add_noise(value: np.ndarray, epsilon: float = 0.1) -> np.ndarray:
    """
    Add Laplace noise to the value for differential privacy.
    """
    sensitivity = np.abs(value).max()  # L1 sensitivity
    noise = np.random.laplace(0, sensitivity / epsilon, value.shape)
    return value + noise

def secure_aggregate(parameters_list: List[Dict[str, Any]], epsilon: float = 0.1) -> Dict[str, Any]:
    """
    Securely aggregate parameters from multiple agents using differential privacy.
    """
    aggregated_parameters = {}
    n_agents = len(parameters_list)
    
    # Use the first agent's parameters as a base
    for key, value in parameters_list[0].items():
        if isinstance(value, np.ndarray):
            # If the value is a numpy array, try to aggregate across all agents
            try:
                stacked_params = np.stack([params[key] for params in parameters_list])
                avg_param = np.mean(stacked_params, axis=0)
                noisy_avg_param = add_noise(avg_param, epsilon * np.sqrt(n_agents))
                aggregated_parameters[key] = noisy_avg_param
            except:
                # If stacking fails, just use the first agent's value
                aggregated_parameters[key] = value
        else:
            # For non-numpy array parameters, just keep the first agent's value
            aggregated_parameters[key] = value
    
    return aggregated_parameters