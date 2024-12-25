from pymdp import utils, maths
from pymdp.agent import Agent
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import logging
from active_inference.env import ResearchAgentEnv
from active_inference.aif_agent import ResearchAgentController
from pymdp import control

def diagnose_policy_selection(env, controller):
    """
    Diagnostic function to help identify issues with policy selection
    """
    # Reset environment and get initial observation
    observation = env.reset()
    controller.agent.reset()
    
    # Run one step of inference
    qs = controller.agent.infer_states(observation)
    
    # Get policy inference details
    q_pi, G = controller.agent.infer_policies()
    
    print("\nDiagnostic Results:")
    print("==================")
    
    # 1. Check policy values
    print("\n1. Policy Distribution Check:")
    print(f"Number of policies: {len(q_pi)}")
    print(f"Sum of policy probabilities: {q_pi.sum():.6f}")
    print(f"Max policy probability: {q_pi.max():.6f}")
    print(f"Min policy probability: {q_pi.min():.6f}")
    print(f"Std dev of policy probabilities: {q_pi.std():.6f}")
    
    # 2. Check expected free energies
    print("\n2. Expected Free Energy Check:")
    print(f"Mean G: {G.mean():.2f}")
    print(f"Max G: {G.max():.2f}")
    print(f"Min G: {G.min():.2f}")
    print(f"Std dev of G: {G.std():.2f}")
    
    # 3. Check policy construction
    print("\n3. Policy Construction Check:")
    unique_actions = set()
    for policy in controller.agent.policies:
        actions = tuple(policy[0])
        unique_actions.add(actions)
    print(f"Number of unique action combinations: {len(unique_actions)}")
    print(f"Sample of policies:")
    for i, policy in enumerate(controller.agent.policies[:5]):
        print(f"Policy {i}: {policy}")
        
    # 4. Check prior preferences (C)
    print("\n4. Prior Preferences Check:")
    print("C matrix shapes:", [C.shape for C in controller.agent.C])
    print("C matrix ranges:", [(C.min(), C.max()) for C in controller.agent.C])
    
    # 5. Verify likelihood mappings
    print("\n5. Observation Model Check:")
    print("A matrix shapes:", [A.shape for A in controller.agent.A])
    print("Any zero/negative values in A matrices:", 
          any([np.any(A <= 0) for A in controller.agent.A]))
    
    return observation, qs, q_pi, G

def run_experiment(
    num_trials: int,
    timesteps_per_trial: int,
    mock_workflow = None,
    learning_rate_pA: float = 0.1,
    gamma: float = 8.0,
    alpha: float = 16.0
) -> ResearchAgentController:
    """
    Run a complete experiment with multiple trials.
    """
    print("\n=== Starting Experiment ===")
    print(f"Number of trials: {num_trials}")
    print(f"Timesteps per trial: {timesteps_per_trial}")
    print(f"Learning rate: {learning_rate_pA}")
    print(f"Gamma: {gamma}")
    print(f"Alpha: {alpha}")

    print("\nCreating environment...")
    env = ResearchAgentEnv(mock_workflow=mock_workflow)
    print("Environment created successfully!")

    print("\nCreating controller...")
    controller = ResearchAgentController(
        env=env,
        learning_rate_pA=learning_rate_pA,
        gamma=gamma,
        alpha=alpha,
        save_history=True
    )
    print("Controller created successfully!")

    # Run diagnostics before starting trials
    print("\nRunning policy selection diagnostics...")
    obs, qs, q_pi, G = diagnose_policy_selection(env, controller)

    # Run trials
    for trial in range(num_trials):
        logging.info(f"Starting trial {trial + 1}/{num_trials}")
        observations, states, actions = controller.run_trial(timesteps_per_trial)
        
        # Log summary statistics
        mean_scores = {
            metric: np.mean([s for s in scores[-timesteps_per_trial:] if s is not None])
            for metric, scores in controller.quality_scores.items()
        }
        logging.info(f"Trial {trial + 1} mean scores: {mean_scores}")
    
    return controller

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run experiment with modified parameters
    num_trials = 5
    timesteps_per_trial = 20
    
    # Run experiment with higher gamma and alpha to make policy selection more decisive
    controller = run_experiment(
        num_trials=num_trials,
        timesteps_per_trial=timesteps_per_trial,
        learning_rate_pA=0.1,
        gamma=32.0,  # Increased from 8.0
        alpha=32.0   # Increased from 16.0
    )
    
    # Plot results
    controller.plot_quality_history()