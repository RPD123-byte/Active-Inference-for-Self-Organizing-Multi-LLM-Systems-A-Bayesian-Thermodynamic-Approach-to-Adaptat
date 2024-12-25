from pymdp import utils, maths
from pymdp.agent import Agent
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import logging
from active_inference.env_fixed import ResearchAgentEnv
from active_inference.agent_evaluator import AgentEvaluator
from pymdp import control

# Constants for state and modality indices
ACCURACY_FACTOR_ID = 0
COMPREHENSIVENESS_FACTOR_ID = 1
RELEVANCE_FACTOR_ID = 2
INFO_RELEVANCE_FACTOR_ID = 3
INFO_USEFULNESS_FACTOR_ID = 4
SOURCE_QUALITY_FACTOR_ID = 5
INFO_STATE_FACTOR_ID = 6

class ResearchAgentController:
    """
    Controller class that handles the configuration and execution of the active inference agent
    for the research task with quality-based states.
    """
    def __init__(
        self,
        env: ResearchAgentEnv,
        learning_rate_pA: float = 1.0,
        learning_rate_pB: float = 1.0,
        gamma: float = 8.0,
        alpha: float = 16.0,
        save_history: bool = True
    ):
        self.env = env
        self.learning_rate_pA = learning_rate_pA
        self.learning_rate_pB = learning_rate_pB
        self.gamma = gamma
        self.alpha = alpha
        self.save_history = save_history
        
        # Initialize agent
        self.agent = self._initialize_agent()
        
        # History tracking
        if save_history:
            self.observation_history = []
            self.state_history = []
            self.action_history = []
            self.policy_checkpoints = []  
            self.free_energy_checkpoints = []  
            self.prediction_history = [] 
            self.actual_outcome_history = []
            self.quality_scores = {
                'accuracy': [],
                'comprehensiveness': [],
                'relevance': [],
                'info_relevance': [],
                'info_usefulness': [],
                'source_quality': [],
                'info_state': []  # Added to track info state progression
            }

    def _initialize_agent(self) -> Agent:
        """
        Initialize the active inference agent with appropriate parameters
        for the quality-based state space.
        """
        print("\nInitializing Active Inference Agent...")

        # Get likelihood and transition distributions from environment
        A = self.env.get_likelihood_dist()
        B = self.env.get_transition_dist()

        # Initialize prior preferences (C matrix)
        C = self._construct_C_prior()
        
        # Initialize beliefs about initial states (D matrix)
        D = self._construct_D_prior()
        
        # Initialize prior parameters for learning
        pA = self._construct_pA_prior()
        pB = self._construct_pB_prior()

        # Construct valid policies
        # In new model, policies are sequences of actions that can affect any quality state
        valid_policies = self._construct_valid_policies()

        # Factor dependencies remain simple 1:1 mapping in new model
        factor_list = [[f] for f in range(self.env.num_factors)]

        # Initialize the agent
        agent = Agent(
            A=A,
            B=B,
            C=C,
            D=D,
            pA=pA,
            pB=pB,
            gamma=self.gamma,
            alpha=self.alpha,
            policies=valid_policies,
            policy_len=1,
            inference_horizon=2,
            action_selection="deterministic",
            sampling_mode="full",
            inference_algo="VANILLA",
            modalities_to_learn="all",
            lr_pA=self.learning_rate_pA,
            lr_pB=self.learning_rate_pB,
            use_utility=True,
            use_states_info_gain=True,
            use_param_info_gain=True,
            A_factor_list=factor_list,
            B_factor_list=factor_list,
            save_belief_hist=self.save_history
        )
        
        return agent

    def _construct_valid_policies(self):
        """
        Construct valid policies for the agent.
        """
        # Get base policies from control utils
        base_policies = control.construct_policies(
            self.env.num_states,
            self.env.num_controls,
            policy_len=1
        )

        # Filter to include:
        # - No action policy [0, 0, ...]
        # - Single prompt actions
        # - Single search actions
        valid_policies = []
        for policy in base_policies:
            policy_reshaped = policy.reshape(1, -1)
            if (np.all(policy_reshaped == 0) or  # Include [0, 0, ...]
                (1 <= policy_reshaped[0, 0] <= self.env.num_prompt_combinations) or  # Prompt actions
                (self.env.num_prompt_combinations < policy_reshaped[0, 0] < self.env.total_actions)):  # Search actions
                valid_policies.append(policy_reshaped)

        return valid_policies

    def _construct_C_prior(self):
        """
        Construct prior preferences over observations.
        In new model, preferences directly relate to quality levels.
        """
        C = utils.obj_array(self.env.num_modalities)
        
        # For quality-based modalities (0-5)
        for modality in range(6):
            C_m = np.zeros(self.env.num_quality_levels)
            # Preference increases quadratically with quality
            for i in range(self.env.num_quality_levels):
                C_m[i] = (i / (self.env.num_quality_levels - 1)) ** 2 * 4.0
            C[modality] = C_m
            
        # Strong preference for higher info states
        C[INFO_STATE_FACTOR_ID] = np.array([-32.0, 8.0, 64.0])
        
        return C

    def _construct_D_prior(self):
        """
        Construct prior beliefs about initial states.
        Initialize with belief that we start at lowest quality levels.
        """
        D = utils.obj_array(self.env.num_factors)
        
        # For quality states, strong prior belief of starting at lowest level
        for factor in range(6):
            D[factor] = np.zeros(self.env.num_quality_levels)
            D[factor][0] = 0.8  # High probability of lowest quality
            D[factor][1] = 0.2  # Small probability of slightly higher
            
        # For info state, certain of starting with no info
        D[INFO_STATE_FACTOR_ID] = np.zeros(self.env.num_info_states)
        D[INFO_STATE_FACTOR_ID][0] = 1.0
        
        return D
    
    def _construct_pA_prior(self):
        """
        Construct prior counts for learning the observation model.
        In new model, each modality directly observes its corresponding state.
        """
        pA = utils.obj_array(self.env.num_modalities)
        
        # For quality-based modalities (0-5)
        for modality in range(6):
            # Create matrix of prior counts
            pA[modality] = np.ones((self.env.num_quality_levels, 
                                  self.env.num_quality_levels)) * 1.0
            
            # Slightly higher prior for accurate observations
            np.fill_diagonal(pA[modality], 2.0)
            
        # For info state modality
        pA[INFO_STATE_FACTOR_ID] = np.ones((self.env.num_info_states,
                                           self.env.num_info_states)) * 1.0
        np.fill_diagonal(pA[INFO_STATE_FACTOR_ID], 2.0)
        
        return pA

    def _construct_pB_prior(self):
        """
        Construct prior concentration parameters for learning the transition model.
        """
        pB = utils.obj_array(self.env.num_factors)
        base_concentration = 1.0
        
        # For quality states (0-5)
        for factor in range(6):
            # Shape: (next_state, current_state, action)
            pB[factor] = np.ones((self.env.num_quality_levels,
                                self.env.num_quality_levels,
                                self.env.total_actions)) * base_concentration
            
            # Add slight bias for:
            # - Staying in same state when no action taken
            # - Gradual improvement for valid actions
            # - Quality decay over time
            
            # No action bias
            np.fill_diagonal(pB[factor][:, :, 0], 2.0)
            
            # Action-specific biases
            if factor < 3:  # Prompt-affected qualities
                for action in range(1, self.env.num_prompt_combinations + 1):
                    target_quality = min(0.9, 0.3 + (action / self.env.num_prompt_combinations) * 0.6)
                    target_idx = int(target_quality * (self.env.num_quality_levels - 1))
                    pB[factor][target_idx, :, action] += 1.0
                    
            elif factor < 6:  # Search-affected qualities
                for action in range(self.env.num_prompt_combinations + 1, self.env.total_actions):
                    search_idx = action - self.env.num_prompt_combinations - 1
                    base_quality = 0.5 + (search_idx / self.env.num_search_terms) * 0.4
                    target_idx = int(base_quality * (self.env.num_quality_levels - 1))
                    pB[factor][target_idx, :, action] += 1.0
        
        # For info state transitions
        pB[INFO_STATE_FACTOR_ID] = np.ones((self.env.num_info_states,
                                          self.env.num_info_states,
                                          self.env.total_actions)) * base_concentration
                                          
        # Add bias for forward progression in info states
        for action in range(1, self.env.total_actions):
            for state in range(self.env.num_info_states - 1):
                pB[INFO_STATE_FACTOR_ID][state + 1, state, action] += 1.0
                
        return pB

    def run_trial(
        self, 
        num_timesteps: int, 
        initial_observation: Optional[List[int]] = None
    ) -> Tuple[List[List[int]], List[List[float]], List[List[int]], bool]:
        """
        Run a single trial of the agent-environment loop.
        """
        print(f"\n=== Starting New Trial (Timesteps: {num_timesteps}) ===")

        # Reset environment and get initial observation
        if initial_observation is None:
            observation = self.env.reset()
        else:
            observation = initial_observation
            
        # Initialize storage
        observations = [observation]
        states = []
        actions = []
        qs_prev = None
        
        for t in range(num_timesteps):
            print(f"\n--- Timestep {t+1}/{num_timesteps} ---")
            
            # 1. Update beliefs about hidden states
            qs = self.agent.infer_states(observation)
            states.append(qs)
            
            # 2. Update beliefs about policies and get expected free energy
            q_pi, G = self.agent.infer_policies()
            
            if self.save_history:
                self.policy_checkpoints.append(self.agent.policies)
                self.free_energy_checkpoints.append(G)
                
            # Print top policies
            indices = np.argsort(q_pi)[-5:]
            print("\nTop 5 policies by probability:")
            for idx in indices:
                print(f"Policy {idx}: prob={q_pi[idx]:.6f}, G={G[idx]:.2f}")
                print(f"Policy actions: {self.agent.policies[idx]}")

            # 3. Sample action
            action = self.agent.sample_action()
            actions.append(action)

            # Get expected observations for selected action
            selected_policy = np.array([action]).reshape(1, -1)
            expected_states = control.get_expected_states_interactions(
                qs, 
                self.agent.B,
                self.agent.B_factor_list,
                selected_policy
            )
            
            expected_obs = control.get_expected_obs_factorized(
                expected_states,
                self.agent.A,
                self.agent.A_factor_list
            )

            if self.save_history:
                probs = [
                    obs[0] if isinstance(obs, np.ndarray) else None 
                    for obs in expected_obs[0]
                ]
                self.prediction_history.append(probs)

            # Check for [0 0 0 ...] action
            if np.all(action == 0):
                print("\nAgent selected no-action - terminating trial early")
                return observations, states, actions, True

            print(f"\nSelected action: {action}")

            # 4. Step environment
            observation = self.env.step_test(action)
            observations.append(observation)

            if self.save_history:
                self.actual_outcome_history.append(observation)
                self._update_history(observation, qs, action)

            # 5. Learn parameters
            if hasattr(self.agent, 'pA'):
                self.agent.update_A(observation)

            if hasattr(self.agent, 'pB') and qs_prev is not None:
                self.agent.update_B(qs_prev)
                
            qs_prev = qs

        return observations, states, actions, False
    
    def _update_history(
        self,
        observation: List[int],
        states: List[np.ndarray],
        action: List[int]
    ) -> None:
        """
        Update history tracking with latest data.
        """
        # Update quality scores for all metrics
        metrics = [
            ('accuracy', 0),
            ('comprehensiveness', 1),
            ('relevance', 2),
            ('info_relevance', 3),
            ('info_usefulness', 4),
            ('source_quality', 5),
            ('info_state', 6)
        ]
        
        for metric_name, modality_id in metrics:
            if observation[modality_id] is not None:
                if modality_id == INFO_STATE_FACTOR_ID:
                    score = observation[modality_id]  # Keep as integer for info state
                else:
                    score = observation[modality_id] / 10.0  # Convert to 0.0-1.0 scale
                self.quality_scores[metric_name].append(score)
            else:
                self.quality_scores[metric_name].append(None)
            
        # Update other history
        self.observation_history.append(observation)
        self.state_history.append(states)
        self.action_history.append(action)

        # Print current scores
        try:
            print("\nCurrent quality scores:")
            for metric, score in self.quality_scores.items():
                if score[-1] is not None:
                    if metric == 'info_state':
                        print(f"{metric}: state {score[-1]}")
                    else:
                        print(f"{metric}: {score[-1]:.2f}")
        except IndexError:
            print("Warning: No quality scores available yet")

    def plot_quality_history(self) -> None:
        """
        Plot the history of quality scores across all metrics.
        """
        if not self.save_history:
            raise ValueError("History tracking is disabled")
            
        plt.figure(figsize=(12, 8))
        
        # Plot quality metrics
        for metric_name in self.quality_scores:
            if metric_name != 'info_state':  # Plot quality metrics normally
                scores = [s for s in self.quality_scores[metric_name] if s is not None]
                if scores:
                    plt.plot(scores, label=metric_name)
                    
        # Plot info state separately with different y-axis
        info_states = [s for s in self.quality_scores['info_state'] if s is not None]
        if info_states:
            ax2 = plt.gca().twinx()
            ax2.plot(info_states, label='info_state', color='black', linestyle='--')
            ax2.set_ylabel('Info State')
            ax2.set_ylim(-0.5, 2.5)
            
        plt.title('Quality Metrics Over Time')
        plt.xlabel('Timestep')
        plt.ylabel('Quality Score')
        plt.legend()
        plt.grid(True)
        plt.show()

def run_experiment(
    num_trials: int,
    timesteps_per_trial: int,
    mock_workflow = None,
    learning_rate_pA: float = 1.0,
    learning_rate_pB: float = 1.0,
    gamma: float = 8.0,
    alpha: float = 16.0
) -> ResearchAgentController:
    """
    Run a complete experiment with multiple trials.
    
    Parameters:
    -----------
    num_trials: int
        Number of trials to run
    timesteps_per_trial: int
        Number of timesteps per trial
    mock_workflow: optional
        Mock workflow for the research agent
    learning_rate_pA: float
        Learning rate for updating the observation model
    learning_rate_pB: float
        Learning rate for updating the transition model
    gamma: float
        Policy precision parameter
    alpha: float
        Action selection precision parameter
        
    Returns:
    --------
    controller: ResearchAgentController
        The controller object containing experiment results
    """
    print("\n=== Starting Experiment ===")
    print(f"Number of trials: {num_trials}")
    print(f"Timesteps per trial: {timesteps_per_trial}")
    print(f"Learning rate pA: {learning_rate_pA}")
    print(f"Learning rate pB: {learning_rate_pB}")
    print(f"Gamma: {gamma}")
    print(f"Alpha: {alpha}")

    print("\nCreating environment...")
    env = ResearchAgentEnv(mock_workflow)
    print("Environment created successfully!")

    print("\nInitializing evaluator...")
    evaluator = AgentEvaluator(save_dir="figures")
    print("Evaluator initialized successfully!")

    print("\nCreating controller...")
    controller = ResearchAgentController(
        env=env,
        learning_rate_pA=learning_rate_pA,
        learning_rate_pB=learning_rate_pB,
        gamma=gamma,
        alpha=alpha,
        save_history=True
    )
    print("Controller created successfully!")
    
    # Store initial model parameters
    initial_A = controller.agent.A.copy()
    initial_B = controller.agent.B.copy()

    completed_trials = 0
    
    # Run trials
    for trial in range(num_trials):
        logging.info(f"Starting trial {trial + 1}/{num_trials}")
        
        observations, states, actions, terminated_early = controller.run_trial(timesteps_per_trial)
        completed_trials += 1

        # Log trial statistics
        mean_scores = {}
        for metric, scores in controller.quality_scores.items():
            valid_scores = [s for s in scores[-timesteps_per_trial:] if s is not None]
            if valid_scores:
                if metric == 'info_state':
                    mean_scores[metric] = f"state {max(valid_scores)}"
                else:
                    mean_scores[metric] = f"{np.mean(valid_scores):.2f}"
                    
        logging.info(f"Trial {trial + 1} mean scores: {mean_scores}")

        if terminated_early:
            logging.info(f"Trial {trial + 1} terminated early due to no-action selection")
            break
        else:
            logging.info(f"Trial {trial + 1} completed full timesteps")
    
    # Store final model parameters
    final_A = controller.agent.A.copy()
    final_B = controller.agent.B.copy()

    # Visualize results
    evaluator.visualize_model_evolution(initial_A, final_A, initial_B, final_B)
    evaluator.visualize_policy_progression(
        controller.policy_checkpoints,
        controller.free_energy_checkpoints,
        completed_trials
    )

    return controller

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Example usage
    num_trials = 20
    timesteps_per_trial = 4
    
    # Run experiment
    controller = run_experiment(
        num_trials=num_trials,
        timesteps_per_trial=timesteps_per_trial,
        gamma=32.0,
        alpha=16.0
    )