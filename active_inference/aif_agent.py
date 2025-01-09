from pymdp import utils, maths
from pymdp.agent import Agent
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import logging
from active_inference.env import ResearchAgentEnv
from active_inference.agent_evaluator import AgentEvaluator
from pymdp import control

# Constants for state and modality indices (same as in environment)
PROMPT_FACTOR_ID = 0
SEARCH_FACTOR_ID = 1
INFO_FACTOR_ID = 2

ACCURACY_MODALITY_ID = 0
RELEVANCE_MODALITY_ID = 1
COMPREHENSIVENESS_MODALITY_ID = 2
INFO_RELEVANCE_MODALITY_ID = 3
INFO_USEFULNESS_MODALITY_ID = 4
SOURCE_QUALITY_MODALITY_ID = 5
INFO_STATE_MODALITY_ID = 6


class ResearchAgentController:
    """
    Controller class that handles the configuration and execution of the active inference agent
    for the research task.
    """
    def __init__(
        self,
        env: ResearchAgentEnv,
        learning_rate_pA: float = 50,
        learning_rate_pB: float = 50,
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
                'relevance': [],
                'comprehensiveness': [],
                'info_relevance': [],
                'info_usefulness': [],
                'source_quality': []
            }

    def _initialize_agent(self) -> Agent:
        """
        Initialize the active inference agent with appropriate parameters.
        """
        print("\nInitializing Active Inference Agent...")

        # Get likelihood and transition distributions from environment
        A = self.env.get_likelihood_dist()

        # print("\nA matrix samples for search vs no search:")
        # print("No search (state 0) likelihood for info relevance:")
        # print(A[3][:, 0])  # Just the observations for search state 0
        # print("Active search (state 1) likelihood for info relevance:")
        # print(A[3][:, 1])  # Just the observations for search state 1

        B = self.env.get_transition_dist()

        # print("\nDEBUG - Initial matrices:")
        # print("B[2] shape:", B[2].shape)  # Check info state transition shape
        # print("Sample B[2] slice (search=0):\n", B[2][:,:,0])  # Check transition probabilities
        
        # Initialize prior preferences (C matrix) - prefer higher quality observations
        C = self._construct_C_prior()

        # print("\nChecking C matrix preferences:")
        # print("Prompt quality preferences:", C[0])  # Should show preference for high quality
        # print("Search quality preferences:", C[3])  # Should show preference for high quality
        # print("Info state preferences:", C[6])      # Should show strong preference for good info states
        
        # Initialize beliefs about initial states (D matrix)
        D = self._construct_D_prior()
        
        # Initialize prior parameters for learning observation model
        pA = self._construct_pA_prior()
        pB = self._construct_pB_prior()

        # print(f"pA: {pA}")

        policies = control.construct_policies(
            self.env.num_states,
            self.env.num_controls,
            policy_len=1
        )

        # # Filter policies
        # valid_policies = []
        # for policy in policies:
        #     # Convert to proper 2D array shape
        #     policy_reshaped = policy.reshape(1,-1) 
        #     if (policy_reshaped[0,0] > 0 and policy_reshaped[0,1] == 0) or (policy_reshaped[0,0] == 0 and policy_reshaped[0,1] > 0):
        #         valid_policies.append(policy_reshaped)
            
        # valid_policies = [p for p in valid_policies if not np.all(p == 0)]

        valid_policies = []
        for policy in policies:
            policy_reshaped = policy.reshape(1,-1)
            if (np.all(policy_reshaped == 0) or  # Include [0 0 0]
                (policy_reshaped[0,0] > 0 and policy_reshaped[0,1] == 0) or
                (policy_reshaped[0,0] == 0 and policy_reshaped[0,1] > 0)):
                valid_policies.append(policy_reshaped)

        print(f"Valid Policies: {valid_policies}")

        # If I have issues where the G values are all changing at the same rate, or the G values are the same for all of the policies it's likely due to the A and B factor List being wrong. The state factors have to be factorized and there has to be mappinds between the state and at least one of the observation modalities.
        B_factor_list = [
            [0],  # prompt factor depends only on itself
            [1],  # search factor depends only on itself
            [2] 
        ]
        
        A_factor_list = [
            [0],  # Accuracy depends only on prompts
            [0],  # Relevance depends only on prompts
            [0],  # Comprehensiveness depends only on prompts
            [1],  # Info relevance depends only on search
            [1],  # Info usefulness depends only on search
            [1],  # Source quality depends only on search
            [2]   # Info state observation depends only on search
        ]

        # B_factor_list = [
        #     [0],  # prompt factor depends only on itself
        #     [1],  # search factor depends only on itself
        #     [2] 
        # ]
        
        # A_factor_list = [
        #     [0],  # Accuracy depends only on prompts
        #     [0],  # Relevance depends only on prompts
        #     [0],  # Comprehensiveness depends only on prompts
        #     [1],  # Info relevance depends only on search
        #     [1],  # Info usefulness depends only on search
        #     [1],  # Source quality depends only on search
        #     [2]   # Info state observation depends only on search
        #     [0,1] # this will be the prompt characteristics state depending on search and prompt. Will have to see is this is a thing I can do. Probably good.
        # ]


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
            # sophisticated=True,
            si_horizon=3,
            policy_len=2,
            inference_horizon=1,
            action_selection="deterministic",
            sampling_mode="full",
            inference_algo="VANILLA",
            modalities_to_learn="all",
            lr_pA=self.learning_rate_pA,
            lr_pB=self.learning_rate_pB,
            use_utility=True,
            use_states_info_gain=True,
            use_param_info_gain=True,
            A_factor_list=A_factor_list,
            B_factor_list=B_factor_list,
            save_belief_hist=self.save_history
        )
        
        # print("\nVerifying agent policies:")
        # print(f"Number of policies: {len(agent.policies)}")
        # for i in range(min(5, len(agent.policies))):
        #     print(f"Agent policy {i}: {agent.policies[i]}")
    
        return agent

    def _construct_C_prior(self):
        C = utils.obj_array(self.env.num_modalities)
        
        # Make [0 0 0] policy have high free energy by making it very undesirable
        # to observe low quality scores when taking no action
        for modality in range(self.env.num_modalities - 1):
            C_m = np.zeros(self.env.num_obs[modality])
            C_m[0] = -16.0  # Much more negative for low quality (was -4.0)
            for i in range(1, self.env.num_quality_levels):
                C_m[i] = (i / (self.env.num_quality_levels - 1)) ** 2 * 2.0  # Less reward for high quality
            C[modality] = C_m

        # Keep strong preferences for info states
        C[INFO_STATE_MODALITY_ID] = np.array([-32.0, 8.0, 64.0])
        
        return C

    # def _construct_C_prior(self):
    #     C = utils.obj_array(self.env.num_modalities)
        
    #     # Make quality metrics less extreme
    #     for modality in range(self.env.num_modalities - 1):
    #         C_m = np.zeros(self.env.num_obs[modality])
    #         C_m[0] = -4.0  # Less negative for low quality
    #         for i in range(1, self.env.num_quality_levels):
    #             C_m[i] = (i / (self.env.num_quality_levels - 1)) ** 2 * 4.0  # Less positive for high quality
    #         C[modality] = C_m

    #     # Make info state preferences MUCH stronger
    #     C[INFO_STATE_MODALITY_ID] = np.array([-32.0, 8.0, 64.0])  # Doubled preferences for info states
        
    #     return C

    def _construct_D_prior(self) -> np.ndarray:
        """
        Construct prior beliefs about initial states.
        Initially uniform across all states.
        """
        D = utils.obj_array(self.env.num_factors)
        
        # Set uniform priors for each factor
        for f, num_states in enumerate(self.env.num_states):
            D[f] = np.ones(num_states) / num_states
            
        return D

    # def _construct_pA_prior(self) -> np.ndarray:
    #     """Construct prior counts for learning the observation model."""
    #     print("\nConstructing pA priors...")
    #     pA = utils.obj_array(self.env.num_modalities)
        
    #     # For prompt-dependent modalities (accuracy, relevance, comprehensiveness)
    #     for m in range(3):
    #         pA[m] = np.ones((self.env.num_quality_levels, self.env.num_states[0])) * 10  # Base concentration of 10
            
    #         # For each prompt combination
    #         for prompt_idx in range(self.env.num_states[0]):
    #             target_quality = min(0.9, 0.3 + (prompt_idx / self.env.num_states[0]) * 0.6)
    #             target_idx = int(target_quality * (self.env.num_quality_levels - 1))
    #             width = 2
                
    #             # Add extra concentration around target index
    #             for i in range(self.env.num_quality_levels):
    #                 distance = abs(i - target_idx)
    #                 if distance <= width:
    #                     pA[m][i, prompt_idx] += 20 * np.exp(-0.5 * (distance / width) ** 2)

    #     # For search-dependent modalities (info relevance, usefulness, source quality)
    #     for m in range(3, 6):
    #         pA[m] = np.ones((self.env.num_quality_levels, self.env.num_states[1])) * 10
            
    #         for search_idx in range(self.env.num_states[1]):
    #             target_idx = 2 if search_idx == 0 else 8  # Low/high quality peaks
    #             width = 1
                
    #             # Add extra concentration around target index
    #             for i in range(self.env.num_quality_levels):
    #                 distance = abs(i - target_idx)
    #                 if distance <= width:
    #                     pA[m][i, search_idx] += 20 * np.exp(-0.5 * (distance / width) ** 2)

    #     # Info state modality 
    #     pA[INFO_STATE_MODALITY_ID] = np.ones((self.env.num_info_states, self.env.num_states[2])) * 10
        
    #     # Add strong diagonal prior for info states
    #     for info_idx in range(self.env.num_states[2]):
    #         pA[INFO_STATE_MODALITY_ID][info_idx, info_idx] += 30  # Strong prior for correct observation

    #     return pA
    
    # def _construct_pB_prior(self):
    #     """Construct prior concentration parameters for learning the transition model."""
    #     print("\nConstructing pB priors...")
    #     pB = utils.obj_array(self.env.num_factors)
        
    #     # For prompt transitions (Factor 0)
    #     # Shape should match B[0]: (num_prompt_combinations, num_prompt_combinations, num_prompt_combinations)
    #     pB[PROMPT_FACTOR_ID] = np.ones((self.env.num_prompt_combinations,
    #                                 self.env.num_prompt_combinations,
    #                                 self.env.num_prompt_combinations)) * 10  # Base concentration of 10
        
    #     # Add stronger priors for staying in current state when no action taken
    #     for state in range(self.env.num_prompt_combinations):
    #         pB[PROMPT_FACTOR_ID][state, state, 0] += 20  # Stronger prior for state persistence under no action
        
    #     # For search transitions (Factor 1)
    #     # Shape should match B[1]: (num_search_states, num_search_states, num_search_states)
    #     pB[SEARCH_FACTOR_ID] = np.ones((self.env.num_search_states,
    #                                 self.env.num_search_states,
    #                                 self.env.num_search_states)) * 10
        
    #     # Add stronger prior for decay pattern under no action
    #     for state in range(self.env.num_search_states):
    #         pB[SEARCH_FACTOR_ID][0, state, 0] += 10  # Prior for decaying to no_search
    #         pB[SEARCH_FACTOR_ID][state, state, 0] += 20  # Prior for staying in current state
        
    #     # For info state transitions (Factor 2)
    #     # Shape should match B[2]: (num_info_states, num_info_states, 1)
    #     pB[INFO_FACTOR_ID] = np.ones((self.env.num_info_states,
    #                                 self.env.num_info_states,
    #                                 1)) * 10
        
    #     # Add stronger priors for the fixed transition pattern
    #     transitions = np.array([
    #         [0.7, 0.2, 0.1],  # From no_info
    #         [0.1, 0.7, 0.2],  # From basic_info
    #         [0.1, 0.2, 0.7]   # From detailed_info
    #     ])
        
    #     # Scale the transition probabilities into concentration parameters
    #     pB[INFO_FACTOR_ID][:, :, 0] += transitions * 30
        
    #     return pB

    # def _construct_pA_prior(self) -> np.ndarray:
    #     """Construct uniform prior counts for learning the observation model."""
    #     print("\nConstructing pA priors...")
    #     pA = utils.obj_array(self.env.num_modalities)
        
    #     # For all modalities, set uniform concentration
    #     for m in range(self.env.num_modalities):
    #         if m < 3:  # Prompt modalities
    #             pA[m] = np.ones((self.env.num_quality_levels, self.env.num_states[0])) * 10
    #         elif m < 6:  # Search modalities
    #             pA[m] = np.ones((self.env.num_quality_levels, self.env.num_states[1])) * 10
    #         else:  # Info state modality
    #             pA[m] = np.ones((self.env.num_info_states, self.env.num_states[2])) * 10
                
    #     return pA
    
    # def _construct_pB_prior(self):
    #     """Construct uniform prior concentration parameters for learning the transition model."""
    #     print("\nConstructing pB priors...")
    #     pB = utils.obj_array(self.env.num_factors)
        
    #     # For prompt transitions
    #     pB[PROMPT_FACTOR_ID] = np.ones((self.env.num_prompt_combinations,
    #                                 self.env.num_prompt_combinations,
    #                                 self.env.num_prompt_combinations)) * 10

    #     # For search transitions  
    #     pB[SEARCH_FACTOR_ID] = np.ones((self.env.num_search_states,
    #                                 self.env.num_search_states,
    #                                 self.env.num_search_states)) * 10
        
    #     # For info state transitions
    #     pB[INFO_FACTOR_ID] = np.ones((self.env.num_info_states,
    #                                 self.env.num_info_states,
    #                                 1)) * 10
            
    #     return pB

    def _construct_pA_prior(self) -> np.ndarray:
        """Construct completely uniform priors for A."""
        print("\nConstructing pA priors...")
        pA = utils.obj_array(self.env.num_modalities)
        
        # Use minimal uniform concentration for all modalities
        base_concentration = 1.0  # Minimum possible to allow maximum learning
        
        for m in range(self.env.num_modalities):
            if m < 3:  # Prompt modalities
                pA[m] = np.ones((self.env.num_quality_levels, self.env.num_states[0])) * base_concentration
            elif m < 6:  # Search modalities
                pA[m] = np.ones((self.env.num_quality_levels, self.env.num_states[1])) * base_concentration
            else:  # Info state modality
                pA[m] = np.ones((self.env.num_info_states, self.env.num_states[2])) * base_concentration
                    
        return pA
    
    # def _construct_pA_prior(self) -> np.ndarray:
    #     """Construct A priors with different distributions for different search states."""
    #     print("\nConstructing pA priors...")
    #     pA = utils.obj_array(self.env.num_modalities)
        
    #     base_concentration = 1.0
        
    #     # Prompt modalities stay uniform
    #     for m in range(3):
    #         pA[m] = np.ones((self.env.num_quality_levels, self.env.num_states[0])) * base_concentration

    #     # Search modalities (info_relevance, usefulness, source_quality)
    #     for m in range(3, 6):
    #         pA[m] = np.ones((self.env.num_quality_levels, self.env.num_states[1])) * base_concentration
            
    #         # For each search state
    #         for search_idx in range(self.env.num_search_states):
    #             if search_idx == 0:  # No search - expect low scores
    #                 peak_score = 2
    #                 width = 1
    #             else:
    #                 # Different peaks for different search terms
    #                 if search_idx in [1, 2]:  # Prompt engineering & LLM chain of thought
    #                     peak_score = 8  # High relevance expected
    #                     width = 1.5
    #                 elif search_idx in [3, 4, 5]:  # Task decomposition & system design
    #                     peak_score = 7  # Medium-high relevance
    #                     width = 2.0
    #                 elif search_idx in [6, 7]:  # Workflow & synthesis methods
    #                     peak_score = 6  # Medium relevance
    #                     width = 2.0
    #                 else:  # Verification & quality assessment
    #                     peak_score = 5  # Medium-low relevance
    #                     width = 2.5
                        
    #             # Add concentration around peak with falloff
    #             for obs_idx in range(self.env.num_quality_levels):
    #                 distance = abs(obs_idx - peak_score)
    #                 if distance <= width * 2:  # Within 2 standard deviations
    #                     pA[m][obs_idx, search_idx] += 2.0 * np.exp(-0.5 * (distance/width)**2)

    #     # Info state modality stays uniform
    #     pA[INFO_STATE_MODALITY_ID] = np.ones((self.env.num_info_states, self.env.num_states[2])) * base_concentration
                    
    #     return pA

    def _construct_pB_prior(self):
        """Construct minimal structured priors for B."""
        print("\nConstructing pB priors...")
        pB = utils.obj_array(self.env.num_factors)
        
        base_concentration = 1.0  # Minimal base concentration
        
        # Prompt transitions - only allow transitions via actions
        pB[PROMPT_FACTOR_ID] = np.ones((self.env.num_prompt_combinations,
                                    self.env.num_prompt_combinations,
                                    self.env.num_prompt_combinations)) * base_concentration
        # Add tiny bias for staying in same state when no action
        for state in range(self.env.num_prompt_combinations):
            pB[PROMPT_FACTOR_ID][state, state, 0] += 0.1
        
        # Search transitions - only allow transitions via actions
        pB[SEARCH_FACTOR_ID] = np.ones((self.env.num_search_states,
                                    self.env.num_search_states,
                                    self.env.num_search_states)) * base_concentration
        # Add tiny bias for decay to no_search when no action
        for state in range(self.env.num_search_states):
            pB[SEARCH_FACTOR_ID][0, state, 0] += 0.1
        
        # Info state transitions - needs minimal structure
        pB[INFO_FACTOR_ID] = np.ones((self.env.num_info_states,
                                    self.env.num_info_states,
                                    1)) * base_concentration
        # Add minimal forward progression bias
        for i in range(self.env.num_info_states-1):
            pB[INFO_FACTOR_ID][i+1, i, 0] += 0.1  # Tiny bias for progressing forward
        
        return pB

    def run_trial(
        self, 
        num_timesteps: int, 
        initial_observation: Optional[List[int]] = None
    ) -> Tuple[List[List[int]], List[List[float]], List[List[int]]]:
        """
        Run a single trial of the agent-environment loop.
        
        Parameters:
        -----------
        num_timesteps: int
            Number of timesteps to run the trial
        initial_observation: Optional[List[int]]
            Initial observation to start the trial (if None, will get from environment reset)
            
        Returns:
        --------
        observations: List[List[int]]
            History of observations
        states: List[List[float]]
            History of belief states
        actions: List[List[int]]
            History of actions taken
        """

        print(f"\n=== Starting New Trial (Timesteps: {num_timesteps}) ===")

        # Reset environment and agent
        if initial_observation is None:
            observation = self.env.reset()
        else:
            observation = initial_observation
            
        # self.agent.reset()
        
        # Initialize storage
        observations = [observation]
        states = []
        actions = []
        # print(f"New observation: {observations}")

        qs_prev = None
        
        for t in range(num_timesteps):

            print(f"\n--- Timestep {t+1}/{num_timesteps} ---")
            # 1. Update beliefs about hidden states
            qs = self.agent.infer_states(observation)

            # print(f"\nTimestep {t} - State Inference:")
            # print(f"Current observation: {observation}")
            # print(f"Updated state beliefs:", {f"Factor {i}": qs[i] for i in range(len(qs))})

            states.append(qs)
            
            # 2. Update beliefs about policies and get expected free energy
            q_pi, G = self.agent.infer_policies()

            if self.save_history:
                self.policy_checkpoints.append(self.agent.policies)
                self.free_energy_checkpoints.append(G)
            
            # print("\nPolicy Inference:")
            # print(f"Policy posterior: {q_pi} Max: {max(q_pi)} Min: {min(q_pi)}")
            # print(f"Expected free energies: {G} Max: {max(G)} Min: {min(G)}")

            # print(f"\nTimestep {t} Policy Details:")
            # print("Policy distribution shape:", q_pi.shape)
            # Print top 5 policies by probability
            indices = np.argsort(q_pi)[-5:]
            print("\nTop 5 policies by probability:")
            for idx in indices:
                print(f"Policy {idx}: prob={q_pi[idx]:.6f}, G={G[idx]:.2f}")
                print(f"Policy actions: {self.agent.policies[idx]}")

            # 3. Sample action
            action = self.agent.sample_action()
            actions.append(action)

            selected_policy = np.array([action]).reshape(1, -1)  # Reshape for single timestep
            expected_states = control.get_expected_states_interactions(
                qs, 
                self.agent.B, 
                self.agent.B_factor_list, 
                selected_policy
            )
            
            # Get expected observations
            expected_obs = control.get_expected_obs_factorized(
                expected_states,
                self.agent.A,
                self.agent.A_factor_list
            )

            print(f"Expected Obs: {expected_obs}")

            if self.save_history:
                # Extract probabilities for each modality
                probs = [
                    obs[0] if isinstance(obs, np.ndarray) else None 
                    for obs in expected_obs[0]
                ]
                self.prediction_history.append(probs)


            # Check for [0 0 0] action
            if np.all(action == 0):
                print("\nAgent selected [0 0 0] action - terminating trial early")
                return observations, states, actions, True


            print("\nAction sampling:")
            print("Raw sampled action:", action)
            print("Action type:", type(action))
            print(f"\nSelected action: {action}")

            # 4. Step environment
            observation = self.env.step_test(action)
            observations.append(observation)

            if self.save_history:
                self.actual_outcome_history.append(observation)

            # 5. Store history if enabled (do this BEFORE learning)
            if self.save_history:
                self._update_history(observation, qs, action)

            print(f"This is the action: {action}")

            # 6. Learn parameters (observation model)
            if hasattr(self.agent, 'pA'):
                self.agent.update_A(observation, action)

            # Only update B if we have previous states to compare with
            if hasattr(self.agent, 'pB') and qs_prev is not None:
                self.agent.update_B(qs_prev)
                
            # Update previous states for next iteration
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
        # Update quality scores first
        quality_metrics = [
            ('accuracy', ACCURACY_MODALITY_ID),
            ('relevance', RELEVANCE_MODALITY_ID),
            ('comprehensiveness', COMPREHENSIVENESS_MODALITY_ID),
            ('info_relevance', INFO_RELEVANCE_MODALITY_ID),
            ('info_usefulness', INFO_USEFULNESS_MODALITY_ID),
            ('source_quality', SOURCE_QUALITY_MODALITY_ID)
        ]
        
        for metric_name, modality_id in quality_metrics:
            if observation[modality_id] is not None:
                score = observation[modality_id] / 10.0  # Convert to 0.1-1.0 scale
                self.quality_scores[metric_name].append(score)
            else:
                self.quality_scores[metric_name].append(None)
            
        # Then update other history
        self.observation_history.append(observation)
        self.state_history.append(states)
        self.action_history.append(action)

        # Print current scores
        try:
            print("Quality scores:", {metric: self.quality_scores[metric][-1] for metric in self.quality_scores})
        except IndexError:
            print("Warning: No quality scores available yet")

    def plot_quality_history(self) -> None:
        """
        Plot the history of quality scores across all metrics.
        """
        if not self.save_history:
            raise ValueError("History tracking is disabled")
            
        plt.figure(figsize=(12, 8))
        for metric_name, scores in self.quality_scores.items():
            plt.plot(scores, label=metric_name)
            
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

    # Create environment and controller
    env = ResearchAgentEnv()
    evaluator = AgentEvaluator(save_dir="figures")
    controller = ResearchAgentController(
        env=env,
        learning_rate_pA=learning_rate_pA,
        gamma=gamma,
        alpha=alpha,
        save_history=True
    )

    # Store initial model state
    initial_A = controller.agent.A.copy()
    initial_B = controller.agent.B.copy()

    # Run trials
    completed_trials = 0
    for trial in range(num_trials):
        logging.info(f"Starting trial {trial + 1}/{num_trials}")
        observations, states, actions, terminated_early = controller.run_trial(timesteps_per_trial)
        completed_trials += 1

        if terminated_early:
            logging.info(f"Trial {trial + 1} terminated early due to [0 0 0] action")
            break

    # Store final model state
    final_A = controller.agent.A.copy()
    final_B = controller.agent.B.copy()

    # Generate visualizations
    print("\nGenerating visualizations...")
    evaluator.visualize_model_evolution(initial_A, final_A, initial_B, final_B)
    evaluator.visualize_policy_progression(
        controller.policy_checkpoints,
        controller.free_energy_checkpoints,
        completed_trials
    )
    evaluator.plot_policy_selection_heatmap(controller)
    evaluator.plot_action_timeline(controller)

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
    
    # # Plot results
    # controller.plot_quality_history()