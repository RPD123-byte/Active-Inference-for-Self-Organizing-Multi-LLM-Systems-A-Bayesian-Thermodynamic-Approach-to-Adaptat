import numpy as np
from pymdp import utils
from pymdp.envs import Env
from pymdp import maths
from agent_graph.graph import create_graph, compile_workflow
import json
import os

# Constants for state factor indices
ACCURACY_FACTOR_ID = 0
COMPREHENSIVENESS_FACTOR_ID = 1
RELEVANCE_FACTOR_ID = 2
INFO_RELEVANCE_FACTOR_ID = 3
INFO_USEFULNESS_FACTOR_ID = 4
SOURCE_QUALITY_FACTOR_ID = 5
INFO_STATE_FACTOR_ID = 6

# Constants for modality indices (same as factors in new model)
ACCURACY_MODALITY_ID = 0
COMPREHENSIVENESS_MODALITY_ID = 1
RELEVANCE_MODALITY_ID = 2
INFO_RELEVANCE_MODALITY_ID = 3
INFO_USEFULNESS_MODALITY_ID = 4
SOURCE_QUALITY_MODALITY_ID = 5
INFO_STATE_MODALITY_ID = 6

class ResearchAgentEnv(Env):
    """
    Environment class for the research agent that implements quality state transitions.
    """
    def __init__(self, mock_workflow=None):
        # Number of quality levels for metrics
        self.num_quality_levels = 11  # 0.0 to 1.0 in 0.1 increments
        self.num_info_states = 3      # no_info, basic_info, detailed_info
        
        # Define state dimensions (now matching observation dimensions)
        self.num_states = [
            self.num_quality_levels,  # accuracy states
            self.num_quality_levels,  # comprehensiveness states
            self.num_quality_levels,  # relevance states
            self.num_quality_levels,  # info relevance states
            self.num_quality_levels,  # info usefulness states
            self.num_quality_levels,  # source quality states
            self.num_info_states      # info states
        ]

        # Number of possible actions
        self.num_prompt_combinations = 33  # Arbitrary number for demonstration
        self.num_search_terms = 10
        
        # Define number of controls for each factor
        self.total_actions = self.num_prompt_combinations + self.num_search_terms + 1  # +1 for no_action
        self.num_controls = [self.total_actions] * len(self.num_states)  # Each state can be affected by any action
        
        # Define observation dimensions (same as states in new model)
        self.num_obs = self.num_states.copy()
        
        self.num_factors = len(self.num_states)
        self.num_modalities = len(self.num_obs)
        
        # Initialize transition and likelihood distributions
        self._transition_dist = self._construct_transition_dist()
        self._likelihood_dist = self._construct_likelihood_dist()
        
        # Store current state
        self._state = None
        
        # Store mock workflow for research agent
        self.workflow = mock_workflow

        # Define factor dependencies
        self.B_factor_list = [
            [0],  # accuracy depends on itself
            [1],  # comprehensiveness depends on itself
            [2],  # relevance depends on itself
            [3],  # info relevance depends on itself
            [4],  # info usefulness depends on itself
            [5],  # source quality depends on itself
            [6]   # info state depends on itself
        ]
        
        self.A_factor_list = [
            [0],  # accuracy observation depends on accuracy state
            [1],  # comprehensiveness observation depends on comprehensiveness state
            [2],  # relevance observation depends on relevance state
            [3],  # info relevance observation depends on info relevance state
            [4],  # info usefulness observation depends on info usefulness state
            [5],  # source quality observation depends on source quality state
            [6]   # info state observation depends on info state
        ]

        # Store factor lists in mb_dict for use in inference
        A_modality_list = []
        for f in range(len(self.num_states)):
            A_modality_list.append([m for m in range(len(self.num_obs)) if f in self.A_factor_list[m]])
        
        self.mb_dict = {
            'A_factor_list': self.A_factor_list,
            'A_modality_list': A_modality_list
        }

    def _construct_transition_dist(self):
        """
        Construct the transition distributions for all state factors.
        In the new model, each quality/info state can transition based on actions.
        """
        B = utils.obj_array(self.num_factors)
        
        # For each factor (quality metrics and info state)
        for factor in range(self.num_factors):
            if factor == INFO_STATE_FACTOR_ID:
                num_states = self.num_info_states
            else:
                num_states = self.num_quality_levels
                
            # Shape: (next_state, current_state, action)
            B_factor = np.zeros((num_states, num_states, self.total_actions))
            
            # For no action (action index 0), high probability of staying in current state
            B_factor[:, :, 0] = np.eye(num_states) * 0.9
            # Small probability of decay
            for i in range(num_states):
                if i > 0:  # Can decay to lower state
                    B_factor[i-1, i, 0] = 0.1
                else:  # At lowest state, stay there
                    B_factor[i, i, 0] = 1.0
            
            # For prompt actions (1-33)
            for action in range(1, self.num_prompt_combinations + 1):
                if factor in [ACCURACY_FACTOR_ID, COMPREHENSIVENESS_FACTOR_ID, RELEVANCE_FACTOR_ID]:
                    # Prompt actions affect first 3 quality metrics
                    target_quality = min(0.9, 0.3 + (action / self.num_prompt_combinations) * 0.6)
                    target_idx = int(target_quality * (num_states - 1))
                    
                    # Create transition probabilities centered around target quality
                    for current in range(num_states):
                        dist = np.zeros(num_states)
                        for next_state in range(num_states):
                            dist[next_state] = np.exp(-0.5 * ((next_state - target_idx) / 2) ** 2)
                        B_factor[:, current, action] = dist / np.sum(dist)
                else:
                    # Prompt actions don't affect other metrics
                    B_factor[:, :, action] = np.eye(num_states)
            
            # For search actions (34-43)
            for action in range(self.num_prompt_combinations + 1, self.total_actions):
                if factor in [INFO_RELEVANCE_FACTOR_ID, INFO_USEFULNESS_FACTOR_ID, SOURCE_QUALITY_FACTOR_ID]:
                    # Search actions affect info-related quality metrics
                    search_idx = action - self.num_prompt_combinations - 1
                    base_quality = 0.5 + (search_idx / self.num_search_terms) * 0.4
                    target_idx = int(base_quality * (num_states - 1))
                    
                    # Create transition probabilities centered around target quality
                    for current in range(num_states):
                        dist = np.zeros(num_states)
                        for next_state in range(num_states):
                            dist[next_state] = np.exp(-0.5 * ((next_state - target_idx) / 2) ** 2)
                        B_factor[:, current, action] = dist / np.sum(dist)
                        
                elif factor == INFO_STATE_FACTOR_ID:
                    # Search actions can improve info state
                    for current in range(self.num_info_states):
                        if current < self.num_info_states - 1:
                            # Can transition to next state with high probability
                            B_factor[current+1, current, action] = 0.7
                            B_factor[current, current, action] = 0.3
                        else:
                            # At highest state, stay there
                            B_factor[current, current, action] = 1.0
                else:
                    # Search actions don't affect other metrics
                    B_factor[:, :, action] = np.eye(num_states)
            
            # Store the factor's transition distribution
            B[factor] = B_factor
        
        return B

    def _construct_likelihood_dist(self):
        """
        Construct the likelihood distributions for all observation modalities.
        In the new model, observations directly reflect states with high probability.
        """
        A = utils.obj_array(self.num_modalities)
        
        # For each modality (matching state factors exactly)
        for modality in range(self.num_modalities):
            if modality == INFO_STATE_MODALITY_ID:
                num_states = self.num_info_states
            else:
                num_states = self.num_quality_levels
                
            # Create likelihood matrix: P(o|s) with high probability of observing true state
            A_modality = np.zeros((num_states, num_states))
            
            # High probability of observing true state, small probability of adjacent states
            for state in range(num_states):
                A_modality[state, state] = 0.8  # 80% chance of correct observation
                
                # Add small probability of observing adjacent states
                if state > 0:
                    A_modality[state-1, state] = 0.1
                if state < num_states - 1:
                    A_modality[state+1, state] = 0.1
                    
                # Normalize if at edges
                A_modality[:, state] = A_modality[:, state] / np.sum(A_modality[:, state])
            
            A[modality] = A_modality
        
        return A
    
    def reset(self, state=None):
        """Reset environment to initial state."""
        print("\nResetting environment...")
        
        # Initialize with all factors at lowest quality/state
        full_state = utils.obj_array(self.num_factors)
        for factor in range(self.num_factors):
            if factor == INFO_STATE_FACTOR_ID:
                full_state[factor] = utils.onehot(0, self.num_info_states)  # Start at no_info
            else:
                full_state[factor] = utils.onehot(0, self.num_quality_levels)  # Start at 0.0 quality
        
        # Override with provided state if one is passed
        if state is not None:
            self._state = state
        else:
            self._state = full_state
        
        # Initial observation matches state exactly in new model
        observation = [np.argmax(s) for s in self._state]
        
        print(f"Initial state: {[np.argmax(s) for s in self._state]}")
        print(f"Initial observation: {observation}")
        
        return observation

    def step_test(self, actions):
        """
        Test version of step function that returns deterministic 'random' observations 
        based on the state and actions.
        """
        print(f"\n{'='*50}")
        print(f"Starting test environment step with action: {actions[0]}")
        print(f"{'='*50}")

        action = actions[0]  # Use first action since all factors use same action space

        # Calculate state transitions based on action
        prob_states = utils.obj_array(self.num_factors)
        
        # Get current state indices for all factors
        current_states = [np.argmax(self._state[f]) for f in range(self.num_factors)]
        
        # Generate seed from current state and action
        state_tuple = tuple(current_states)
        seed = hash((state_tuple, action)) % (2**32 - 1)
        rng = np.random.RandomState(seed)

        # If no action selected, maintain current state with small decay chance
        if action == 0:
            print("\nNo action selected (zero)")
            return [None] * self.num_modalities

        # Handle transitions for each factor based on action type
        next_states = []
        
        for factor in range(self.num_factors):
            current_state = current_states[factor]
            
            if 1 <= action <= self.num_prompt_combinations:
                # Prompt action effects
                if factor in [ACCURACY_FACTOR_ID, COMPREHENSIVENESS_FACTOR_ID, RELEVANCE_FACTOR_ID]:
                    # Calculate target quality based on prompt number
                    prompt_idx = action - 1
                    base_quality = min(0.9, 0.3 + (prompt_idx / self.num_prompt_combinations) * 0.6)
                    target_idx = int(base_quality * (self.num_quality_levels - 1))
                    
                    # Add some randomness around target
                    shift = rng.randint(-2, 3)  # -2 to +2
                    next_state = min(self.num_quality_levels - 1, 
                                   max(0, target_idx + shift))
                    
                    # Sometimes maintain current state if it's better
                    if current_state > next_state and rng.random() < 0.3:
                        next_state = current_state
                        
                else:
                    # Prompt actions don't affect other metrics
                    next_state = current_state
                    
            elif action > self.num_prompt_combinations:
                # Search action effects
                if factor in [INFO_RELEVANCE_FACTOR_ID, INFO_USEFULNESS_FACTOR_ID, SOURCE_QUALITY_FACTOR_ID]:
                    # Calculate target quality based on search term
                    search_idx = action - self.num_prompt_combinations - 1
                    base_quality = 0.5 + (search_idx / self.num_search_terms) * 0.4
                    target_idx = int(base_quality * (self.num_quality_levels - 1))
                    
                    # Add some randomness around target
                    shift = rng.randint(-2, 3)  # -2 to +2
                    next_state = min(self.num_quality_levels - 1, 
                                   max(0, target_idx + shift))
                    
                    # Sometimes maintain current state if it's better
                    if current_state > next_state and rng.random() < 0.3:
                        next_state = current_state
                        
                elif factor == INFO_STATE_FACTOR_ID:
                    # Search actions can improve info state
                    if current_state < self.num_info_states - 1 and rng.random() < 0.7:
                        next_state = current_state + 1
                    else:
                        next_state = current_state
                        
                else:
                    # Search actions don't affect other metrics
                    next_state = current_state
                    
            else:
                # Shouldn't happen but maintain current state
                next_state = current_state
                
            next_states.append(next_state)

        # Update environment state
        self._state = self._construct_state(next_states)
        
        # In test environment, observations exactly match states
        observation = next_states.copy()

        # Special cases for interesting test scenarios
        if action == 1:  # First prompt action
            print("\nSpecial case: First prompt action")
            # Boost accuracy and comprehensiveness
            observation[ACCURACY_FACTOR_ID] = min(10, observation[ACCURACY_FACTOR_ID] + 2)
            observation[COMPREHENSIVENESS_FACTOR_ID] = min(10, observation[COMPREHENSIVENESS_FACTOR_ID] + 2)
            
        elif action == self.num_prompt_combinations + 1:  # First search action
            print("\nSpecial case: First search action")
            # Boost info relevance and usefulness
            observation[INFO_RELEVANCE_FACTOR_ID] = min(10, observation[INFO_RELEVANCE_FACTOR_ID] + 2)
            observation[INFO_USEFULNESS_FACTOR_ID] = min(10, observation[INFO_USEFULNESS_FACTOR_ID] + 2)

        print(f"\nTest observation: {observation}")
        return observation

    def step(self, actions):
        """
        Execute action and return new observation.
        Actions format: Single integer representing the chosen action
        (0 = no_action, 1-33 = prompt combinations, 34-43 = search terms)
        """
        print(f"\n{'='*50}")
        print(f"Starting environment step with action: {actions[0]}")
        print(f"{'='*50}")

        # Calculate state transitions based on action
        action = actions[0]  # Use first action since all factors use same action space
        prob_states = utils.obj_array(self.num_factors)
        
        # Handle state transitions for each factor
        for factor in range(self.num_factors):
            prob_states[factor] = self._transition_dist[factor][:, :, action].dot(self._state[factor])
        
        # Sample new states
        state = [utils.sample(ps_i) for ps_i in prob_states]
        self._state = self._construct_state(state)
        
        # In new model, observations directly reflect states
        observation = [np.argmax(s) for s in self._state]

        # If no action selected, return None for all observations
        if action == 0:
            print("\nNo action selected (zero)")
            return [None] * self.num_modalities

        # Print action interpretation
        if 1 <= action <= self.num_prompt_combinations:
            prompt_idx = action - 1
            print(f"\nExecuting prompt action {prompt_idx}")
            try:
                # Execute prompt workflow and evaluate results
                self._execute_prompt_workflow(prompt_idx)
            except Exception as e:
                print(f"\nError during prompt workflow: {str(e)}")
                
        elif action > self.num_prompt_combinations:
            search_idx = action - self.num_prompt_combinations - 1
            print(f"\nExecuting search action {search_idx}")
            try:
                # Execute search workflow and evaluate results
                self._execute_search_workflow(search_idx)
            except Exception as e:
                print(f"\nError during search workflow: {str(e)}")

        print(f"\nObservation: {observation}")
        return observation

    def _execute_prompt_workflow(self, prompt_idx):
        """Execute the prompt-based workflow and evaluate results."""
        try:
            query = "Generate a report on coffee production"
            graph = create_graph(
                server='openai',
                model='gpt-4o-mini'
            )
            workflow = compile_workflow(graph)
            dict_inputs = {"research_question": query}
            limit = {"recursion_limit": 40}
            
            print("\nExecuting prompt workflow...")
            final_state = None
            for event in workflow.stream(dict_inputs, limit):
                for key in event.keys():
                    if isinstance(event[key], dict):
                        final_state = event[key]
            print("Prompt workflow completed")
            
            # Results would affect accuracy, comprehensiveness, and relevance states
            # State transitions are handled by transition_dist
            
        except Exception as e:
            print(f"\nError during prompt workflow: {str(e)}")

    def _execute_search_workflow(self, search_idx):
        """Execute the search-based workflow and evaluate results."""
        search_terms = [
            "research agent prompt engineering best practices",
            "LLM chain of thought prompting techniques",
            "collaborative AI agent prompt patterns",
            "task decomposition prompt strategies",
            "multi-agent system prompt design",
            "research workflow prompt optimization",
            "information synthesis prompt methods",
            "fact verification prompt techniques",
            "source evaluation prompt patterns",
            "research quality assessment prompts"
        ]
        
        try:
            selected_term = search_terms[search_idx]
            graph = create_graph(
                server='openai',
                model='gpt-4o-mini'
            )
            workflow = compile_workflow(graph)
            dict_inputs = {"research_question": selected_term}
            limit = {"recursion_limit": 40}
            
            print("\nExecuting search workflow...")
            final_state = None
            for event in workflow.stream(dict_inputs, limit):
                for key in event.keys():
                    if isinstance(event[key], dict):
                        final_state = event[key]
            print("Search workflow completed")
            
            # Results would affect info relevance, usefulness, and source quality states
            # State transitions are handled by transition_dist
            
        except Exception as e:
            print(f"\nError during search workflow: {str(e)}")

    def _construct_state(self, state_tuple):
        """Construct a proper state representation from state indices."""
        state = utils.obj_array(self.num_factors)
        for f, ns in enumerate(self.num_states):
            state[f] = utils.onehot(state_tuple[f], ns)
        return state

    def get_likelihood_dist(self):
        """Return copy of likelihood distribution."""
        return self._likelihood_dist.copy()

    def get_transition_dist(self):
        """Return copy of transition distribution."""
        return self._transition_dist.copy()

    @property
    def state(self):
        """Return current state."""
        return self._state