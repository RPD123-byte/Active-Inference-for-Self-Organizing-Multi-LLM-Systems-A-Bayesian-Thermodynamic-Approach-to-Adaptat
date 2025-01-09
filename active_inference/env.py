import numpy as np
from pymdp import utils
from pymdp.envs import Env
from pymdp import maths
from agent_graph.graph import create_graph, compile_workflow
from prompts.aif_param_prompts import planner_prompt_templates, selector_prompt_templates, reporter_prompt_templates, reviewer_prompt_templates, router_prompt_templates
from agents.aif_helper_agents import EvaluatorAgent, InfoEvaluatorAgent
import json
import os

# Constants for state factor indices
PROMPT_FACTOR_ID = 0
SEARCH_FACTOR_ID = 1
INFO_FACTOR_ID = 2

# Constants for modality indices
ACCURACY_MODALITY_ID = 0
RELEVANCE_MODALITY_ID = 1
COMPREHENSIVENESS_MODALITY_ID = 2
INFO_RELEVANCE_MODALITY_ID = 3
INFO_USEFULNESS_MODALITY_ID = 4
SOURCE_QUALITY_MODALITY_ID = 5
INFO_STATE_MODALITY_ID = 6

class ResearchAgentEnv(Env):
    """
    Environment class for the research agent that implements prompt selection and search functionality.
    """
    def __init__(self, mock_workflow=None):
        # Number of prompts per agent and number of agents
        self.prompts_per_agent = 2
        self.num_agents = 5
        self.num_prompt_combinations = (self.prompts_per_agent ** self.num_agents) + 1
        
        
        # print(f"\nState space size check:")
        # print(f"Prompt combinations: {self.num_prompt_combinations}")
        total_states = self.num_prompt_combinations * 11 * 3  # 11 search states, 3 info states
        # print(f"Total state combinations: {total_states:,}")
        # print(f"Estimated memory for A matrix: {(total_states * 10 * 6 * 8) / (1024*1024*1024):.2f} GB")
    
        if total_states > 100000:
            print("\nWARNING: Very large state space detected!")
        # Number of search terms and information states
        self.num_search_terms = 10
        self.num_search_states = self.num_search_terms + 1  # +1 for no_search
        self.num_info_states = 3  # no_info, basic_info, detailed_info
        
        # Number of observation levels for each quality metric
        self.num_quality_levels = 11  # 0.0 to 1.0
        
        # Define state dimensions
        self.num_states = [
            self.num_prompt_combinations,  # prompt combinations
            self.num_search_states,        # search terms + no_search
            self.num_info_states          # information states
        ]
        
        # Define number of controls for each factor
        self.num_controls = [
            self.num_prompt_combinations,  # can transition to any prompt combination
            self.num_search_states,        # can issue any search or no_search
            1                             # info state transitions automatically
        ]
        
        # Define observation dimensions for each modality
        self.num_obs = [
            self.num_quality_levels,  # accuracy
            self.num_quality_levels,  # relevance
            self.num_quality_levels,  # comprehensiveness
            self.num_quality_levels,  # information relevance
            self.num_quality_levels,  # information usefulness
            self.num_quality_levels,  # source quality
            self.num_info_states      # Added: info state modality
        ]
        
        self.num_factors = len(self.num_states)
        self.num_modalities = len(self.num_obs)
        
        # Initialize transition and likelihood distributions
        self._transition_dist = self._construct_transition_dist()
        self._likelihood_dist = self._construct_likelihood_dist()
        
        # Store current state
        self._state = None
        
        # Store mock workflow for research agent
        self.workflow = mock_workflow

        # print("\nInitializing Research Agent Environment...")
        # print(f"Number of possible prompt combinations: {self.prompts_per_agent ** self.num_agents}")
        # print(f"Number of search states: {self.num_search_states}")
        # print(f"Number of info states: {self.num_info_states}")

        # # Define which factors each factor depends on
        # self.B_factor_list = [
        #     [0],  # prompt factor depends only on itself
        #     [1],  # search factor depends only on itself
        #     [1, 2]  # info factor depends on search and itself
        # ]
        
        # self.A_factor_list = [
        #     [0],  # Accuracy depends only on prompts
        #     [0],  # Relevance depends only on prompts
        #     [0],  # Comprehensiveness depends only on prompts
        #     [1],  # Info relevance depends only on search
        #     [1],  # Info usefulness depends only on search
        #     [1],  # Source quality depends only on search
        #     [2]   # Info state observation depends only on info state factor
        # ]

        # # Store factor lists in mb_dict for use in inference
        # A_modality_list = []
        # for f in range(len(self.num_states)):
        #     A_modality_list.append([m for m in range(len(self.num_obs)) if f in self.A_factor_list[m]])
        
        # self.mb_dict = {
        #     'A_factor_list': self.A_factor_list,
        #     'A_modality_list': A_modality_list
        # }

    def reset(self, state=None):
        """Reset environment to initial state."""
        print("\nResetting environment...")
        
        # Initialize the fixed state [0 1 0]
        prompt_state = utils.onehot(0, self.num_prompt_combinations)  # 0
        search_state = utils.onehot(3, self.num_search_states)       # 1
        info_state = utils.onehot(0, self.num_info_states)          # 0
        
        full_state = utils.obj_array(self.num_factors)
        full_state[PROMPT_FACTOR_ID] = prompt_state
        full_state[SEARCH_FACTOR_ID] = search_state
        full_state[INFO_FACTOR_ID] = info_state
        
        # Override with provided state if one is passed
        if state is not None:
            self._state = state
        else:
            self._state = full_state
        
        # Return fixed observation [None, None, None, 9, 8, 7, 1]
        observation = [None] * 7  # Initialize list with 7 None values
        observation[3] = 9
        observation[4] = 8
        observation[5] = 7
        observation[6] = 1
        
        print(f"Initial state: {[np.argmax(s) for s in self._state]}")
        print(f"Initial observation: {observation}")
        return observation

    def step(self, actions):
        print(f"\n{'='*50}")
        print(f"Starting environment step with actions: {actions}")
        print(f"{'='*50}")

        prob_states = utils.obj_array(self.num_factors)

        # Handle each factor separately based on its dependencies
        for factor, state in enumerate(self._state):
            if factor == INFO_FACTOR_ID:
                # Info factor only depends on itself with fixed transitions
                # Use the single control dimension (0) for info state transitions
                trans_probs = self._transition_dist[factor][:, :, 0]
                prob_states[factor] = trans_probs.dot(state)
            else:
                # For prompt and search factors, use original calculation
                prob_states[factor] = self._transition_dist[factor][:, :, int(actions[factor])].dot(state)
                    
        state = [utils.sample(ps_i) for ps_i in prob_states]
        self._state = self._construct_state(state)
        
        observation = [None] * self.num_modalities

        if np.all(actions == 0):
            print("\nNo action selected (all zeros)")
            return observation
        
        # Print action interpretation
        if actions[0] > 0:
            prompt_selections = self._decode_prompt_combination(actions[0])
            print("\nExecuting prompt action:")
            print(f"Planner Prompt: {prompt_selections[0]}")
            print(f"Selector Prompt: {prompt_selections[1]}")
            print(f"Reporter Prompt: {prompt_selections[2]}")
            print(f"Reviewer Prompt: {prompt_selections[3]}")
            print(f"Router Prompt: {prompt_selections[4]}")
        elif actions[1] > 0:
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
            print(f"\nExecuting search action:")
            print(f"Search term: {search_terms[int(actions[1])-1]}")
        else:
            print("\nNo action selected (all zeros)")

        search_action = np.argmax(self._state[SEARCH_FACTOR_ID])
        if search_action > 0:
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
            
            selected_term = search_terms[int(search_action) - 1]
            try:
                graph = create_graph(
                    server='openai',
                    model='gpt-4o-mini'                
                )
                workflow = compile_workflow(graph)
                
                dict_inputs = {"research_question": selected_term}
                limit = {"recursion_limit": 40}
                
                print("\nExecuting search workflow...")
                for event in workflow.stream(dict_inputs, limit):
                    for key in event.keys():
                        if isinstance(event[key], dict):
                            final_state = event[key]
                print("Search workflow completed")

                print("\nEvaluating search results...")
                info_evaluator = InfoEvaluatorAgent(model="gpt-4o-mini")
                scores = info_evaluator.invoke(
                    final_state["final_reports"],
                    final_state["selector_response"]
                )

                print(f"Final State: {final_state['final_reports']}")
                
                observation[INFO_RELEVANCE_MODALITY_ID] = int(scores['info_relevance'] * 10)
                observation[INFO_USEFULNESS_MODALITY_ID] = int(scores['info_usefulness'] * 10)
                observation[SOURCE_QUALITY_MODALITY_ID] = int(scores['source_quality'] * 10)
                
                print("\nSearch evaluation scores:")
                print(f"Info Relevance: {scores['info_relevance']:.2f}")
                print(f"Info Usefulness: {scores['info_usefulness']:.2f}")
                print(f"Source Quality: {scores['source_quality']:.2f}")

                try:
                    kb_file = 'research_knowledge_base.json'
                    results = {}
                    
                    if os.path.exists(kb_file):
                        with open(kb_file, 'r') as f:
                            try:
                                results = json.load(f)
                            except json.JSONDecodeError:
                                results = {}
                    
                    next_num = len(results) + 1
                    report_content = final_state["final_reports"][0].content 
                    results[f'Result {next_num}'] = report_content
                    
                    with open(kb_file, 'w') as f:
                        json.dump(results, f, indent=2)
                        
                    num_results = len(results)
                    if num_results == 0:
                        observation[INFO_STATE_MODALITY_ID] = 0
                    elif num_results == 1:
                        observation[INFO_STATE_MODALITY_ID] = 1
                    else:
                        observation[INFO_STATE_MODALITY_ID] = 2
                        
                except Exception as e:
                    print(f"\nError saving to knowledge base: {str(e)}")
                
            except Exception as e:
                print(f"\nError during search workflow: {str(e)}")
                observation[INFO_RELEVANCE_MODALITY_ID] = 0
                observation[INFO_USEFULNESS_MODALITY_ID] = 0
                observation[SOURCE_QUALITY_MODALITY_ID] = 0
        
        else:
            prompt_state = np.argmax(self._state[PROMPT_FACTOR_ID])
            prompt_selections = self._decode_prompt_combination(prompt_state)
            
            selected_prompts = {
                'planner': planner_prompt_templates[prompt_selections[0]],
                'selector': selector_prompt_templates[prompt_selections[1]],
                'reporter': reporter_prompt_templates[prompt_selections[2]],
                'reviewer': reviewer_prompt_templates[prompt_selections[3]],
                'router': router_prompt_templates[prompt_selections[4]]
            }
            
            try:
                query = "Generate a report on coffee production"
                graph = create_graph(
                    server='openai',
                    model='gpt-4o-mini',
                    prompt_templates=selected_prompts
                )
                workflow = compile_workflow(graph)
                dict_inputs = {"research_question": query}
                limit = {"recursion_limit": 40}
                
                print("\nExecuting prompt workflow...")
                for event in workflow.stream(dict_inputs, limit):
                    for key in event.keys():
                        if isinstance(event[key], dict):
                            final_state = event[key]
                print("Prompt workflow completed")
                
                print("\nEvaluating research report...")
                evaluator = EvaluatorAgent(model="gpt-4o-mini")
                scores = evaluator.invoke(final_state["final_reports"], query)
                
                observation[ACCURACY_MODALITY_ID] = int(scores['accuracy'] * 10)
                observation[RELEVANCE_MODALITY_ID] = int(scores['relevance'] * 10)
                observation[COMPREHENSIVENESS_MODALITY_ID] = int(scores['comprehensiveness'] * 10)
                
                print("\nReport evaluation scores:")
                print(f"Accuracy: {scores['accuracy']:.2f}")
                print(f"Relevance: {scores['relevance']:.2f}")
                print(f"Comprehensiveness: {scores['comprehensiveness']:.2f}")
                
            except Exception as e:
                print(f"\nError during workflow: {str(e)}")
                observation[ACCURACY_MODALITY_ID] = 0
                observation[RELEVANCE_MODALITY_ID] = 0
                observation[COMPREHENSIVENESS_MODALITY_ID] = 0

            try:
                kb_file = 'research_knowledge_base.json'
                results = {}
                
                if os.path.exists(kb_file):
                    with open(kb_file, 'r') as f:
                        try:
                            results = json.load(f)
                        except json.JSONDecodeError:
                            results = {}
                
                next_num = len(results) + 1
                report_content = final_state["final_reports"][0].content 
                results[f'Result {next_num}'] = report_content
                
                with open(kb_file, 'w') as f:
                    json.dump(results, f, indent=2)
                    
                num_results = len(results)
                if num_results == 0:
                    observation[INFO_STATE_MODALITY_ID] = 0
                elif num_results == 1:
                    observation[INFO_STATE_MODALITY_ID] = 1
                else:
                    observation[INFO_STATE_MODALITY_ID] = 2
                    
            except Exception as e:
                print(f"\nError saving to knowledge base: {str(e)}")

        print(f"\nObservation: {observation}")
        return observation

    def step_test(self, actions):
        """
        Test version of step function that returns deterministic 'random' observations 
        based on the state and actions, with a special case high-performing prompt case.
        """
        print(f"\n{'='*50}")
        print(f"Starting test environment step with actions: {actions}")
        print(f"{'='*50}")

        # Handle state transitions same as original
        prob_states = utils.obj_array(self.num_factors)
        for factor, state in enumerate(self._state):
            if factor == INFO_FACTOR_ID:
                trans_probs = self._transition_dist[factor][:, :, 0]
                prob_states[factor] = trans_probs.dot(state)
            else:
                prob_states[factor] = self._transition_dist[factor][:, :, int(actions[factor])].dot(state)
        
        state = [utils.sample(ps_i) for ps_i in prob_states]
        self._state = self._construct_state(state)
        
        # Initialize observation array with None
        observation = [None] * self.num_modalities
        
        if np.all(actions == 0):
            print("\nNo action selected (all zeros)")
            return observation
            
        # Generate seed from current state and actions
        state_tuple = tuple(np.argmax(s) for s in self._state)
        action_tuple = tuple(actions)
        seed = hash((state_tuple, action_tuple)) % (2**32 - 1)
        rng = np.random.RandomState(seed)
        
        # Track mock knowledge base results counter
        if not hasattr(self, '_test_kb_counter'):
            self._test_kb_counter = 0
            
        if actions[0] > 0:  # Prompt action
            print("\nExecuting test prompt action:")
            prompt_selections = self._decode_prompt_combination(actions[0])
            print(f"Planner Prompt: {prompt_selections[0]}")
            print(f"Selector Prompt: {prompt_selections[1]}")
            print(f"Reporter Prompt: {prompt_selections[2]}")
            print(f"Reviewer Prompt: {prompt_selections[3]}")
            print(f"Router Prompt: {prompt_selections[4]}")
            
            base_quality = rng.randint(5, 11)
            for i in range(3):
                observation[i] = min(10, max(0, base_quality + rng.randint(-2, 3)))
                
            # Increment knowledge base counter for prompt action
            self._test_kb_counter += 1
                
        elif actions[1] > 0:  # Search action
            search_idx = int(actions[1]) - 1
            
            # Much more distinct distributions per search type
            if search_idx in [0, 1]:  # Prompt engineering & LLM topics
                observation[3] = 9  # Highly relevant
                observation[4] = 8
                observation[5] = 8
            elif search_idx in [2, 3, 4]:  # Task decomposition & system design
                observation[3] = 7  # Moderately relevant
                observation[4] = 6
                observation[5] = 7
            elif search_idx in [5, 6]:  # Workflow & synthesis
                observation[3] = 5  # Less relevant
                observation[4] = 5
                observation[5] = 5
            else:  # Verification & assessment
                observation[3] = 4  # Least relevant
                observation[4] = 3
                observation[5] = 4
                # Increment knowledge base counter for search action
            self._test_kb_counter += 1
        
        # Set info state based on knowledge base counter, same logic as step()
        if self._test_kb_counter == 0:
            observation[INFO_STATE_MODALITY_ID] = 0
        elif self._test_kb_counter == 1:
            observation[INFO_STATE_MODALITY_ID] = 1
        else:
            observation[INFO_STATE_MODALITY_ID] = 2
            
        print(f"\nTest observation: {observation}")
        print(f"Knowledge base counter: {self._test_kb_counter}")
        
        return observation
    
    def _get_observation(self):
        """Generate observations based on current state."""
        prob_obs = utils.obj_array(self.num_modalities)
        
        # Get current actions from state
        prompt_state = np.argmax(self._state[PROMPT_FACTOR_ID])
        search_state = np.argmax(self._state[SEARCH_FACTOR_ID])
        info_state = np.argmax(self._state[INFO_FACTOR_ID])

        # Check if in null state (all zeros)
        if prompt_state == 0 and search_state == 0 and info_state == 0:
            obs = [0] * self.num_modalities  # Return all zeros
        else:
            # First three modalities depend only on prompt state
            for m in range(3):
                prob_obs[m] = self._likelihood_dist[m][:, prompt_state]
            
            # Next three modalities depend only on search state
            for m in range(3, 6):
                prob_obs[m] = self._likelihood_dist[m][:, search_state]
            
            # Info state modality depends only on info state
            prob_obs[INFO_STATE_MODALITY_ID] = self._likelihood_dist[INFO_STATE_MODALITY_ID][:, info_state]
            
            # Sample observations
            obs = [utils.sample(po_i) for po_i in prob_obs]

        # print("\nGenerated observation:")
        # print(f"Prompt-dependent observations: {obs[:3]}")
        # print(f"Search-dependent observations: {obs[3:6]}")
        # print(f"Info state observation: {obs[6]}")
        
        return obs

    def _decode_prompt_combination(self, prompt_state_idx):
        """
        Decode a prompt combination index into individual prompt selections for each agent.
        Returns a list of 5 numbers (0-4) representing prompt choice for each agent.
        """
        prompt_indices = []
        remaining = prompt_state_idx
        
        for _ in range(self.num_agents):
            prompt_idx = remaining % self.prompts_per_agent
            prompt_indices.append(prompt_idx)
            remaining //= self.prompts_per_agent
            
        return prompt_indices  # [planner_idx, selector_idx, reporter_idx, reviewer_idx, router_idx]
    
    def _construct_transition_dist(self):
        """Construct the transition distributions for all state factors."""
        B = utils.obj_array(self.num_factors)
        
        # Prompt transitions (Factor 0)
        # Shape: (num_prompt_combinations, num_prompt_combinations, num_prompt_combinations)
        B_prompts = np.zeros((self.num_prompt_combinations, 
                            self.num_prompt_combinations,
                            self.num_prompt_combinations))
        
        # For no action (action index 0), stay in current state
        B_prompts[:, :, 0] = np.eye(self.num_prompt_combinations)
        
        # For each prompt action
        for action in range(1, self.num_prompt_combinations):
            B_prompts[:, :, action] = np.zeros((self.num_prompt_combinations, self.num_prompt_combinations))
            B_prompts[action, :, action] = 1.0
        
        B[PROMPT_FACTOR_ID] = B_prompts
        
        # Search transitions (Factor 1)
        # Shape: (num_search_states, num_search_states, num_search_states)
        B_search = np.zeros((self.num_search_states,
                            self.num_search_states,
                            self.num_search_states))
        
        # For no action, decay pattern
        B_search[:, :, 0] = np.eye(self.num_search_states)
        B_search[0, :, 0] = 0.3
        for i in range(self.num_search_states):
            B_search[i, i, 0] = 0.7
        
        # Normalize no-action transitions
        B_search[:, :, 0] = B_search[:, :, 0] / B_search[:, :, 0].sum(axis=0, keepdims=True)
        
        # For each search action
        for action in range(1, self.num_search_states):
            B_search[:, :, action] = np.zeros((self.num_search_states, self.num_search_states))
            B_search[action, :, action] = 1.0
        
        B[SEARCH_FACTOR_ID] = B_search
        
        # Info state transitions (Factor 2)
        # Shape: (num_info_states, num_info_states, 1)
        B_info = np.zeros((self.num_info_states,
                        self.num_info_states,
                        1))  # Only one control dimension for info states
        
        # Define transition probabilities for info states
        # Transitions are independent of actions
        transitions = np.array([
            [0.7, 0.2, 0.1],  # From no_info
            [0.1, 0.7, 0.2],  # From basic_info
            [0.1, 0.2, 0.7]   # From detailed_info
        ])
        
        B_info[:, :, 0] = transitions
        
        # Normalize
        B_info = B_info / B_info.sum(axis=0, keepdims=True)
        
        B[INFO_FACTOR_ID] = B_info

        # Print shapes for debugging
        print("\nB matrix shapes:")
        for f in range(len(B)):
            print(f"B[{f}] shape: {B[f].shape}")
        
        return B

    def _construct_likelihood_dist(self):
        """Construct the likelihood distributions for all observation modalities."""
        print("\nConstructing likelihood distribution...")
        A = utils.obj_array(self.num_modalities)
        
        # For prompt-dependent modalities (accuracy, relevance, comprehensiveness)
        # Shape should be: (num_quality_levels, num_states[0])
        for m in range(3):
            A[m] = np.zeros((self.num_quality_levels, self.num_states[0]))  # num_states[0] is num_prompt_combinations
            
            # For each prompt combination
            for prompt_idx in range(self.num_states[0]):
                target_quality = min(0.9, 0.3 + (prompt_idx / self.num_states[0]) * 0.6)
                dist = np.zeros(self.num_quality_levels)
                target_idx = int(target_quality * (self.num_quality_levels - 1))
                width = 2
                
                for i in range(self.num_quality_levels):
                    dist[i] = np.exp(-0.5 * ((i - target_idx) / width) ** 2)
                
                # Normalize
                A[m][:, prompt_idx] = dist / np.sum(dist)

        # For search-dependent modalities (info relevance, usefulness, source quality)
        # Shape should be: (num_quality_levels, num_states[1])
        for m in range(3, 6):
            A[m] = np.zeros((self.num_quality_levels, self.num_states[1]))  # num_states[1] is num_search_states
            
            for search_idx in range(self.num_states[1]):
                if search_idx == 0:  # No search
                    target_idx = 2  # Peak at low quality
                else:  # Active search
                    target_idx = 8  # Peak at high quality
                
                width = 1
                dist = np.zeros(self.num_quality_levels)
                
                for i in range(self.num_quality_levels):
                    dist[i] = np.exp(-0.5 * ((i - target_idx) / width) ** 2)
                
                # Normalize
                A[m][:, search_idx] = dist / np.sum(dist)

        # Info state modality 
        # Shape should be: (num_info_states, num_states[2])
        A[INFO_STATE_MODALITY_ID] = np.zeros((self.num_info_states, self.num_states[2]))  # num_states[2] is num_info_states
        
        # Each info state is clearly observable
        for info_idx in range(self.num_states[2]):
            dist = np.zeros(self.num_info_states)
            dist[info_idx] = 0.9  # High probability of correct observation
            remaining_prob = 0.1 / (self.num_info_states - 1)
            for i in range(self.num_info_states):
                if i != info_idx:
                    dist[i] = remaining_prob
            A[INFO_STATE_MODALITY_ID][:, info_idx] = dist

        # Print shapes for debugging
        print("\nA matrix shapes:")
        for m in range(len(A)):
            print(f"A[{m}] shape: {A[m].shape}")

        return A


    def _construct_state(self, state_tuple):
        """Construct a proper state representation from state indices."""
        state = utils.obj_array(self.num_factors)
        for f, ns in enumerate(self.num_states):
            state[f] = utils.onehot(state_tuple[f], ns)
        return state

    # def sample_action(self):
    #     """Sample a random action."""
    #     return [np.random.randint(self.num_controls[i]) for i in range(self.num_factors)]

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
    
