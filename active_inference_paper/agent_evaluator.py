import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from typing import List, Dict, Any, Union
import os
from pathlib import Path
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

class AgentEvaluator:
    def __init__(self, save_dir: str = "figures"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Initialize storage for metrics
        self.metrics = {
            'action_entropy': [],
            'policy_diversity': [],
            'info_state_progression': [],
            'quality_metrics': {
                'accuracy': [],
                'relevance': [],
                'comprehensiveness': [],
                'info_relevance': [],
                'info_usefulness': [],
                'source_quality': []
            },
            'action_distributions': {
                'prompt_actions': [],
                'search_actions': []
            },
            'state_uncertainties': [],
            'learning_progress': [],
            'info_seeking_behavior': []
        }

        
    def update_metrics(self, 
                      timestep: int,
                      trial: int,
                      observation: List[int],
                      action: List[int],
                      state_beliefs: Union[List[np.ndarray], np.ndarray],
                      policy_distribution: np.ndarray,
                      expected_free_energy: np.ndarray) -> None:
        """Update all tracking metrics for a single timestep."""
        
        # 1. Track action entropy
        if isinstance(policy_distribution, np.ndarray) and policy_distribution.size > 0:
            # Add small epsilon to avoid log(0)
            valid_probs = policy_distribution[policy_distribution > 0]
            if valid_probs.size > 0:
                action_entropy = -np.sum(valid_probs * np.log(valid_probs))
                self.metrics['action_entropy'].append({
                    'timestep': timestep,
                    'trial': trial,
                    'entropy': action_entropy
                })
            
        # 2. Track policy diversity
        if isinstance(action, (list, np.ndarray)) and len(action) >= 2:
            self.metrics['action_distributions']['prompt_actions'].append(action[0])
            self.metrics['action_distributions']['search_actions'].append(action[1])
            
        # 3. Track info state progression
        if isinstance(observation, (list, np.ndarray)) and len(observation) > 6 and observation[6] is not None:
            self.metrics['info_state_progression'].append({
                'timestep': timestep,
                'trial': trial,
                'info_state': observation[6]
            })
            
        # 4. Track quality metrics
        quality_indices = [0, 1, 2, 3, 4, 5]  # Indices for quality observations
        for idx, metric in zip(quality_indices, self.metrics['quality_metrics'].keys()):
            if isinstance(observation, (list, np.ndarray)) and len(observation) > idx and observation[idx] is not None:
                self.metrics['quality_metrics'][metric].append({
                    'timestep': timestep,
                    'trial': trial,
                    'score': observation[idx] / 10.0  # Normalize to 0-1
                })
                
        # 5. Track state uncertainty
        if isinstance(state_beliefs, (list, np.ndarray)) and len(state_beliefs) > 0:
            # Handle both list of arrays and single array
            if isinstance(state_beliefs, list):
                uncertainties = [self._calculate_entropy(belief) for belief in state_beliefs if belief is not None]
            else:
                uncertainties = [self._calculate_entropy(state_beliefs)]
                
            if uncertainties:  # Only append if we have valid uncertainties
                self.metrics['state_uncertainties'].append({
                    'timestep': timestep,
                    'trial': trial,
                    'uncertainty': np.mean(uncertainties)
                })
            
        # 6. Track info seeking behavior
        if isinstance(action, (list, np.ndarray)) and len(action) >= 2 and \
           isinstance(observation, (list, np.ndarray)) and len(observation) > 6 and observation[6] is not None:
            info_gain = self._calculate_info_gain(action, observation[6])
            self.metrics['info_seeking_behavior'].append({
                'timestep': timestep,
                'trial': trial,
                'info_gain': info_gain
            })

    def _calculate_entropy(self, distribution: Union[List[np.ndarray], np.ndarray]) -> float:
        """
        Calculate entropy for state beliefs.
        
        Args:
            distribution: List of belief distributions for each factor
                        [prompt_beliefs, search_beliefs, info_state_beliefs]
        Returns:
            float: Average entropy across factors
        """
        try:
            # Handle distribution as a list of separate factor beliefs
            if isinstance(distribution, list):
                entropies = []
                for dist in distribution:
                    if dist is not None and isinstance(dist, np.ndarray):
                        # Calculate entropy for this factor's distribution
                        entropy = self._single_distribution_entropy(dist)
                        if entropy is not None:
                            entropies.append(entropy)
                return np.mean(entropies) if entropies else 0.0
                
            # If somehow passed a single array
            elif isinstance(distribution, np.ndarray):
                return self._single_distribution_entropy(distribution)
                
            return 0.0
            
        except Exception as e:
            print(f"Error in _calculate_entropy: {e}")
            return 0.0
            
    def _single_distribution_entropy(self, dist: np.ndarray) -> float:
        """Calculate entropy for a single probability distribution."""
        try:
            if not isinstance(dist, np.ndarray):
                return 0.0
                
            # Find valid probabilities (greater than 0)
            valid_probs_mask = dist > 0
            if not np.any(valid_probs_mask):  # Check if we have any valid probabilities
                return 0.0
                
            valid_probs = dist[valid_probs_mask]
            return -np.sum(valid_probs * np.log(valid_probs))
            
        except Exception as e:
            print(f"Error in _single_distribution_entropy: {e}")
            return 0.0
        
    def display_matrix(self, data, title_str):
        """
        Display and save visualization of matrices or tensors, handling pymdp object arrays.
        """
        if isinstance(data, np.ndarray) and data.dtype == object:
            num_matrices = len(data)
            
            if "B" in title_str:  # B matrices need special handling
                # First count how many slices we need to display for first two matrices
                total_slices = 0
                for i in range(2):
                    if i < num_matrices and data[i] is not None:
                        if data[i].ndim == 3:
                            total_slices += data[i].shape[2]
                        else:
                            total_slices += 1
                
                num_cols = 9  # Fixed number of columns
                num_rows = 1 + ((total_slices-1) // num_cols)  # Calculate rows needed
                
                # Much wider figure size to accommodate columns
                fig = plt.figure(figsize=(40, 6 * num_rows))
                plt.subplots_adjust(wspace=0.4, hspace=0.6)  # Add more space between subplots
                
                # Plot first two matrices (potentially with multiple slices)
                slice_count = 0
                for i in range(2):
                    if i < num_matrices and data[i] is not None:
                        matrix = data[i]
                        if isinstance(matrix, np.ndarray):
                            if matrix.ndim == 3:  # 3D tensor
                                for j in range(matrix.shape[2]):
                                    ax = plt.subplot(num_rows, num_cols, slice_count + 1)
                                    im = plt.imshow(matrix[:, :, j], cmap='viridis', aspect='equal')
                                    plt.colorbar(im, fraction=0.046, pad=0.04)
                                    ax.set_title(f'Matrix {i} Slice {j}', pad=10)
                                    ax.set_xticks(np.arange(matrix.shape[1]))
                                    ax.set_yticks(np.arange(matrix.shape[0]))
                                    ax.grid(True, color='w', linewidth=0.5)
                                    slice_count += 1
                            else:  # 2D matrix
                                ax = plt.subplot(num_rows, num_cols, slice_count + 1)
                                im = plt.imshow(matrix, cmap='viridis', aspect='equal')
                                plt.colorbar(im, fraction=0.046, pad=0.04)
                                ax.set_title(f'Matrix {i}', pad=10)
                                ax.set_xticks(np.arange(matrix.shape[1]))
                                ax.set_yticks(np.arange(matrix.shape[0]))
                                ax.grid(True, color='w', linewidth=0.5)
                                slice_count += 1
                
                # Then handle matrix 2 (info state transitions) separately
                if num_matrices > 2 and data[2] is not None:
                    ax = plt.subplot(num_rows, num_cols, slice_count + 1)
                    matrix = data[2]
                    if isinstance(matrix, np.ndarray):
                        im = plt.imshow(matrix, cmap='viridis', aspect='equal')
                        plt.colorbar(im, fraction=0.046, pad=0.04)
                        ax.set_title('Matrix 2 (Info State Transitions)', pad=10)
                        ax.set_xticks(np.arange(matrix.shape[1]))
                        ax.set_yticks(np.arange(matrix.shape[0]))
                        ax.grid(True, color='w', linewidth=0.5)
            
            else:  # A matrices
                # Create a more organized horizontal layout for A matrices
                cols = 4  # Increased columns for horizontal layout
                rows = int(np.ceil(num_matrices / cols))
                
                # Adjust figure size for horizontal layout
                # Increase height slightly to accommodate rotated labels
                fig = plt.figure(figsize=(24, 6 * rows))
                
                # Add generous spacing between subplots
                plt.subplots_adjust(
                    left=0.05,   # Left margin
                    right=0.95,  # Right margin
                    bottom=0.15, # Increased bottom margin for labels
                    top=0.9,     # Top margin
                    wspace=0.4,  # Width spacing between subplots
                    hspace=0.5   # Height spacing between subplots
                )
                
                for i in range(num_matrices):
                    ax = plt.subplot(rows, cols, i+1)
                    matrix = data[i]
                    
                    if matrix is not None and isinstance(matrix, np.ndarray):
                        im = plt.imshow(matrix, cmap='viridis', aspect='equal')
                        
                        # Adjust colorbar size and position
                        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
                        cbar.ax.tick_params(labelsize=10)
                        
                        # Improve title and axis labels
                        if i <= 2:
                            title = "Prompt Quality Matrix"
                        elif i <= 5:
                            title = "Search Quality Matrix"
                        else:
                            title = "Info State Matrix"
                            
                        ax.set_title(f'{title} {i}', pad=10, fontsize=11, fontweight='bold')
                        
                        # Add and style grid
                        ax.set_xticks(np.arange(matrix.shape[1]))
                        ax.set_yticks(np.arange(matrix.shape[0]))
                        ax.grid(True, color='w', linewidth=0.5, alpha=0.3)
                        
                        # Improved x-axis label handling
                        if matrix.shape[1] > 20:  # For matrices with many columns
                            # Show every nth label
                            n = max(1, matrix.shape[1] // 20)  # Show ~20 labels max
                            xticks = np.arange(0, matrix.shape[1], n)
                            ax.set_xticks(xticks)
                            ax.set_xticklabels([str(x) for x in xticks])
                            
                            # Rotate labels more and adjust alignment
                            plt.setp(ax.get_xticklabels(), rotation=90, ha='center', fontsize=8)
                        else:
                            # For matrices with fewer columns
                            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
                        
                        # Y-axis label handling
                        plt.setp(ax.get_yticklabels(), fontsize=10)
                        
                        # Add subtle box around subplot
                        for spine in ax.spines.values():
                            spine.set_linewidth(0.5)
                            spine.set_color('gray')
            
            # Adjust main title
            plt.suptitle(title_str, y=1.02, fontsize=16, fontweight='bold')
            
            try:
                plt.tight_layout()
            except:
                print(f"Warning: Tight layout failed for {title_str}")
                
            # Save with high quality settings
            plt.savefig(
                self.save_dir / f"{title_str.replace(' ', '_')}.png",
                bbox_inches='tight',
                dpi=300,
                facecolor='white',
                edgecolor='none'
            )
            plt.close()

    def visualize_model_evolution(self, initial_A, final_A, initial_B, final_B):
        """Visualize initial and final states of A and B matrices."""
        self.display_matrix(initial_A, "Initial A Matrices")
        self.display_matrix(final_A, "Final A Matrices")
        self.display_matrix(initial_B, "Initial B Matrices")
        self.display_matrix(final_B, "Final B Matrices")

    def visualize_policy_progression(self, policy_checkpoints, free_energies_checkpoints, total_trials):
        """
        Visualize the progression of policy free energies across checkpoints.
        All trials use the same coloring scheme based on EFE values.
        
        Parameters:
        -----------
        policy_checkpoints : list
            List of policy arrays at each checkpoint
        free_energies_checkpoints : list
            List of free energy arrays at each checkpoint
        total_trials : int
            Total number of trials in the experiment
        """
        # Select 4 evenly spaced checkpoints
        total_checkpoints = len(free_energies_checkpoints)
        if total_checkpoints < 4:
            print("Not enough checkpoints for visualization")
            return
            
        # Calculate checkpoint indices and corresponding trial numbers
        indices = [
            0,
            total_checkpoints // 3,
            2 * total_checkpoints // 3,
            total_checkpoints - 1
        ]
        
        # Calculate actual trial numbers
        checkpoints_per_trial = total_checkpoints / total_trials
        trial_numbers = [int(idx / checkpoints_per_trial) + 1 for idx in indices]
        
        selected_policies = [policy_checkpoints[i] for i in indices]
        selected_energies = [free_energies_checkpoints[i] for i in indices]
        
        # Find global min and max EFE values for consistent color scaling
        all_energies = np.concatenate(selected_energies)
        vmin, vmax = np.min(all_energies), np.max(all_energies)
        
        # Create figure
        fig = plt.figure(figsize=(20, 8))
        gs = gridspec.GridSpec(1, 7, width_ratios=[1, 0.3, 1, 0.3, 1, 0.3, 1])
        
        def format_policy(policy):
            if policy[0][0] > 0:
                return f"P{policy[0][0]}"
            else:
                return f"S{policy[0][1]}"
        
        # Plot each checkpoint
        for i in range(4):
            ax = plt.subplot(gs[i*2])
            
            # Get policies and free energies
            policies = selected_policies[i]
            free_energies = selected_energies[i]
            
            # Create policy labels
            policy_labels = [format_policy(p) for p in policies]
            
            # Reshape data for heatmap
            data = free_energies.reshape(-1, 1)
            
            # Create heatmap with consistent color scaling
            im = ax.imshow(data, aspect='auto', cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
            
            # Add colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            
            # Configure axes
            ax.set_yticks(range(len(policy_labels)))
            ax.set_yticklabels(policy_labels)
            ax.set_xticks([])
            
            # Add trial number label
            title = f"Trial {trial_numbers[i]}"
            ax.set_title(title, pad=10, fontsize=12, fontweight='bold')
            
            # Add grid
            ax.grid(True, color='white', linewidth=0.5, alpha=0.3)
            
            # Add value annotations
            for j in range(len(free_energies)):
                text = ax.text(0, j, f'{free_energies[j]:.5f}',
                             ha='center', va='center', 
                             color='white' if free_energies[j] < (vmin + vmax)/2 else 'black')
            
            # Add arrow if not the last matrix
            if i < 3:
                ax_arrow = plt.subplot(gs[i*2 + 1])
                ax_arrow.set_xticks([])
                ax_arrow.set_yticks([])
                ax_arrow.spines['top'].set_visible(False)
                ax_arrow.spines['right'].set_visible(False)
                ax_arrow.spines['bottom'].set_visible(False)
                ax_arrow.spines['left'].set_visible(False)
                
                # Draw arrow
                arrow = plt.arrow(0.1, 0.5, 0.8, 0,
                                head_width=0.1, head_length=0.1,
                                fc='black', ec='black',
                                transform=ax_arrow.transAxes)
        
        plt.suptitle('Policy Free Energy Progression', y=1.02, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(
            self.save_dir / "policy_progression.png",
            bbox_inches='tight',
            dpi=300,
            facecolor='white',
            edgecolor='none'
        )
        plt.close()

    def save_checkpoint(self, step, policies, free_energies):
        """
        Save a checkpoint of policy free energies during the experiment.
        
        Parameters:
        -----------
        step : int
            Current timestep
        policies : array-like
            Current policy array
        free_energies : array-like
            Current free energy array for the policies
        """
        self.policy_checkpoints.append(policies.copy())
        self.free_energy_checkpoints.append(free_energies.copy())

    def visualize_prediction_accuracy(self, prediction_history, actual_outcome_history):
        if len(prediction_history) != len(actual_outcome_history):
            print("Error: History lengths don't match")
            return

        modality_names = ['Accuracy', 'Relevance', 'Comprehensiveness', 
                        'Info Relevance', 'Info Usefulness', 'Source Quality', 
                        'Info State']
        
        num_modalities = len(modality_names)
        num_timesteps = len(prediction_history)
        accuracies = np.full((num_timesteps, num_modalities), np.nan)
        
        for t in range(num_timesteps):
            pred_probs = prediction_history[t]
            actual = actual_outcome_history[t]
            
            for m in range(num_modalities):
                if actual[m] is not None:
                    accuracies[t, m] = pred_probs[m]  # Just use the probability directly as accuracy

        # Rest of visualization code stays the same
        plt.figure(figsize=(15, 8))
        for m in range(num_modalities):
            valid_indices = ~np.isnan(accuracies[:, m])
            if np.any(valid_indices):
                timestamps = np.arange(num_timesteps)[valid_indices]
                valid_accuracies = accuracies[valid_indices, m]
                plt.plot(timestamps, valid_accuracies, 
                        label=modality_names[m],
                        marker='o',
                        linestyle='-',
                        markersize=8,
                        alpha=0.7)

        plt.title('Prediction Accuracy Over Time\n(Only showing timesteps with actual observations)')
        plt.xlabel('Timestep')
        plt.ylabel('Prediction Accuracy')
        plt.grid(True, alpha=0.3)
        plt.ylim(-0.1, 1.1)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(self.save_dir / "prediction_accuracy.png", bbox_inches='tight', dpi=300)
        plt.close()

    def visualize_policy_history(self):
        """
        Create visualization using saved checkpoints.
        Should be called at the end of the experiment.
        """
        if len(self.policy_checkpoints) < 4:
            print("Not enough checkpoints collected for visualization")
            return
            
        # Select 4 evenly spaced checkpoints (including first and last)
        total_checkpoints = len(self.policy_checkpoints)
        indices = [0,
                total_checkpoints // 3,
                2 * total_checkpoints // 3,
                total_checkpoints - 1]
        
        selected_policies = [self.policy_checkpoints[i] for i in indices]
        selected_energies = [self.free_energy_checkpoints[i] for i in indices]
        
        self.visualize_policy_progression(selected_policies, selected_energies)
                
    def _calculate_info_gain(self, action: List[int], new_info_state: int) -> float:
        """Calculate information gain from an action."""
        # Simple heuristic: higher info states indicate more information gained
        return new_info_state / 2.0  # Normalize to 0-1 range
        
    def plot_policy_selection_heatmap(self, controller) -> None:
        """Create a heatmap of policy selections from controller's action history."""
        if not controller.action_history:
            print("No action history available")
            return
            
        # Extract prompt and search actions from action history
        actions = np.array(controller.action_history)
        prompt_actions = actions[:, 0]
        search_actions = actions[:, 1]
        
        # Create 2D histogram of actions
        hist, xedges, yedges = np.histogram2d(
            prompt_actions,
            search_actions,
            bins=[range(33), range(12)]
        )
        
        # Create figure and axis objects
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create heatmap
        hm = sns.heatmap(hist.T, cmap='viridis', ax=ax)
        
        # Get the colorbar and set integer ticks
        cbar = hm.collections[0].colorbar
        max_count = int(np.max(hist))
        cbar.set_ticks(range(max_count + 1))  # Set integer ticks from 0 to max count
        cbar.set_ticklabels(range(max_count + 1))  # Set labels to match
        cbar.set_label('# of times action was selected', rotation=270, labelpad=15)
        
        plt.title('Policy Selection Distribution')
        plt.xlabel('Prompt Actions')
        plt.ylabel('Search Actions')
        
        plt.savefig(self.save_dir / 'policy_selection_heatmap.png', 
                    bbox_inches='tight', 
                    dpi=300)
        plt.close()
    
    def plot_action_timeline(self, controller) -> None:
        """Create a timeline visualization of actions taken during the experiment."""
        if not controller.action_history:
            print("No action history available")
            return

        actions = np.array(controller.action_history)
        timesteps = range(len(actions))
        
        # Create figure with wide aspect ratio
        fig, ax = plt.subplots(figsize=(30, 8))
        
        # Set background color and grid
        ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('#f8f9fa')
        
        # Define colors
        prompt_color = '#2E86C1'  # Nice blue
        search_color = '#E74C3C'  # Nice red
        
        # Set y positions for dots
        prompt_y = 1.2
        search_y = 0
        
        # Reduced label offset
        label_offset = 0.2  # Reduced from 0.4
        
        # Plot points and labels
        prompt_count = 0  # Counter for prompt labels
        search_count = 0  # Counter for search labels
        
        for t in timesteps:
            prompt_action = actions[t][0]
            search_action = actions[t][1]
            
            if prompt_action > 0:
                # Plot prompt action
                ax.scatter(t, prompt_y, color=prompt_color, s=150, alpha=0.7, 
                        edgecolor='white', linewidth=2)
                
                # Alternate labels above and below
                if prompt_count % 2 == 0:
                    label_y = prompt_y + label_offset
                    va = 'bottom'
                else:
                    label_y = prompt_y - label_offset
                    va = 'top'
                
                ax.text(t, label_y, f'P{prompt_action}', 
                    ha='center', va=va,
                    fontsize=10, color=prompt_color, fontweight='bold')
                prompt_count += 1
                
            elif search_action > 0:
                # Plot search action
                ax.scatter(t, search_y, color=search_color, s=150, alpha=0.7, 
                        edgecolor='white', linewidth=2)
                
                # Alternate labels above and below
                if search_count % 2 == 0:
                    label_y = search_y + label_offset
                    va = 'bottom'
                else:
                    label_y = search_y - label_offset
                    va = 'top'
                
                ax.text(t, label_y, f'S{search_action}', 
                    ha='center', va=va,
                    fontsize=10, color=search_color, fontweight='bold')
                search_count += 1
        
        # Customize the plot
        ax.set_yticks([search_y, prompt_y])
        ax.set_yticklabels(['Search Actions', 'Prompt Actions'], fontsize=12)
        ax.set_xlabel('Timestep', fontsize=12, labelpad=10)
        
        # Set title with better styling
        ax.set_title('Action Selection Timeline Throughout Experiment', 
                    pad=20, fontsize=14, fontweight='bold')
        
        # Add grid with custom styling
        ax.grid(True, linestyle='--', alpha=0.3, color='gray')
        
        # Set y-axis limits with adjusted padding for closer labels
        ax.set_ylim(-0.6, 1.8)  # Reduced from (-1, 2.2)
        
        # Add legend with better styling
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=prompt_color, 
                    label='Prompt Actions', markersize=12, alpha=0.7, 
                    markeredgecolor='white', markeredgewidth=2),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=search_color, 
                    label='Search Actions', markersize=12, alpha=0.7, 
                    markeredgecolor='white', markeredgewidth=2)
        ]
        legend = ax.legend(handles=legend_elements, loc='upper right', 
                        frameon=True, framealpha=0.95, fontsize=10)
        legend.get_frame().set_facecolor('white')
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('gray')
        ax.spines['bottom'].set_color('gray')
        
        # Set x-axis ticks
        ax.set_xticks(range(len(actions)))
        ax.set_xticklabels(range(len(actions)), rotation=0)
        
        # Adjust layout and save with high quality
        plt.tight_layout()
        plt.savefig(self.save_dir / 'action_timeline.png',
                    bbox_inches='tight',
                    dpi=300,
                    facecolor='#f8f9fa',
                    edgecolor='none')
        plt.close()