"""

Generates a heatmap based on simulations of an infectious disease model.
For each combination of infection rate (beta) and recovery rate (gamma), the model is run multiple times.
The number of infected agents at the end of each simulation is recorded, and the average across all runs is used
to generate a heatmap, which is saved for later analysis.

"""


from model import interaction_model
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def create_heatmap():
    # Set parameters
    steps = 5000
    num_simulations = 10
    gamma_values = np.linspace(0, 1, 11)
    beta_values = np.arange(0.001, 0.051, 0.001)

    # Create a directory for saving results
    base_folder = "data/heatmap"
    os.makedirs(base_folder, exist_ok=True)

    # Initialise heatmap data
    heatmap_data = np.zeros((len(gamma_values), len(beta_values)))

    # Run simulations for each combination of gamma and beta
    for i, gamma in enumerate(tqdm(gamma_values, desc="Gamma Progress")):
        for j, beta in enumerate(beta_values):
            infected_counts = []
            for sim_num in range(1, num_simulations + 1):
                try:
                    # Initialise the model
                    model = interaction_model(
                        gamma=gamma,
                        n=6,
                        infection_rate=beta,
                        recovery_rate=0.1,
                        seed=sim_num
                    )
                    model.reset()

                    # Run the model for the specified number of steps
                    for step in range(steps):
                        model.step()

                    # Count infected agents at the final step
                    infected_count = sum(model.agent_flex[:, -1] == 1)
                    infected_counts.append(infected_count)
                except Exception as e:
                    print(f"Error during simulation {sim_num} for gamma {gamma:.2f}, beta {beta:.3f}: {e}")

            # Record the average number of infected agents
            heatmap_data[i, j] = np.mean(infected_counts)
            print(f"Finished: Gamma = {gamma:.2f}, Beta = {beta:.3f}, Avg Infected = {heatmap_data[i, j]:.2f}")

    # Save heatmap data
    np.save(os.path.join(base_folder, "heatmap_data.npy"), heatmap_data)


# Entry point for the script
if __name__ == "__main__":
    create_heatmap()
