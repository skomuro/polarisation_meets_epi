"""

To reproduce data in "data/low_infection_rate", please input the following command:
    python simulation.py --infection_rate 0.005 --recovery_rate 0.1 --output_folder_name "low_infection_rate"

Similarly, to reproduce data in "data/high_infection_rate", please follow these steps:
1. Update the step function in the interaction_model class to use the faster-attitude update version.
2. Change the number of steps in the "run_simulation" function to 5000 (This ensures that attitude updates remain at 50000 steps).
3. Input the following command:
    python simulation.py --infection_rate 0.05 --recovery_rate 0.01 --output_folder_name "high_infection_rate"

"""



from model import interaction_model
import argparse
import os
import numpy as np



def run_simulation(infection_rate, recovery_rate, output_folder, steps=50000, num_simulations=100):
    """Run simulations for sorting and infection dynamics."""
    gamma_values = np.linspace(0, 1, 51)  # Change gamma from 0 to 1 in 51steps

    # Create the output folder if it does not exist
    output_path = f"data/{output_folder}"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Run simulations for each gamma value
    for gamma in gamma_values:
        all_psi = []
        all_infected = []
        all_aware = []  

        for i in range(num_simulations):
            # Initialise the model
            model = interaction_model(
                gamma=gamma,
                n=6,  # Includes infection dimension
                infection_rate=infection_rate,
                recovery_rate=recovery_rate,
                seed=97 + i  # Change random seed for each simulation
            )
            model.reset()

            # Track the number of infected individuals
            for _ in range(steps):
                model.step()

            # Calculate sorting , number of infected, and number of aware individuals
            psi = model.calculate_sorting()
            num_infected = model.infected_population()
            num_aware = sum(model.agent_flex[:, -2] == 0)  # Calculate the number of aware individuals
            # Append results to lists
            all_psi.append(psi)
            all_infected.append(num_infected)
            all_aware.append(num_aware)
            print(f"Gamma: {gamma:.2f}, Simulation: {i}")

        # Calculate the median values
        mean_psi = np.mean(all_psi)
        mean_infected = np.mean(all_infected)
        mean_aware = np.mean(all_aware)  # Calculate the average of num_aware

        # Save results corresponding to each gamma value to a file
        filename = os.path.join(output_path, f"infection_rate_{infection_rate:.3f}_recovery_rate_{recovery_rate:.2f}_gamma_{gamma:.2f}".replace('.', ''))
        np.save(filename, {
            'gamma': gamma, 'psi': all_psi, 'infected': all_infected, 'aware': all_aware,
            'mean_psi': mean_psi, 'mean_infected': mean_infected, 'mean_aware': mean_aware
        })
        print(f"Results saved to {filename}")



# Ensure the script runs only when executed directly
if __name__ == "__main__":
    # Command-line argument setup
    parser = argparse.ArgumentParser(description="Simulate sorting and infection dynamics.")
    parser.add_argument("--infection_rate", type=float, required=True, help="Infection rate for the model.")
    parser.add_argument("--recovery_rate", type=float, required=True, help="Recovery rate for the model.")
    parser.add_argument("--output_folder_name", type=str, required=True, help="Folder to save output results.")
    args = parser.parse_args()

    # Run the simulation
    run_simulation(args.infection_rate, args.recovery_rate, args.output_folder_name)