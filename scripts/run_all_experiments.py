# scripts/run_all_experiments.py
import subprocess
import sys
import os
from pathlib import Path
from tqdm import tqdm 

# Ensure the script can find other modules in the project
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- Configuration ---
# Define the two satellite series to be processed
SATELLITE_SERIES = {
    "Jason (LEO)": ["Jason-1", "Jason-2", "Jason-3"],
    "Fengyun (GEO)": ["Fengyun-2D", "Fengyun-2F", "Fengyun-4A", "Fengyun-2H"],
}

# Define the models to run. The keys are descriptive names,
# and the values are the script files to be executed.
MODELS_TO_RUN = {
    "ARIMA Fusion": "scripts/run_fusion_detector.py",
    "XGBoost": "scripts/final_xgb_detector.py",
}

def run_experiment(model_name: str, script_path: str, satellite_name: str):
    """
    Executes a single experiment using a subprocess and streams its output in real-time.

    Args:
        model_name (str): The descriptive name of the model being run.
        script_path (str): The path to the Python script to execute.
        satellite_name (str): The name of the satellite to pass as an argument.
    """
    command = [sys.executable, script_path, satellite_name]
    
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        # By removing `capture_output=True`, the child process's stdout and stderr
        # will be streamed directly to the console in real-time. This allows
        # us to see the progress bars from the child scripts.
        # The `check=True` argument will still ensure that an error in the child
        # script will raise a CalledProcessError.
        subprocess.run(
            command,
            check=True,
            env=env
        )
    except subprocess.CalledProcessError as e:
        # The detailed error message from the child script will have already
        # been printed to the console. We just add a summary message here.
        print(f"\n--- ERROR: Experiment '{model_name} on {satellite_name}' failed. ---")
        print(f"--- See the traceback above for details. Exit code: {e.returncode} ---")
        # To stop the entire batch run on the first error, uncomment the next line
        # raise e
    except FileNotFoundError:
        print(f"\nERROR: Script not found at '{script_path}'")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")


def main():
    """
    Main function to iterate through all defined models and satellites
    and run the experiments with a tqdm progress bar.
    """
    print("=" * 70)
    print("STARTING BATCH MANEUVER DETECTION EXPERIMENTS")
    print("=" * 70)

    # Create a flat list of all experiments to run
    experiments = []
    for model_name, script_path in MODELS_TO_RUN.items():
        for series_name, satellites in SATELLITE_SERIES.items():
            for satellite in satellites:
                experiments.append({
                    "model": model_name,
                    "script": script_path,
                    "satellite": satellite,
                    "series": series_name
                })

    # Use tqdm to create a progress bar for the experiments
    with tqdm(total=len(experiments), desc="Overall Progress", unit="exp") as pbar:
        for exp in experiments:
            # Update the progress bar description to show the current task
            pbar.set_description(f"Running {exp['model']} on {exp['satellite']}")
            
            # Print a header before each experiment to clearly separate the logs
            print(f"\n\n{'='*25} RUNNING: {exp['model']} on {exp['satellite']} {'='*25}")
            
            # Run the actual experiment
            run_experiment(exp['model'], exp['script'], exp['satellite'])
            
            print(f"{'='*25} COMPLETED: {exp['model']} on {exp['satellite']} {'='*25}")
            pbar.update(1)

    print("\n\nAll experiments completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
