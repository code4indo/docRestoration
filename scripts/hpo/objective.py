import optuna
import subprocess
import mlflow
from mlflow.tracking import MlflowClient
import os
import json
from datetime import datetime
import time # Import time module
import argparse # Import argparse for command line arguments

# --- Configuration ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
TRAIN_SCRIPT_PATH = os.path.join(PROJECT_ROOT, 'dual_modal_gan', 'scripts', 'train32.py')
MLFLOW_TRACKING_URI = f"file:{os.path.join(PROJECT_ROOT, 'mlruns')}"
MLFLOW_EXPERIMENT_NAME = "HPO_Loss_Weights_v2" # A dedicated experiment for HPO runs

# --- Objective Function ---
def objective(trial: optuna.trial.Trial):
    # Ensure Python output is unbuffered for better logging visibility
    os.environ['PYTHONUNBUFFERED'] = '1'

    # --- 1. Suggest Hyperparameters (Loss Weights) ---
    pixel_loss_weight = trial.suggest_float('pixel_loss_weight', 1.0, 200.0, step=10.0) # Extreme range
    rec_feat_loss_weight = trial.suggest_float('rec_feat_loss_weight', 1.0, 150.0, step=10.0) # Extreme range
    adv_loss_weight = trial.suggest_float('adv_loss_weight', 0.1, 10.0, step=0.5) # Extreme range
    
    # Fixed parameters for the HPO runs (from our previous analysis and decisions)
    epochs = 2 # Short duration for end-to-end simulation
    batch_size = 4 # Changed back to 4 due to OOM error
    gpu_id = "1" # Assuming GPU 1 is available
    recognizer_weights = os.path.join(PROJECT_ROOT, "models", "best_htr_recognizer", "best_model.weights.h5")
    
    # --- 2. Prepare MLflow for this trial ---
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    # Start a new MLflow run for this trial
    with mlflow.start_run(run_name=f"trial_{trial.number}") as run:
        run_id = run.info.run_id
        print(f"Starting MLflow run {run_id} for Optuna trial {trial.number}")
        
        # Log hyperparameters for this trial
        mlflow.log_params({
            "trial_number": trial.number,
            "pixel_loss_weight": pixel_loss_weight,
            "rec_feat_loss_weight": rec_feat_loss_weight,
            "adv_loss_weight": adv_loss_weight,
            "epochs": epochs,
            "batch_size": batch_size,
            "gpu_id": gpu_id,
            "no_restore": True, # Always start from scratch for HPO
            "contrastive_loss_weight": 0.0, # Disabled as per previous analysis
            "ctc_loss_weight": 1.0, # Monitoring only
            "early_stopping": True,
            "patience": 15,
            "min_delta": 0.01,
            "restore_best_weights": True,
            "seed": 42 + trial.number # Vary seed for each trial
        })
        
        # --- 3. Construct and Run the Training Command ---
        # Set environment variable for consistent experiment name
        env = os.environ.copy()
        env['MLFLOW_EXPERIMENT_NAME'] = MLFLOW_EXPERIMENT_NAME
        
        command = [
            "poetry", "run", "python", TRAIN_SCRIPT_PATH,
            "--epochs", str(epochs),
            "--batch_size", str(batch_size),
            "--gpu_id", gpu_id,
            "--recognizer_weights", recognizer_weights,
            "--pixel_loss_weight", str(pixel_loss_weight),
            "--rec_feat_loss_weight", str(rec_feat_loss_weight),
            "--adv_loss_weight", str(adv_loss_weight),
            "--contrastive_loss_weight", "0.0", # Explicitly disabled
            "--ctc_loss_weight", "1.0", # Monitoring only
            "--early_stopping",
            "--patience", "15",
            "--min_delta", "0.01",
            "--restore_best_weights",
            "--no_restore", # Ensure clean slate for each trial
            "--seed", str(42 + trial.number), # Vary seed for each trial
            # Direct output to a temporary log file to avoid cluttering stdout
            # and to capture potential errors.
            # We will not parse stdout for metrics, but rely on MLflow.
            "--checkpoint_dir", os.path.join(PROJECT_ROOT, "dual_modal_gan", "outputs", "hpo_checkpoints", f"trial_{trial.number}"),
            "--sample_dir", os.path.join(PROJECT_ROOT, "dual_modal_gan", "outputs", "hpo_samples", f"trial_{trial.number}")
        ]
        
        # Ensure output directories are clean for this trial
        # (train32.py already handles mkdir -p, but we want to ensure no old files)
        subprocess.run(["rm", "-rf", os.path.join(PROJECT_ROOT, "dual_modal_gan", "outputs", "hpo_checkpoints", f"trial_{trial.number}")])
        subprocess.run(["rm", "-rf", os.path.join(PROJECT_ROOT, "dual_modal_gan", "outputs", "hpo_samples", f"trial_{trial.number}")])

        # Set TF_GPU_ALLOCATOR environment variable for the subprocess
        env['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

        print(f"Executing command: {' '.join(command)}")
        process = subprocess.run(command, capture_output=True, text=True, cwd=PROJECT_ROOT, env=env)
        
        # Always print stdout and stderr for debugging
        print(f"--- Subprocess STDOUT for trial {trial.number} ---")
        print(process.stdout)
        print(f"--- Subprocess STDERR for trial {trial.number} ---")
        print(process.stderr)
        print(f"--- End Subprocess Output for trial {trial.number} ---")

        if process.returncode != 0:
            print(f"Error during training for trial {trial.number}:")
            print(process.stdout)
            print(process.stderr)
            raise optuna.exceptions.TrialPruned(f"Training failed with exit code {process.returncode}")

        # --- 4. Retrieve Metrics from MLflow ---
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        
        # Implement a robust retry mechanism for MLflow metric retrieval
        max_retries = 30
        retry_delay = 5 # seconds
        print(f"Waiting for MLflow metrics to become available for run {run_id}...")
        
        for i in range(max_retries):
            try:
                print(f"Attempt {i+1}/{max_retries}: Checking MLflow run {run_id}")
                run_data = client.get_run(run_id)
                
                # Print all available metrics for debugging
                available_metrics = list(run_data.data.metrics.keys())
                print(f"Available metrics: {available_metrics}")
                
                # Try multiple metric names that might contain validation data
                val_psnr = run_data.data.metrics.get("val/psnr") or \
                           run_data.data.metrics.get("best_val_psnr") or \
                           run_data.data.metrics.get("val_psnr")
                           
                val_cer = run_data.data.metrics.get("val/cer") or \
                          run_data.data.metrics.get("best_val_cer") or \
                          run_data.data.metrics.get("val_cer")
                          
                val_ssim = run_data.data.metrics.get("val/ssim") or \
                           run_data.data.metrics.get("best_val_ssim") or \
                           run_data.data.metrics.get("val_ssim")
                
                if val_psnr is not None and val_cer is not None and val_ssim is not None:
                    print(f"Metrics found: PSNR={val_psnr}, CER={val_cer}, SSIM={val_ssim}")
                    break # Metrics found, exit retry loop
                else:
                    print(f"Metrics not yet available. Trying metric history...")
                    
                    # Fallback to metric history if direct metrics not available
                    try:
                        val_psnr_history = client.get_metric_history(run_id, "val/psnr")
                        val_cer_history = client.get_metric_history(run_id, "val/cer") 
                        val_ssim_history = client.get_metric_history(run_id, "val/ssim")
                        
                        if val_psnr_history and val_cer_history and val_ssim_history:
                            val_psnr = val_psnr_history[-1].value
                            val_cer = val_cer_history[-1].value
                            val_ssim = val_ssim_history[-1].value
                            print(f"Metrics from history: PSNR={val_psnr}, CER={val_cer}, SSIM={val_ssim}")
                            break
                    except Exception as hist_e:
                        print(f"Metric history also not available yet: {hist_e}")
                        
            except Exception as e:
                print(f"Attempt {i+1}/{max_retries}: Error fetching MLflow run {run_id}: {e}")
            
            if i < max_retries - 1:  # Don't sleep after the last attempt
                print(f"Waiting {retry_delay} seconds before retry...")
                time.sleep(retry_delay)
        
        # Check if we found metrics after all retries
        if val_psnr is None or val_cer is None or val_ssim is None:
            print(f"Failed to retrieve metrics. Final values: PSNR={val_psnr}, CER={val_cer}, SSIM={val_ssim}")
            # Try to get experiment ID properly
            try:
                experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
                if experiment:
                    list_runs = client.search_runs(experiment_ids=[experiment.experiment_id])
                    print(f"Total runs in experiment '{MLFLOW_EXPERIMENT_NAME}': {len(list_runs)}")
                    for run in list_runs[:3]:  # Show first 3 runs for debugging
                        print(f"Run {run.info.run_id}: {run.info.status}, metrics: {list(run.data.metrics.keys())[:5]}")
                else:
                    print(f"Experiment '{MLFLOW_EXPERIMENT_NAME}' not found")
            except Exception as e:
                print(f"Error listing runs: {e}")
            raise optuna.exceptions.TrialPruned(f"Failed to retrieve MLflow metrics after {max_retries} retries.")

        final_val_psnr = float(val_psnr)
        final_val_cer = float(val_cer) 
        final_val_ssim = float(val_ssim)
        
        # --- 5. Calculate Objective Score ---
        # We want to maximize PSNR and SSIM, and minimize CER.
        # A simple combined score: PSNR + SSIM - (CER_weight * CER)
        # CER_weight is chosen to give CER significant importance.
        CER_WEIGHT = 100.0 # Adjust this based on how much you want to penalize CER
        objective_score = final_val_psnr + final_val_ssim - (CER_WEIGHT * final_val_cer)
        
        print(f"Trial {trial.number} results: PSNR={final_val_psnr:.2f}, SSIM={final_val_ssim:.4f}, CER={final_val_cer:.4f}, Score={objective_score:.2f}")
        
        # Log the objective score to MLflow as well
        mlflow.log_metric("objective_score", objective_score)
        
        return objective_score

# --- Main Optuna Study Execution ---
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run Bayesian optimization for GAN-HTR loss weights")
    parser.add_argument("--n_trials", type=int, default=1, help="Number of trials to run (default: 1)")
    args = parser.parse_args()
    
    # Create output directories for HPO trials
    os.makedirs(os.path.join(PROJECT_ROOT, "dual_modal_gan", "outputs", "hpo_checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "dual_modal_gan", "outputs", "hpo_samples"), exist_ok=True)

    # Set MLflow tracking URI for the study itself
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Create an Optuna study
    # We want to maximize the objective score
    study_db_path = os.path.join(PROJECT_ROOT, "scripts", "hpo", "optuna_study.db")
    study = optuna.create_study(direction="maximize", study_name="Loss_Weight_HPO",
                                storage=f"sqlite:///{study_db_path}", load_if_exists=True)
    
    print(f"Starting Optuna study '{study.study_name}' (persistent storage: {study_db_path}) with {MLFLOW_EXPERIMENT_NAME} experiment.")
    print(f"Number of trials to run: {args.n_trials}")
    
    # Run the optimization for a specified number of trials
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)
