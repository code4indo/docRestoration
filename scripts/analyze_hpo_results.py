#!/usr/bin/env python3
"""
Script to analyze Bayesian optimization results from Optuna study
"""

import optuna
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import json
from datetime import datetime
import mlflow
from mlflow.tracking import MlflowClient

def load_optuna_study(study_db_path):
    """Load Optuna study from database"""
    print(f"Loading Optuna study from: {study_db_path}")
    study = optuna.load_study(
        study_name="Loss_Weight_HPO",
        storage=f"sqlite:///{study_db_path}"
    )
    return study

def get_study_summary(study):
    """Get comprehensive study summary"""
    print("\n" + "="*60)
    print("STUDY SUMMARY")
    print("="*60)
    
    # Basic info
    print(f"Study Name: {study.study_name}")
    print(f"Total Trials: {len(study.trials)}")
    
    # Check if there are completed trials
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    if completed_trials:
        print(f"Completed Trials: {len(completed_trials)}")
        print(f"Best Trial: {study.best_trial.number}")
        print(f"Best Value: {study.best_value:.4f}")
        print(f"Best Parameters: {study.best_params}")
    else:
        print("No completed trials found")
        print("Best Trial: N/A")
        print("Best Value: N/A")
        print("Best Parameters: N/A")
    
    # Trial status breakdown
    trials_df = study.trials_dataframe()
    status_counts = trials_df['state'].value_counts()
    print(f"\nTrial Status Breakdown:")
    for status, count in status_counts.items():
        print(f"  {status}: {count}")
    
    return study, trials_df

def plot_optimization_history(study, output_dir):
    """Plot optimization history"""
    plt.figure(figsize=(12, 6))
    
    # Get trial data
    trials = study.trials
    trial_numbers = []
    values = []
    
    for trial in trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            trial_numbers.append(trial.number)
            values.append(trial.value)
    
    if not trial_numbers:
        print("No completed trials to plot")
        return
    
    plt.plot(trial_numbers, values, 'bo-', alpha=0.7)
    plt.axhline(y=study.best_value, color='r', linestyle='--', label=f'Best: {study.best_value:.4f}')
    plt.xlabel('Trial Number')
    plt.ylabel('Objective Score')
    plt.title('Optimization History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plot_path = os.path.join(output_dir, 'optimization_history.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Optimization history plot saved to: {plot_path}")

def plot_parameter_importance(study, output_dir):
    """Plot parameter importance"""
    try:
        # Get parameter importance
        importance = optuna.importance.get_param_importances(study)
        
        # Plot
        plt.figure(figsize=(10, 6))
        importance_df = pd.DataFrame(list(importance.items()), columns=['Parameter', 'Importance'])
        importance_df = importance_df.sort_values('Importance', ascending=True)
        
        plt.barh(importance_df['Parameter'], importance_df['Importance'])
        plt.xlabel('Importance')
        plt.title('Parameter Importance')
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, 'parameter_importance.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Parameter importance plot saved to: {plot_path}")
        
    except Exception as e:
        print(f"Could not plot parameter importance: {e}")

def plot_parameter_relationships(study, output_dir):
    """Plot parameter relationships"""
    try:
        # Get trials dataframe
        trials_df = study.trials_dataframe()
        completed_trials = trials_df[trials_df['state'] == 'COMPLETE']
        
        if len(completed_trials) < 2:
            print("Not enough completed trials for parameter relationship plots")
            return
        
        # Get parameter columns
        param_cols = [col for col in completed_trials.columns if col.startswith('params_')]
        param_names = [col.replace('params_', '') for col in param_cols]
        
        if len(param_names) < 2:
            print("Not enough parameters for relationship plots")
            return
        
        # Create pair plots
        fig, axes = plt.subplots(len(param_names), len(param_names), figsize=(15, 12))
        if len(param_names) == 1:
            axes = np.array([[axes]])
        
        for i, param1 in enumerate(param_names):
            for j, param2 in enumerate(param_names):
                ax = axes[i, j] if len(param_names) > 1 else axes
                
                if i == j:
                    # Diagonal: histogram
                    ax.hist(completed_trials[f'params_{param1}'], bins=20, alpha=0.7)
                    ax.set_title(f'{param1} Distribution')
                else:
                    # Off-diagonal: scatter plot
                    ax.scatter(
                        completed_trials[f'params_{param1}'],
                        completed_trials[f'params_{param2}'],
                        c=completed_trials['value'],
                        cmap='viridis',
                        alpha=0.7
                    )
                    ax.set_xlabel(param1)
                    ax.set_ylabel(param2)
                    ax.set_title(f'{param1} vs {param2}')
                
                if i == len(param_names) - 1:
                    ax.set_xlabel(param2)
                if j == 0:
                    ax.set_ylabel(param1)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'parameter_relationships.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Parameter relationships plot saved to: {plot_path}")
        
    except Exception as e:
        print(f"Could not plot parameter relationships: {e}")

def generate_best_config_report(study, output_dir):
    """Generate report with best configuration"""
    # Check if there are completed trials
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    if not completed_trials:
        print("No completed trials - cannot generate best configuration report")
        return None
    
    best_trial = study.best_trial
    
    report = {
        "study_info": {
            "study_name": study.study_name,
            "direction": str(study.direction),
            "total_trials": len(study.trials),
            "completed_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            "best_trial_number": best_trial.number,
            "best_value": float(best_trial.value),
            "generation_time": datetime.now().isoformat()
        },
        "best_parameters": best_trial.params,
        "trial_statistics": {
            "mean_value": float(np.mean([t.value for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])),
            "std_value": float(np.std([t.value for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])),
            "min_value": float(study.best_value),
            "max_value": float(max([t.value for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]))
        }
    }
    
    # Save report
    report_path = os.path.join(output_dir, 'best_config_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Best configuration report saved to: {report_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("BEST CONFIGURATION")
    print("="*60)
    print(f"Best Trial: {best_trial.number}")
    print(f"Objective Score: {best_trial.value:.4f}")
    print("\nOptimal Loss Weights:")
    for param, value in best_trial.params.items():
        print(f"  {param}: {value}")
    
    return report

def check_mlflow_runs(mlflow_uri, experiment_name="HPO_Loss_Weights"):
    """Check MLflow runs for HPO experiment"""
    try:
        client = MlflowClient(tracking_uri=mlflow_uri)
        experiment = client.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            print(f"MLflow experiment '{experiment_name}' not found")
            return None
        
        runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        completed_runs = [r for r in runs if r.info.status == 'FINISHED']
        
        print(f"\nMLflow Experiment: {experiment_name}")
        print(f"Total Runs: {len(runs)}")
        print(f"Completed Runs: {len(completed_runs)}")
        
        if completed_runs:
            best_run = max(completed_runs, key=lambda r: r.data.metrics.get('objective_score', -float('inf')))
            print(f"Best Run: {best_run.info.run_id}")
            print(f"Best Objective Score: {best_run.data.metrics.get('objective_score', 'N/A')}")
            
            # Print best run metrics
            metrics = best_run.data.metrics
            print("\nBest Run Metrics:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value}")
        
        return runs
        
    except Exception as e:
        print(f"Error checking MLflow runs: {e}")
        return None

def main():
    # Configuration
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    study_db_path = os.path.join(project_root, "scripts", "hpo", "optuna_study.db")
    mlflow_uri = f"file:{os.path.join(project_root, 'mlruns')}"
    output_dir = os.path.join(project_root, "dual_modal_gan", "outputs", "hpo_analysis")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Bayesian Optimization Results Analysis")
    print("=" * 60)
    print(f"Project Root: {project_root}")
    print(f"Study Database: {study_db_path}")
    print(f"MLflow URI: {mlflow_uri}")
    print(f"Output Directory: {output_dir}")
    
    # Load and analyze study
    if not os.path.exists(study_db_path):
        print(f"Study database not found: {study_db_path}")
        return
    
    study = load_optuna_study(study_db_path)
    study, trials_df = get_study_summary(study)
    
    # Generate plots
    print("\nGenerating analysis plots...")
    plot_optimization_history(study, output_dir)
    plot_parameter_importance(study, output_dir)
    plot_parameter_relationships(study, output_dir)
    
    # Generate best configuration report
    report = generate_best_config_report(study, output_dir)
    
    # Check MLflow runs
    print("\nChecking MLflow runs...")
    mlflow_runs = check_mlflow_runs(mlflow_uri)
    
    # Save trials data
    trials_csv_path = os.path.join(output_dir, 'trials_data.csv')
    trials_df.to_csv(trials_csv_path, index=False)
    print(f"\nTriials data saved to: {trials_csv_path}")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Analysis results saved to: {output_dir}")
    print(f"Next steps:")
    print("  1. Review the plots and reports")
    print("  2. Use the best configuration for full training")
    print("  3. Run: poetry run python scripts/hpo/monitor_hpo.py")
    print("  4. View MLflow: poetry run mlflow ui")

if __name__ == "__main__":
    main()