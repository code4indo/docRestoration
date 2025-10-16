#!/usr/bin/env python3
"""
Monitoring script for Bayesian Optimization HPO process
"""

import os
import subprocess
import time
import json
from datetime import datetime
import mlflow
from mlflow.tracking import MlflowClient

def check_process_status():
    """Check if objective.py process is still running"""
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        lines = result.stdout.split('\n')
        for line in lines:
            if 'objective.py' in line and 'grep' not in line:
                parts = line.split()
                if len(parts) > 10:
                    pid = parts[1]
                    cpu_percent = parts[2]
                    mem_percent = parts[3]
                    runtime = parts[9]
                    return {
                        'running': True,
                        'pid': pid,
                        'cpu_percent': cpu_percent,
                        'mem_percent': mem_percent,
                        'runtime': runtime
                    }
        return {'running': False}
    except Exception as e:
        return {'running': False, 'error': str(e)}

def check_mlflow_runs():
    """Check MLflow runs for HPO experiment"""
    try:
        PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        MLFLOW_TRACKING_URI = f"file:{os.path.join(PROJECT_ROOT, 'mlruns')}"
        MLFLOW_EXPERIMENT_NAME = "HPO_Loss_Weights"
        
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        
        if experiment is None:
            return {'experiment_exists': False}
        
        runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        
        completed_runs = [r for r in runs if r.info.status == 'FINISHED']
        running_runs = [r for r in runs if r.info.status == 'RUNNING']
        failed_runs = [r for r in runs if r.info.status == 'FAILED']
        
        return {
            'experiment_exists': True,
            'total_runs': len(runs),
            'completed_runs': len(completed_runs),
            'running_runs': len(running_runs),
            'failed_runs': len(failed_runs),
            'latest_run': runs[-1].info if runs else None
        }
    except Exception as e:
        return {'error': str(e)}

def main():
    print(f"=== HPO Monitoring - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    
    # Check process status
    process_status = check_process_status()
    print(f"Process Status: {'RUNNING' if process_status.get('running', False) else 'NOT RUNNING'}")
    
    if process_status.get('running'):
        print(f"  PID: {process_status.get('pid')}")
        print(f"  CPU: {process_status.get('cpu_percent')}%")
        print(f"  Memory: {process_status.get('mem_percent')}%")
        print(f"  Runtime: {process_status.get('runtime')}")
    
    # Check MLflow runs
    mlflow_status = check_mlflow_runs()
    if 'error' in mlflow_status:
        print(f"MLflow Error: {mlflow_status['error']}")
    else:
        print(f"MLflow Experiment: {'EXISTS' if mlflow_status.get('experiment_exists', False) else 'NOT FOUND'}")
        if mlflow_status.get('experiment_exists'):
            print(f"  Total Runs: {mlflow_status.get('total_runs', 0)}")
            print(f"  Completed: {mlflow_status.get('completed_runs', 0)}")
            print(f"  Running: {mlflow_status.get('running_runs', 0)}")
            print(f"  Failed: {mlflow_status.get('failed_runs', 0)}")
            
            if mlflow_status.get('latest_run'):
                latest = mlflow_status['latest_run']
                print(f"  Latest Run: {latest.run_id}")
                print(f"    Status: {latest.status}")
                print(f"    Start Time: {latest.start_time}")

if __name__ == "__main__":
    main()