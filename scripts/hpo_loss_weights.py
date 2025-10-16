#!/usr/bin/env python3
"""
Hyperparameter Optimization for Loss Weights using Optuna
Bayesian Optimization approach for efficient search in 2 hours

Optimizes:
- pixel_loss_weight
- rec_feat_loss_weight  
- contrastive_loss_weight

Target: Maximize PSNR + SSIM, Minimize CER/WER
Duration: ~30-40 trials in 30-40 minutes
"""

import argparse
import os
import sys
import subprocess
import json
from datetime import datetime
import optuna
import mlflow

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


def objective(trial, args):
    """
    Optuna objective function.
    
    Samples hyperparameters, runs training, returns objective score.
    """
    # Sample hyperparameters from search space
    pixel_loss_weight = trial.suggest_float('pixel_loss_weight', 50.0, 200.0, step=10.0)
    rec_feat_loss_weight = trial.suggest_float('rec_feat_loss_weight', 10.0, 100.0, step=5.0)
    adv_loss_weight = trial.suggest_float('adv_loss_weight', 1.0, 10.0, step=0.5)
    
    # Fixed parameters (not effective or monitoring only)
    contrastive_loss_weight = 0.0  # FAILED in ablation study - disabled
    ctc_loss_weight = 10.0  # Only for monitoring, not in backprop
    
    # Log trial info
    print(f"\n{'='*80}")
    print(f"Trial {trial.number + 1}/{args.n_trials}")
    print(f"{'='*80}")
    print(f"Hyperparameters:")
    print(f"  pixel_loss_weight:        {pixel_loss_weight:.1f}")
    print(f"  rec_feat_loss_weight:     {rec_feat_loss_weight:.1f}")
    print(f"  adv_loss_weight:          {adv_loss_weight:.1f}")
    print(f"  contrastive_loss_weight:  {contrastive_loss_weight:.1f} (disabled - failed in ablation)")
    print(f"  ctc_loss_weight:          {ctc_loss_weight:.1f} (monitoring only)")
    print(f"{'='*80}\n")
    
    # Prepare training command
    train_script = os.path.join(args.project_root, 'dual_modal_gan/scripts/train32.py')
    checkpoint_dir = os.path.join(args.project_root, f'dual_modal_gan/outputs/hpo_trial_{trial.number}')
    sample_dir = os.path.join(args.project_root, f'dual_modal_gan/outputs/hpo_samples_{trial.number}')
    
    # Create unique MLflow experiment name for this trial
    mlflow_exp_name = f"HPO_Trial_{trial.number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Determine which GPU to use for this trial
    gpu_id = trial.number % args.n_gpus  # Round-robin GPU assignment
    
    cmd = [
        'poetry', 'run', 'python', train_script,
        '--tfrecord_path', args.tfrecord_path,
        '--charset_path', args.charset_path,
        '--recognizer_weights', args.recognizer_weights,
        '--checkpoint_dir', checkpoint_dir,
        '--sample_dir', sample_dir,
        '--gpu_id', str(gpu_id),
        '--epochs', str(args.epochs),
        '--steps_per_epoch', str(args.steps_per_epoch),
        '--batch_size', str(args.batch_size),
        '--lr_g', str(args.lr_g),
        '--lr_d', str(args.lr_d),
        '--pixel_loss_weight', str(pixel_loss_weight),
        '--rec_feat_loss_weight', str(rec_feat_loss_weight),
        '--contrastive_loss_weight', str(contrastive_loss_weight),
        '--adv_loss_weight', str(adv_loss_weight),
        '--ctc_loss_weight', str(ctc_loss_weight),
        '--gradient_clip_norm', str(args.gradient_clip_norm),
        '--ctc_loss_clip_max', str(args.ctc_loss_clip_max),
        '--warmup_epochs', str(args.warmup_epochs),
        '--annealing_epochs', str(args.annealing_epochs),
        '--eval_interval', str(args.eval_interval),
        '--save_interval', str(args.save_interval),
        '--discriminator_mode', args.discriminator_mode,
        '--no_restore',  # Always start from scratch for fair comparison
        '--seed', str(args.seed),
        '--max_checkpoints', '1',  # Save space
    ]
    
    # Set environment variables for MLflow and GPU memory management
    env = os.environ.copy()
    env['MLFLOW_EXPERIMENT_NAME'] = mlflow_exp_name
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    env['ALLOW_GPU'] = '1'
    env['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    env['TF_CPP_MIN_LOG_LEVEL'] = '2'
    env['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'
    # Additional memory management
    env['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    env['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
    
    # Run training
    print(f"Running training on GPU {gpu_id}...")
    print(f"Command: {' '.join(cmd[:10])}... (truncated)\n")
    
    # Create log file for this trial
    trial_log = os.path.join(args.project_root, 'logbook', f'hpo_trial_{trial.number}.log')
    
    try:
        with open(trial_log, 'w') as log_f:
            result = subprocess.run(
                cmd,
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
                check=True,
                cwd=args.project_root,
                timeout=600  # 10 minute timeout per trial
            )
        
        print(f"✅ Training completed. Log: {trial_log}")
        
        # Parse metrics from output JSON file
        metrics_file = os.path.join(checkpoint_dir, 'metrics', 'training_metrics_fp32.json')
        
        if not os.path.exists(metrics_file):
            print(f"❌ Error: Metrics file not found: {metrics_file}")
            print(f"   Check training log: {trial_log}")
            raise optuna.exceptions.TrialPruned()
        
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        # Get validation metrics from last epoch
        last_epoch = metrics['epochs'][-1]
        
        if last_epoch.get('validation') is None:
            print(f"❌ Error: No validation metrics in last epoch")
            raise optuna.exceptions.TrialPruned()
        
        val_metrics = last_epoch['validation']
        psnr = val_metrics['psnr']
        ssim = val_metrics['ssim']
        cer = val_metrics['cer']
        wer = val_metrics['wer']
        
        # Calculate combined objective score
        # Maximize: PSNR (target ~40) + SSIM (target ~0.99)
        # Minimize: CER + WER
        # Normalized scoring:
        # - PSNR: normalize to [0, 1] assuming range [20, 50]
        # - SSIM: already in [0, 1]
        # - CER: penalize, normalize assuming range [0, 1]
        # - WER: penalize, normalize assuming range [0, 1]
        
        psnr_normalized = min(max((psnr - 20) / 30, 0), 1)  # Map [20, 50] to [0, 1]
        ssim_normalized = ssim  # Already [0, 1]
        cer_penalty = cer  # Lower is better
        wer_penalty = wer  # Lower is better
        
        # Combined score: maximize visual quality, minimize text error
        # Weight: 40% PSNR, 40% SSIM, 10% CER, 10% WER
        objective_score = (
            0.4 * psnr_normalized + 
            0.4 * ssim_normalized - 
            0.1 * cer_penalty - 
            0.1 * wer_penalty
        )
        
        print(f"\n{'='*80}")
        print(f"Trial {trial.number + 1} Results:")
        print(f"{'='*80}")
        print(f"Validation Metrics:")
        print(f"  PSNR:           {psnr:.2f}")
        print(f"  SSIM:           {ssim:.4f}")
        print(f"  CER:            {cer:.4f}")
        print(f"  WER:            {wer:.4f}")
        print(f"\nObjective Score:  {objective_score:.4f}")
        print(f"{'='*80}\n")
        
        # Report intermediate values to Optuna for pruning
        trial.set_user_attr('psnr', psnr)
        trial.set_user_attr('ssim', ssim)
        trial.set_user_attr('cer', cer)
        trial.set_user_attr('wer', wer)
        trial.set_user_attr('gpu_id', gpu_id)
        trial.set_user_attr('checkpoint_dir', checkpoint_dir)
        
        # Clean up checkpoint to save space (keep only best trial)
        if not args.keep_all_checkpoints:
            import shutil
            if os.path.exists(checkpoint_dir):
                shutil.rmtree(checkpoint_dir)
                print(f"🗑️  Cleaned up checkpoint: {checkpoint_dir}")
        
        # Force garbage collection to free GPU memory
        import gc
        gc.collect()
        
        return objective_score
        
    except subprocess.TimeoutExpired:
        print(f"❌ Training timed out after 10 minutes")
        print(f"   Check training log: {trial_log}")
        raise optuna.exceptions.TrialPruned()
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with error:")
        print(f"   Return code: {e.returncode}")
        print(f"   Check training log: {trial_log}")
        raise optuna.exceptions.TrialPruned()
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        raise optuna.exceptions.TrialPruned()


def main():
    parser = argparse.ArgumentParser(description='HPO for Loss Weights using Optuna')
    
    # Paths
    parser.add_argument('--project_root', type=str, 
                       default='/home/lambda_one/tesis/GAN-HTR-ORI/docRestoration',
                       help='Project root directory')
    parser.add_argument('--tfrecord_path', type=str,
                       default='dual_modal_gan/data/dataset_gan.tfrecord')
    parser.add_argument('--charset_path', type=str,
                       default='real_data_preparation/real_data_charlist.txt')
    parser.add_argument('--recognizer_weights', type=str,
                       default='models/best_htr_recognizer/best_model.weights.h5')
    
    # HPO settings
    parser.add_argument('--n_trials', type=int, default=30,
                       help='Number of Optuna trials (default: 30)')
    parser.add_argument('--n_gpus', type=int, default=2,
                       help='Number of GPUs available (default: 2)')
    parser.add_argument('--study_name', type=str, 
                       default=f'loss_weights_hpo_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                       help='Optuna study name')
    parser.add_argument('--storage', type=str,
                       default='sqlite:///hpo_study.db',
                       help='Optuna storage URL')
    parser.add_argument('--keep_all_checkpoints', action='store_true',
                       help='Keep checkpoints from all trials (requires more disk space)')
    
    # Training settings (minimal for speed)
    parser.add_argument('--epochs', type=int, default=2,
                       help='Epochs per trial (default: 2 for speed)')
    parser.add_argument('--steps_per_epoch', type=int, default=30,
                       help='Steps per epoch (default: 30 for speed)')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Batch size (default: 2 for memory efficiency)')
    parser.add_argument('--lr_g', type=float, default=0.0004)
    parser.add_argument('--lr_d', type=float, default=0.0004)
    parser.add_argument('--gradient_clip_norm', type=float, default=1.0)
    parser.add_argument('--ctc_loss_clip_max', type=float, default=100.0)
    parser.add_argument('--warmup_epochs', type=int, default=1,
                       help='Warmup epochs (default: 1 for speed)')
    parser.add_argument('--annealing_epochs', type=int, default=1,
                       help='Annealing epochs (default: 1 for speed)')
    parser.add_argument('--eval_interval', type=int, default=2,
                       help='Evaluate only at end (default: 2)')
    parser.add_argument('--save_interval', type=int, default=2,
                       help='Save samples only at end (default: 2)')
    parser.add_argument('--discriminator_mode', type=str, default='ground_truth')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Convert relative paths to absolute
    if not os.path.isabs(args.tfrecord_path):
        args.tfrecord_path = os.path.join(args.project_root, args.tfrecord_path)
    if not os.path.isabs(args.charset_path):
        args.charset_path = os.path.join(args.project_root, args.charset_path)
    if not os.path.isabs(args.recognizer_weights):
        args.recognizer_weights = os.path.join(args.project_root, args.recognizer_weights)
    
    print("\n" + "="*80)
    print("🔬 HYPERPARAMETER OPTIMIZATION - Loss Weights")
    print("="*80)
    print(f"Method:           Bayesian Optimization (Optuna TPE)")
    print(f"Study Name:       {args.study_name}")
    print(f"Number of Trials: {args.n_trials}")
    print(f"GPUs Available:   {args.n_gpus}")
    print(f"Storage:          {args.storage}")
    print(f"\nSearch Space:")
    print(f"  pixel_loss_weight:       [50.0, 200.0] step=10.0")
    print(f"  rec_feat_loss_weight:    [10.0, 100.0] step=5.0")
    print(f"  adv_loss_weight:         [1.0, 10.0] step=0.5")
    print(f"\nFixed Parameters:")
    print(f"  contrastive_loss_weight: 0.0 (DISABLED - failed in ablation study)")
    print(f"  ctc_loss_weight:         10.0 (monitoring only, not in backprop)")
    print(f"\nTraining Config (per trial):")
    print(f"  Epochs:          {args.epochs}")
    print(f"  Steps/epoch:     {args.steps_per_epoch}")
    print(f"  Batch size:      {args.batch_size}")
    print(f"  Warmup:          {args.warmup_epochs} epochs")
    print(f"  Annealing:       {args.annealing_epochs} epochs")
    print(f"\nObjective:")
    print(f"  Maximize: 0.4*PSNR_norm + 0.4*SSIM - 0.1*CER - 0.1*WER")
    print(f"  Target: PSNR ~40, SSIM ~0.99, CER/WER minimal")
    print("="*80 + "\n")
    
    # Create Optuna study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        load_if_exists=True
    )
    
    # Run optimization
    print(f"🚀 Starting optimization with {args.n_trials} trials...\n")
    
    try:
        study.optimize(
            lambda trial: objective(trial, args),
            n_trials=args.n_trials,
            show_progress_bar=True,
            gc_after_trial=True  # Garbage collect after each trial
        )
    except KeyboardInterrupt:
        print("\n⚠️  Optimization interrupted by user")
    
    # Print results
    print("\n" + "="*80)
    print("✅ OPTIMIZATION COMPLETED")
    print("="*80)
    
    # Check if we have any completed trials
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    if len(completed_trials) == 0:
        print("\n⚠️  WARNING: No trials completed successfully!")
        print("   All trials were pruned or failed.")
        print("   Please check the error logs and try again with adjusted parameters.")
        return
    
    print(f"\nCompleted Trials: {len(completed_trials)}/{len(study.trials)}")
    print(f"\nBest Trial: #{study.best_trial.number}")
    print(f"Best Objective Score: {study.best_trial.value:.4f}")
    print(f"\n🏆 Best Hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key:30s}: {value}")
    
    print(f"\n📊 Best Validation Metrics:")
    print(f"  PSNR: {study.best_trial.user_attrs.get('psnr', 'N/A')}")
    print(f"  SSIM: {study.best_trial.user_attrs.get('ssim', 'N/A')}")
    print(f"  CER:  {study.best_trial.user_attrs.get('cer', 'N/A')}")
    print(f"  WER:  {study.best_trial.user_attrs.get('wer', 'N/A')}")
    
    # Save results to JSON
    results_file = os.path.join(args.project_root, 'logbook', 
                                f'hpo_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    
    results = {
        'study_name': args.study_name,
        'n_trials': args.n_trials,
        'best_trial': {
            'number': study.best_trial.number,
            'objective_score': study.best_trial.value,
            'params': study.best_trial.params,
            'metrics': {
                'psnr': study.best_trial.user_attrs.get('psnr'),
                'ssim': study.best_trial.user_attrs.get('ssim'),
                'cer': study.best_trial.user_attrs.get('cer'),
                'wer': study.best_trial.user_attrs.get('wer'),
            }
        },
        'all_trials': []
    }
    
    # Add all trials info
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            results['all_trials'].append({
                'number': trial.number,
                'objective_score': trial.value,
                'params': trial.params,
                'metrics': {
                    'psnr': trial.user_attrs.get('psnr'),
                    'ssim': trial.user_attrs.get('ssim'),
                    'cer': trial.user_attrs.get('cer'),
                    'wer': trial.user_attrs.get('wer'),
                }
            })
    
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    print("="*80 + "\n")
    
    # Print importance of hyperparameters
    if len(study.trials) > 10:
        print("\n📈 Hyperparameter Importance:")
        try:
            importances = optuna.importance.get_param_importances(study)
            for param, importance in importances.items():
                print(f"  {param:30s}: {importance:.4f}")
        except Exception as e:
            print(f"  Could not calculate importances: {e}")
    
    print("\n🎯 Next Steps:")
    print(f"  1. Review results in: {results_file}")
    print(f"  2. Use best hyperparameters for full training")
    print(f"  3. Visualize with: poetry run optuna-dashboard {args.storage}")
    print(f"  4. Update logbook with findings\n")


if __name__ == '__main__':
    main()
