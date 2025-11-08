#!/usr/bin/env python3
"""
Universal JSON Config Parser for GAN Training
Parses any JSON config format and outputs training command arguments
"""
import json
import sys

def get_value(config, keys, default=None):
    """Get value from nested dict using list of keys, fallback to flat structure"""
    # Try nested first
    current = config
    for key in keys[:-1]:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            # Fallback to flat structure with last key
            return config.get(keys[-1], default)
    return current.get(keys[-1], default) if isinstance(current, dict) else default

def parse_config(config_path):
    """Parse JSON config and return training arguments"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Extract experiment name
        exp_name = config.get('name', config.get('experiment_name', 'experiment'))
        
        # Build training arguments
        training_args = []
        
        # Model architecture
        gen_arch = config.get('generator_version', config.get('generator', 'enhanced'))
        if isinstance(gen_arch, dict):
            gen_arch = gen_arch.get('architecture', 'enhanced')
        
        disc_arch = config.get('discriminator_version', config.get('discriminator', 'enhanced_v2_fixed'))
        if isinstance(disc_arch, dict):
            disc_arch = disc_arch.get('architecture', 'enhanced_v2_fixed')
        
        training_args.append(f"--generator_version {gen_arch}")
        training_args.append(f"--discriminator_version {disc_arch}")
        
        # Data paths
        tfrecord = config.get('tfrecord_path', 'dual_modal_gan/data/dataset_gan.tfrecord')
        charset = config.get('charset_path', 'real_data_preparation/real_data_charlist.txt')
        recognizer = config.get('recognizer_weights', None)
        
        training_args.append(f"--tfrecord_path {tfrecord}")
        training_args.append(f"--charset_path {charset}")
        if recognizer and str(recognizer).lower() not in ['none', 'null']:
            training_args.append(f"--recognizer_weights {recognizer}")
        
        # GPU
        gpu_id = config.get('gpu_id', '0')
        training_args.append(f"--gpu_id {gpu_id}")
        
        # Directories
        checkpoint_dir = config.get('checkpoint_dir', f"dual_modal_gan/checkpoints/{exp_name}")
        sample_dir = config.get('sample_dir', f"dual_modal_gan/outputs/samples_{exp_name}")
        max_ckpts = config.get('max_checkpoints', 1)
        
        training_args.append(f"--checkpoint_dir {checkpoint_dir}")
        training_args.append(f"--sample_dir {sample_dir}")
        training_args.append(f"--max_checkpoints {max_ckpts}")
        
        # Best model saving
        checkpoints_cfg = config.get('checkpoints', {})
        if checkpoints_cfg.get('save_best_model_separately', True):
            training_args.append("--save_best_model_separately")
        
        # Training hyperparameters
        epochs = config.get('epochs', 1)
        steps = config.get('steps_per_epoch', 0)
        batch_size = config.get('batch_size', 2)
        save_interval = config.get('save_interval', 5)
        eval_interval = config.get('eval_interval', 1)
        seed = config.get('seed', 42)
        
        training_args.append(f"--epochs {epochs}")
        training_args.append(f"--steps_per_epoch {steps}")
        training_args.append(f"--batch_size {batch_size}")
        training_args.append(f"--save_interval {save_interval}")
        training_args.append(f"--eval_interval {eval_interval}")
        training_args.append(f"--seed {seed}")
        
        # Learning rates - CRITICAL FIX
        lr_g = config.get('lr_g', 0.0002)
        lr_d = config.get('lr_d', 0.0002)
        training_args.append(f"--lr_g {lr_g}")
        training_args.append(f"--lr_d {lr_d}")
        
        # Gradient clipping
        grad_clip = config.get('gradient_clip_norm', 1.0)
        training_args.append(f"--gradient_clip_norm {grad_clip}")
        
        # Loss weights
        pixel_w = config.get('pixel_loss_weight', 200.0)
        adv_w = config.get('adv_loss_weight', 1.5)
        recfeat_w = config.get('rec_feat_loss_weight', 0.0)
        ctc_w = config.get('ctc_loss_weight', 0.0)
        percep_w = config.get('perceptual_loss_weight', 0.0)
        
        training_args.append(f"--pixel_loss_weight {pixel_w}")
        training_args.append(f"--adv_loss_weight {adv_w}")
        training_args.append(f"--rec_feat_loss_weight {recfeat_w}")
        training_args.append(f"--ctc_loss_weight {ctc_w}")
        training_args.append(f"--perceptual_loss_weight {percep_w}")
        training_args.append(f"--ctc_loss_clip_max {config.get('ctc_loss_clip_max', 300.0)}")
        
        # Curriculum learning
        warmup = config.get('warmup_epochs', 0)
        annealing = config.get('annealing_epochs', 0)
        training_args.append(f"--warmup_epochs {warmup}")
        training_args.append(f"--annealing_epochs {annealing}")
        
        # Other settings
        disc_mode = config.get('discriminator_mode', 'predicted')
        cer_weight = config.get('cer_weight', 0.2)
        training_args.append(f"--discriminator_mode {disc_mode}")
        training_args.append(f"--cer_weight {cer_weight}")
        
        # Early stopping
        early_stop_cfg = config.get('early_stopping', {})
        if early_stop_cfg.get('enabled', False):
            training_args.append("--early_stopping")
            training_args.append(f"--patience {early_stop_cfg.get('patience', 15)}")
            training_args.append(f"--min_delta {early_stop_cfg.get('min_delta', 0.01)}")
            if early_stop_cfg.get('restore_best_weights', True):
                training_args.append("--restore_best_weights")
            
            metric = config.get('early_stopping_metric', 'combined')
            psnr_thresh = config.get('psnr_improvement_threshold', 2.0)
            training_args.append(f"--early_stopping_metric {metric}")
            training_args.append(f"--psnr_improvement_threshold {psnr_thresh}")
        
        # LR scheduling
        if config.get('use_lr_schedule', False):
            training_args.append("--use_lr_schedule")
            training_args.append(f"--lr_decay_epochs {config.get('lr_decay_epochs', 50)}")
            training_args.append(f"--lr_alpha {config.get('lr_alpha', 0.0)}")
        
        # Resume / pretrained
        if config.get('resume', False):
            training_args.append("--resume")
        
        if config.get('no_restore', False):
            training_args.append("--no_restore")
        
        pretrained = config.get('pretrained_checkpoint', None)
        if pretrained:
            training_args.append(f"--pretrained_checkpoint {pretrained}")
        
        # Print summary
        print("=" * 80)
        print("CONFIGURATION SUMMARY:")
        print("=" * 80)
        print(f"Experiment: {exp_name}")
        print(f"Description: {config.get('description', 'N/A')}")
        print("")
        print(f"Training:")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rates: G={lr_g}, D={lr_d}")
        print(f"  LR schedule: {config.get('use_lr_schedule', False)}")
        print("")
        print(f"Loss Weights:")
        print(f"  Pixel: {pixel_w}")
        print(f"  Adversarial: {adv_w}")
        print(f"  Perceptual: {percep_w}")
        print(f"  CTC: {ctc_w}")
        print(f"  RecFeat: {recfeat_w}")
        print("=" * 80)
        print("")
        
        # Output: experiment_name|command_args
        print(f"{exp_name}|{' '.join(training_args)}")
        
        return 0
        
    except Exception as e:
        print(f"ERROR: Failed to parse config: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: parse_training_config.py <config.json>", file=sys.stderr)
        sys.exit(1)
    
    sys.exit(parse_config(sys.argv[1]))
