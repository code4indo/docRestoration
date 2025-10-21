#!/usr/bin/env python3
"""
Investigasi Detail: Generator Output Range Limitation
Root Cause Analysis untuk PSNR Gap

Menginvestigasi mengapa model hanya output [0.468, 1.0] instead of [0, 1]
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GeneratorOutputAnalyzer:
    """Analyze generator behavior and output range"""
    
    def __init__(self, checkpoint_path, mlruns_path='./mlruns', run_id=None):
        self.checkpoint_path = checkpoint_path
        self.mlruns_path = mlruns_path
        self.run_id = run_id
        self.model = None
        
    def load_model(self):
        """Load generator model from checkpoint"""
        logger.info(f"Loading model from: {self.checkpoint_path}")
        
        # Import generator architecture
        from dual_modal_gan.src.models.generator_enhanced import unet_enhanced
        
        # Create model with correct input shape
        self.model = unet_enhanced(input_size=(1024, 128, 1))
        
        # Load weights
        checkpoint = tf.train.Checkpoint(generator=self.model)
        checkpoint.restore(self.checkpoint_path).expect_partial()
        
        logger.info("✅ Model loaded successfully")
        
    def inspect_architecture(self):
        """Inspect generator architecture for range-limiting components"""
        logger.info("\n" + "="*60)
        logger.info("ARCHITECTURE INSPECTION")
        logger.info("="*60)
        
        # Model summary
        logger.info("\nModel Summary:")
        self.model.summary(print_fn=logger.info)
        
        # Find final layers
        logger.info("\n📌 CRITICAL LAYERS (Last 5):")
        for i, layer in enumerate(self.model.layers[-5:]):
            logger.info(f"\nLayer {len(self.model.layers) - 5 + i}: {layer.name}")
            logger.info(f"  Type: {layer.__class__.__name__}")
            logger.info(f"  Output shape: {layer.output_shape}")
            
            # Check activation
            if hasattr(layer, 'activation'):
                activation = layer.activation
                logger.info(f"  🔥 Activation: {activation}")
            
            # Check batch normalization
            if 'batch_norm' in layer.name.lower():
                logger.info(f"  ⚠️ Batch Normalization detected")
                if hasattr(layer, 'moving_mean'):
                    mean = layer.moving_mean.numpy()
                    logger.info(f"     Moving mean: [{mean.min():.3f}, {mean.max():.3f}]")
                if hasattr(layer, 'moving_variance'):
                    var = layer.moving_variance.numpy()
                    logger.info(f"     Moving variance: [{var.min():.3f}, {var.max():.3f}]")
        
        # Find all activation functions
        logger.info("\n📋 ALL ACTIVATIONS IN MODEL:")
        activations = {}
        for layer in self.model.layers:
            if hasattr(layer, 'activation'):
                act_name = str(layer.activation).split()[1] if 'function' in str(layer.activation) else str(layer.activation)
                activations[act_name] = activations.get(act_name, 0) + 1
        
        for act, count in sorted(activations.items()):
            logger.info(f"  {act}: {count} layers")
    
    def test_input_output_mapping(self, num_tests=20):
        """Test how model maps different input ranges to output"""
        logger.info("\n" + "="*60)
        logger.info("INPUT-OUTPUT MAPPING TEST")
        logger.info("="*60)
        
        results = []
        
        # Test different input patterns
        test_cases = [
            ("All Black", np.zeros((1024, 128, 1))),
            ("All White", np.ones((1024, 128, 1))),
            ("Mid Gray", np.ones((1024, 128, 1)) * 0.5),
            ("Random Uniform", np.random.uniform(0, 1, (1024, 128, 1))),
            ("Random Normal", np.clip(np.random.normal(0.5, 0.2, (1024, 128, 1)), 0, 1)),
        ]
        
        # Add gradient tests
        for intensity in np.linspace(0, 1, 10):
            test_cases.append((f"Constant {intensity:.2f}", np.ones((1024, 128, 1)) * intensity))
        
        for name, test_input in test_cases:
            # Normalize to [-1, 1] as model expects
            test_input_norm = (test_input * 2.0) - 1.0
            test_tensor = tf.convert_to_tensor(test_input_norm[np.newaxis, ...].transpose(0, 2, 1, 3), dtype=tf.float32)
            
            # Get output
            output = self.model(test_tensor, training=False)
            output_np = output.numpy()[0, ..., 0]
            
            # Denormalize output (assuming tanh: [-1,1] -> [0,1])
            output_denorm = (output_np + 1.0) / 2.0
            
            # Calculate statistics
            stats = {
                'name': name,
                'input_min': float(test_input.min()),
                'input_max': float(test_input.max()),
                'input_mean': float(test_input.mean()),
                'output_min': float(output_denorm.min()),
                'output_max': float(output_denorm.max()),
                'output_mean': float(output_denorm.mean()),
                'output_std': float(output_denorm.std()),
                'range_utilization': float(output_denorm.max() - output_denorm.min())
            }
            
            results.append(stats)
            
            logger.info(f"\n{name}:")
            logger.info(f"  Input:  [{stats['input_min']:.3f}, {stats['input_max']:.3f}] mean={stats['input_mean']:.3f}")
            logger.info(f"  Output: [{stats['output_min']:.3f}, {stats['output_max']:.3f}] mean={stats['output_mean']:.3f} std={stats['output_std']:.3f}")
            logger.info(f"  Range utilization: {stats['range_utilization']:.3f}")
        
        # Find actual achievable range
        all_mins = [r['output_min'] for r in results]
        all_maxs = [r['output_max'] for r in results]
        
        logger.info("\n" + "="*60)
        logger.info("🔥 RANGE ANALYSIS:")
        logger.info(f"Absolute min output achieved: {min(all_mins):.4f}")
        logger.info(f"Absolute max output achieved: {max(all_maxs):.4f}")
        logger.info(f"Achievable range: [{min(all_mins):.4f}, {max(all_maxs):.4f}]")
        logger.info(f"Missing lower range: [0.0000, {min(all_mins):.4f}] = {min(all_mins)*100:.1f}%")
        logger.info(f"Missing upper range: [{max(all_maxs):.4f}, 1.0000] = {(1-max(all_maxs))*100:.1f}%")
        logger.info("="*60)
        
        return results
    
    def analyze_layer_outputs(self, test_input):
        """Analyze intermediate layer outputs to find where range gets limited"""
        logger.info("\n" + "="*60)
        logger.info("LAYER-BY-LAYER OUTPUT RANGE")
        logger.info("="*60)
        
        # Normalize input
        test_input_norm = (test_input * 2.0) - 1.0
        test_tensor = tf.convert_to_tensor(test_input_norm[np.newaxis, ...].transpose(0, 2, 1, 3), dtype=tf.float32)
        
        # Create intermediate models
        important_layers = [0, len(self.model.layers)//4, len(self.model.layers)//2, 
                           3*len(self.model.layers)//4, len(self.model.layers)-1]
        
        for idx in important_layers:
            layer = self.model.layers[idx]
            intermediate_model = tf.keras.Model(inputs=self.model.input, outputs=layer.output)
            output = intermediate_model(test_tensor, training=False).numpy()
            
            logger.info(f"\nLayer {idx}: {layer.name}")
            logger.info(f"  Type: {layer.__class__.__name__}")
            logger.info(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")
            logger.info(f"  Output mean: {output.mean():.4f}, std: {output.std():.4f}")
            
            if hasattr(layer, 'activation'):
                logger.info(f"  Activation: {layer.activation}")
    
    def plot_response_curve(self, output_dir='output_range_investigation'):
        """Plot input-output response curve"""
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info("\n" + "="*60)
        logger.info("GENERATING RESPONSE CURVES")
        logger.info("="*60)
        
        # Test with varying input intensities
        intensities = np.linspace(0, 1, 50)
        output_means = []
        output_mins = []
        output_maxs = []
        
        for intensity in intensities:
            test_input = np.ones((1024, 128, 1)) * intensity
            test_input_norm = (test_input * 2.0) - 1.0
            test_tensor = tf.convert_to_tensor(test_input_norm[np.newaxis, ...].transpose(0, 2, 1, 3), dtype=tf.float32)
            
            output = self.model(test_tensor, training=False)
            output_np = output.numpy()[0, ..., 0]
            output_denorm = (output_np + 1.0) / 2.0
            
            output_means.append(output_denorm.mean())
            output_mins.append(output_denorm.min())
            output_maxs.append(output_denorm.max())
        
        # Plot
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(intensities, output_means, 'b-', linewidth=2, label='Output Mean')
        plt.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Ideal (y=x)')
        plt.xlabel('Input Intensity')
        plt.ylabel('Output Intensity (Mean)')
        plt.title('Input-Output Response Curve')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.fill_between(intensities, output_mins, output_maxs, alpha=0.3, label='Output Range')
        plt.plot(intensities, output_means, 'b-', linewidth=2, label='Output Mean')
        plt.xlabel('Input Intensity')
        plt.ylabel('Output Intensity')
        plt.title('Output Range per Input')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(2, 2, 3)
        output_range = np.array(output_maxs) - np.array(output_mins)
        plt.plot(intensities, output_range, 'g-', linewidth=2)
        plt.xlabel('Input Intensity')
        plt.ylabel('Output Dynamic Range')
        plt.title('Dynamic Range Utilization')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        plt.hist(output_means, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Output Mean Value')
        plt.ylabel('Frequency')
        plt.title('Distribution of Output Means')
        plt.axvline(x=0.468, color='r', linestyle='--', label='Observed Min (0.468)')
        plt.axvline(x=1.0, color='r', linestyle='--', label='Observed Max (1.0)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'response_curves.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"✅ Response curves saved: {output_path}")
        plt.close()
        
        return {
            'intensities': intensities.tolist(),
            'output_means': output_means,
            'output_mins': output_mins,
            'output_maxs': output_maxs
        }
    
    def diagnose_range_limitation(self):
        """Main diagnostic function"""
        logger.info("\n" + "="*60)
        logger.info("🔍 GENERATOR OUTPUT RANGE DIAGNOSTIC")
        logger.info("="*60)
        
        # 1. Load model
        self.load_model()
        
        # 2. Inspect architecture
        self.inspect_architecture()
        
        # 3. Test input-output mapping
        mapping_results = self.test_input_output_mapping()
        
        # 4. Analyze layer outputs
        test_input = np.random.uniform(0, 1, (1024, 128, 1))
        self.analyze_layer_outputs(test_input)
        
        # 5. Plot response curves
        response_data = self.plot_response_curve()
        
        # 6. Save results
        output_dir = 'output_range_investigation'
        os.makedirs(output_dir, exist_ok=True)
        
        results = {
            'checkpoint': self.checkpoint_path,
            'run_id': self.run_id,
            'mapping_tests': mapping_results,
            'response_curve': response_data,
            'conclusions': {
                'achievable_min': float(min(r['output_min'] for r in mapping_results)),
                'achievable_max': float(max(r['output_max'] for r in mapping_results)),
                'missing_lower_percentage': float(min(r['output_min'] for r in mapping_results) * 100),
                'missing_upper_percentage': float((1 - max(r['output_max'] for r in mapping_results)) * 100)
            }
        }
        
        with open(os.path.join(output_dir, 'diagnostic_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n✅ Diagnostic complete! Results saved to {output_dir}/")
        
        return results


def main():
    """Main execution"""
    checkpoint_path = './models/full_training_production_v1/best_model/ckpt-70'
    run_id = '704979cb9bbe40c3aeb4201f65141990'
    
    analyzer = GeneratorOutputAnalyzer(checkpoint_path, run_id=run_id)
    results = analyzer.diagnose_range_limitation()
    
    # Print summary
    print("\n" + "="*60)
    print("🎯 DIAGNOSTIC SUMMARY")
    print("="*60)
    print(f"Achievable output range: [{results['conclusions']['achievable_min']:.4f}, {results['conclusions']['achievable_max']:.4f}]")
    print(f"Missing lower range: {results['conclusions']['missing_lower_percentage']:.1f}%")
    print(f"Missing upper range: {results['conclusions']['missing_upper_percentage']:.1f}%")
    print("\n💡 RECOMMENDATION:")
    print("Check final layer activation function in generator_enhanced.py")
    print("Consider adding dynamic range loss during training")
    print("="*60)


if __name__ == '__main__':
    main()
