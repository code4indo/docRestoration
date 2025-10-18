"""
GradNorm: Gradient Normalization for Adaptive Loss Balancing
Based on: Chen et al., "GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks"
https://arxiv.org/abs/1711.02257

Automatically balances multiple loss components by adjusting their weights based on gradient magnitudes.
"""

import tensorflow as tf
import numpy as np
from typing import Dict, List, Optional


class GradNorm(tf.keras.layers.Layer):
    """
    GradNorm layer for adaptive loss balancing.
    
    Balances loss components by normalizing gradient magnitudes relative to initial training rates.
    """
    
    def __init__(
        self,
        num_losses: int,
        loss_names: List[str],
        alpha: float = 1.5,
        initial_task_loss: Optional[Dict[str, float]] = None,
        update_frequency: int = 1,
        name: str = "gradnorm"
    ):
        """
        Args:
            num_losses: Number of loss components to balance
            loss_names: Names of loss components (e.g., ['pixel', 'perceptual', 'ctc'])
            alpha: Hyperparameter controlling relative weighting (0.5-2.0 typical)
                   Higher alpha = stronger adaptation to training dynamics
            initial_task_loss: Initial loss values for normalization (if None, uses first batch)
            update_frequency: Update weights every N batches (default=1 for every batch)
        """
        super(GradNorm, self).__init__(name=name)
        self.num_losses = num_losses
        self.loss_names = loss_names
        self.alpha = alpha
        self.update_frequency = update_frequency
        self.step_counter = 0
        
        # Loss weights (trainable parameters)
        self.loss_weights = self.add_weight(
            name="loss_weights",
            shape=(num_losses,),
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=True,
            constraint=tf.keras.constraints.NonNeg()  # Ensure non-negative
        )
        
        # Initial loss values (for normalization)
        self.initial_losses = self.add_weight(
            name="initial_losses",
            shape=(num_losses,),
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=False
        )
        
        # Track if initialized
        self.initialized = self.add_weight(
            name="initialized",
            shape=(),
            initializer=tf.keras.initializers.Constant(0.0),
            trainable=False,
            dtype=tf.float32
        )
        
    def initialize_losses(self, loss_values: tf.Tensor):
        """Initialize loss values from first batch."""
        if tf.equal(self.initialized, 0.0):
            self.initial_losses.assign(tf.maximum(loss_values, 1e-8))
            self.initialized.assign(1.0)
    
    def compute_inverse_training_rate(self, current_losses: tf.Tensor) -> tf.Tensor:
        """
        Compute inverse training rate: L_i(t) / L_i(0)
        Shows how much each loss has decreased relative to start.
        """
        # Avoid division by zero
        return tf.math.divide_no_nan(
            current_losses,
            tf.maximum(self.initial_losses, 1e-8)
        )
    
    def compute_target_gradnorm(self, inverse_training_rates: tf.Tensor) -> tf.Tensor:
        """
        Compute target gradient norms for each loss.
        
        Target = average_inverse_rate * (r_i ^ alpha)
        where r_i is relative inverse training rate.
        """
        # Average inverse training rate
        avg_rate = tf.reduce_mean(inverse_training_rates)
        
        # Relative rates (normalized to mean)
        relative_rates = tf.math.divide_no_nan(
            inverse_training_rates,
            tf.maximum(avg_rate, 1e-8)
        )
        
        # Apply power scaling
        target_norms = avg_rate * tf.pow(relative_rates, self.alpha)
        
        return target_norms
    
    def update_weights(
        self,
        gradients: List[tf.Tensor],
        current_losses: tf.Tensor,
        last_shared_layer: tf.Variable
    ) -> Dict[str, float]:
        """
        Update loss weights based on gradient norms.
        
        Args:
            gradients: List of gradient tensors for each loss component
            current_losses: Current values of each loss [num_losses]
            last_shared_layer: Shared layer to compute gradient norms on
            
        Returns:
            Dictionary with updated weights and metrics
        """
        self.step_counter += 1
        
        # Only update every N steps
        if self.step_counter % self.update_frequency != 0:
            return self._get_weight_dict()
        
        # Initialize on first call
        self.initialize_losses(current_losses)
        
        # Compute gradient norms for each loss
        grad_norms = []
        for grad in gradients:
            if grad is not None:
                # L2 norm of gradients
                norm = tf.norm(tf.reshape(grad, [-1]), ord=2)
                grad_norms.append(norm)
            else:
                grad_norms.append(tf.constant(0.0))
        
        grad_norms = tf.stack(grad_norms)
        
        # Compute inverse training rates
        inv_rates = self.compute_inverse_training_rate(current_losses)
        
        # Compute target gradient norms
        target_norms = self.compute_target_gradnorm(inv_rates)
        
        # Compute GradNorm loss: L1 distance between actual and target norms
        # Normalize both to make scale-invariant
        mean_grad_norm = tf.reduce_mean(grad_norms)
        mean_target_norm = tf.reduce_mean(target_norms)
        
        normalized_grads = tf.math.divide_no_nan(grad_norms, mean_grad_norm)
        normalized_targets = tf.math.divide_no_nan(target_norms, mean_target_norm)
        
        gradnorm_loss = tf.reduce_sum(tf.abs(normalized_grads - normalized_targets))
        
        # Compute gradients of GradNorm loss w.r.t. loss weights
        with tf.GradientTape() as tape:
            tape.watch(self.loss_weights)
            # Recompute with watched weights
            weighted_grad_norms = normalized_grads * self.loss_weights
            loss = tf.reduce_sum(tf.abs(weighted_grad_norms - normalized_targets))
        
        weight_gradients = tape.gradient(loss, [self.loss_weights])
        
        # Update weights using simple gradient descent
        if weight_gradients[0] is not None:
            lr = 0.01  # Small learning rate for weight updates
            new_weights = self.loss_weights - lr * weight_gradients[0]
            # Normalize to sum to num_losses (preserve total weight magnitude)
            new_weights = tf.nn.relu(new_weights)  # Ensure non-negative
            new_weights = new_weights * self.num_losses / tf.reduce_sum(new_weights)
            self.loss_weights.assign(new_weights)
        
        # Return metrics
        return {
            **self._get_weight_dict(),
            'gradnorm_loss': float(gradnorm_loss.numpy()),
            'avg_grad_norm': float(mean_grad_norm.numpy()),
            'avg_inv_rate': float(tf.reduce_mean(inv_rates).numpy())
        }
    
    def _get_weight_dict(self) -> Dict[str, float]:
        """Get current weights as dictionary."""
        weights_np = self.loss_weights.numpy()
        return {f"{name}_weight": float(weights_np[i]) for i, name in enumerate(self.loss_names)}
    
    def get_weights(self) -> tf.Tensor:
        """Get current loss weights."""
        return self.loss_weights
    
    def get_normalized_weights(self) -> Dict[str, float]:
        """Get weights normalized to percentages."""
        weights = self.loss_weights.numpy()
        total = np.sum(weights)
        return {
            name: float(weights[i] / total * 100)
            for i, name in enumerate(self.loss_names)
        }


class SimpleAdaptiveBalancer:
    """
    Simplified adaptive balancing based on loss magnitude ratios.
    Faster and simpler than GradNorm, good for quick testing.
    """
    
    def __init__(
        self,
        loss_names: List[str],
        target_ratios: Optional[Dict[str, float]] = None,
        adaptation_rate: float = 0.1
    ):
        """
        Args:
            loss_names: Names of losses to balance
            target_ratios: Target contribution ratios (e.g., {'ctc': 0.65, 'visual': 0.35})
            adaptation_rate: How fast to adapt weights (0.01-0.5)
        """
        self.loss_names = loss_names
        self.weights = {name: 1.0 for name in loss_names}
        self.target_ratios = target_ratios or {name: 1.0 / len(loss_names) for name in loss_names}
        self.adaptation_rate = adaptation_rate
        self.ema_losses = {name: None for name in loss_names}
        self.ema_decay = 0.9
        
    def update(self, loss_dict: Dict[str, float]) -> Dict[str, float]:
        """
        Update weights based on loss magnitudes.
        
        Args:
            loss_dict: Dictionary of loss name -> value
            
        Returns:
            Updated weights dictionary
        """
        # Update EMA of losses
        for name in self.loss_names:
            if name in loss_dict:
                current_loss = loss_dict[name]
                if self.ema_losses[name] is None:
                    self.ema_losses[name] = current_loss
                else:
                    self.ema_losses[name] = (
                        self.ema_decay * self.ema_losses[name] + 
                        (1 - self.ema_decay) * current_loss
                    )
        
        # Compute current ratios
        total_loss = sum(
            self.ema_losses[name] * self.weights[name]
            for name in self.loss_names
            if self.ema_losses[name] is not None
        )
        
        if total_loss > 0:
            current_ratios = {
                name: (self.ema_losses[name] * self.weights[name]) / total_loss
                for name in self.loss_names
                if self.ema_losses[name] is not None
            }
            
            # Adjust weights to move ratios toward targets
            for name in self.loss_names:
                if name in current_ratios and name in self.target_ratios:
                    ratio_error = self.target_ratios[name] - current_ratios[name]
                    # Increase weight if contribution too low, decrease if too high
                    adjustment = 1.0 + self.adaptation_rate * ratio_error
                    self.weights[name] *= adjustment
                    # Ensure positive
                    self.weights[name] = max(self.weights[name], 0.01)
        
        return self.weights.copy()
    
    def get_weights(self) -> Dict[str, float]:
        """Get current weights."""
        return self.weights.copy()
    
    def get_ratios(self) -> Dict[str, float]:
        """Get current contribution ratios."""
        total = sum(
            self.ema_losses[name] * self.weights[name]
            for name in self.loss_names
            if self.ema_losses[name] is not None
        )
        
        if total > 0:
            return {
                name: (self.ema_losses[name] * self.weights[name]) / total * 100
                for name in self.loss_names
                if self.ema_losses[name] is not None
            }
        return {name: 0.0 for name in self.loss_names}
