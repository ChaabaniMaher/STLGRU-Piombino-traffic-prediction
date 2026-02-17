#!/usr/bin/env python3
"""
Evaluate STLGRU Results for Piombino Dataset
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt

# Load results
print("="*60)
print("STLGRU PIOMBINO - RESULTS EVALUATION")
print("="*60)

with open('training_results.pkl', 'rb') as f:
    results = pickle.load(f)

predictions = results['test_results']['predictions']
targets = results['test_results']['targets']

print(f"\n📊 Predictions shape: {predictions.shape}")
print(f"📊 Targets shape: {targets.shape}")

# Calculate metrics properly
print("\n📈 Test Results by Horizon:")

horizon_metrics = []
for h in range(predictions.shape[1]):
    pred_h = predictions[:, h, :]  # Shape: (samples, sensors)
    target_h = targets[:, h, :]
    
    # Flatten to combine all sensors
    pred_flat = pred_h.flatten()
    target_flat = target_h.flatten()
    
    # Calculate metrics
    mae = np.mean(np.abs(pred_flat - target_flat))
    mse = np.mean((pred_flat - target_flat) ** 2)
    rmse = np.sqrt(mse)
    
    # Calculate MAPE only for non-zero values
    non_zero_mask = target_flat > 1.0  # Only consider values > 1 vehicle
    if np.sum(non_zero_mask) > 0:
        mape = np.mean(np.abs((pred_flat[non_zero_mask] - target_flat[non_zero_mask]) / 
                              target_flat[non_zero_mask])) * 100
    else:
        mape = np.nan
    
    # Calculate R²
    ss_res = np.sum((target_flat - pred_flat) ** 2)
    ss_tot = np.sum((target_flat - np.mean(target_flat)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-10))
    
    print(f"\n  Horizon {h+1} ({15*(h+1)} minutes):")
    print(f"    MAE:  {mae:.4f} vehicles")
    print(f"    RMSE: {rmse:.4f} vehicles")
    print(f"    MAPE: {mape:.2f}% (non-zero values only)")
    print(f"    R²:   {r2:.4f}")
    
    horizon_metrics.append({
        'horizon': h+1,
        'minutes': 15*(h+1),
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2
    })

# Average metrics
avg_mae = np.mean([m['mae'] for m in horizon_metrics])
avg_rmse = np.mean([m['rmse'] for m in horizon_metrics])
avg_mape = np.nanmean([m['mape'] for m in horizon_metrics])
avg_r2 = np.mean([m['r2'] for m in horizon_metrics])

print("\n" + "="*60)
print("📊 AVERAGE TEST METRICS")
print("="*60)
print(f"  MAE:  {avg_mae:.4f} vehicles")
print(f"  RMSE: {avg_rmse:.4f} vehicles")
print(f"  MAPE: {avg_mape:.2f}% (non-zero values only)")
print(f"  R²:   {avg_r2:.4f}")
print("="*60)

# Plot training curves
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(results['train_losses'], label='Train Loss', linewidth=2)
plt.plot(results['val_losses'], label='Val Loss', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(results['train_maes'], label='Train MAE', linewidth=2)
plt.plot(results['val_maes'], label='Val MAE', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.title('Training and Validation MAE')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
plt.show()

print("\n💾 Training curves saved to training_curves.png")

# Plot predictions vs actual for first sensor
plt.figure(figsize=(15, 6))
sensor_idx = 0  # STM1
horizon_idx = 0  # First prediction horizon (15 min)

# Take first 200 samples
n_samples = min(200, len(predictions))
time_steps = np.arange(n_samples)

pred_values = predictions[:n_samples, horizon_idx, sensor_idx]
actual_values = targets[:n_samples, horizon_idx, sensor_idx]

plt.plot(time_steps, actual_values, 'b-', label='Actual', linewidth=2, alpha=0.8)
plt.plot(time_steps, pred_values, 'r--', label='Predicted', linewidth=2, alpha=0.7)
plt.xlabel('Sample')
plt.ylabel('Traffic Flow (vehicles)')
plt.title(f'STM1 - 15-min Predictions vs Actual (First {n_samples} samples)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('predictions_vs_actual.png', dpi=150)
plt.show()

print("\n💾 Predictions plot saved to predictions_vs_actual.png")

# Summary statistics
print("\n" + "="*60)
print("📋 SUMMARY STATISTICS")
print("="*60)
print(f"Best validation loss: {results['best_val_loss']:.6f}")
print(f"Training completed in: 526.54 seconds (8.77 minutes)")
print(f"Model: STLGRU with 64 hidden units")
print(f"Dataset: Piombino (4 sensors, 52,700 time steps)")
print(f"Prediction: 12 steps history → 3 steps future (45 min)")
