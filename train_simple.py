import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import os
import time
from torch.utils.data import DataLoader, TensorDataset

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============================================
# Simple STLGRU Model Definition
# ============================================
class SimpleSTLGRU(nn.Module):
    def __init__(self, num_nodes=4, hidden_dim=64, seq_length=12, horizon=3):
        super(SimpleSTLGRU, self).__init__()
        
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length
        self.horizon = horizon
        
        # GRU for temporal processing
        self.gru = nn.GRU(
            input_size=num_nodes,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, num_nodes * horizon)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, num_nodes)
        Returns:
            out: Predictions of shape (batch, horizon, num_nodes)
        """
        # GRU forward pass
        gru_out, _ = self.gru(x)
        
        # Take last output
        last_out = gru_out[:, -1, :]
        
        # Project to predictions
        out = self.fc(last_out)
        
        # Reshape to (batch, horizon, num_nodes)
        out = out.view(-1, self.horizon, self.num_nodes)
        
        return out
    
    def get_summary(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }

# ============================================
# Training Functions
# ============================================
def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_mae = 0
    num_batches = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # Calculate MAE
        mae = torch.mean(torch.abs(output - target))
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_mae += mae.item()
        num_batches += 1
        
        if batch_idx % 50 == 0:
            print(f'    Batch {batch_idx}/{len(train_loader)}: Loss = {loss.item():.6f}, MAE = {mae.item():.6f}')
    
    return total_loss / num_batches, total_mae / num_batches

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    total_mae = 0
    num_batches = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            mae = torch.mean(torch.abs(output - target))
            
            total_loss += loss.item()
            total_mae += mae.item()
            num_batches += 1
    
    return total_loss / num_batches, total_mae / num_batches

def test(model, test_loader, criterion, scaler, device):
    """Test the model and return metrics"""
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data)
            predictions.append(output.cpu().numpy())
            targets.append(target.numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    
    # Inverse transform to original scale
    original_shape = predictions.shape
    predictions_flat = predictions.reshape(-1, predictions.shape[-1])
    targets_flat = targets.reshape(-1, targets.shape[-1])
    
    predictions_original = scaler.inverse_transform(predictions_flat)
    targets_original = scaler.inverse_transform(targets_flat)
    
    predictions_original = predictions_original.reshape(original_shape)
    targets_original = targets_original.reshape(targets.shape)
    
    # Calculate metrics by horizon
    print("\n📊 Test Results by Horizon:")
    horizon_metrics = []
    for h in range(predictions.shape[1]):
        pred_h = predictions_original[:, h, :]
        target_h = targets_original[:, h, :]
        
        mae = np.mean(np.abs(pred_h - target_h))
        mse = np.mean((pred_h - target_h) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((pred_h - target_h) / (target_h + 1e-8))) * 100
        
        print(f"  Horizon {h+1}: MAE = {mae:.4f}, RMSE = {rmse:.4f}, MAPE = {mape:.2f}%")
        horizon_metrics.append({'mae': mae, 'rmse': rmse, 'mape': mape})
    
    # Average metrics
    avg_mae = np.mean([m['mae'] for m in horizon_metrics])
    avg_rmse = np.mean([m['rmse'] for m in horizon_metrics])
    avg_mape = np.mean([m['mape'] for m in horizon_metrics])
    
    print("\n" + "="*60)
    print("📈 AVERAGE TEST METRICS")
    print("="*60)
    print(f"  MAE:  {avg_mae:.4f}")
    print(f"  RMSE: {avg_rmse:.4f}")
    print(f"  MAPE: {avg_mape:.2f}%")
    print("="*60)
    
    return {
        'predictions': predictions_original,
        'targets': targets_original,
        'avg_mae': avg_mae,
        'avg_rmse': avg_rmse,
        'avg_mape': avg_mape,
        'horizon_metrics': horizon_metrics
    }

# ============================================
# Main Training Function
# ============================================
def main():
    print("="*60)
    print("SIMPLE STLGRU TRAINING FOR PIOMBINO DATASET")
    print("="*60)
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    epochs = 50
    learning_rate = 0.001
    hidden_dim = 64
    
    print(f"\n📱 Device: {device}")
    
    # Load data
    print("\n📂 Loading Piombino dataset...")
    
    required_files = ["X_train.npy", "Y_train.npy", "X_val.npy", "Y_val.npy", 
                      "X_test.npy", "Y_test.npy", "scaler.pkl"]
    
    for file in required_files:
        if not os.path.exists(file):
            print(f"❌ ERROR: {file} not found!")
            return
    
    X_train = np.load("X_train.npy")
    Y_train = np.load("Y_train.npy")
    X_val = np.load("X_val.npy")
    Y_val = np.load("Y_val.npy")
    X_test = np.load("X_test.npy")
    Y_test = np.load("Y_test.npy")
    
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    print(f"\n📊 Dataset shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  Y_train: {Y_train.shape}")
    print(f"  X_val: {X_val.shape}")
    print(f"  Y_val: {Y_val.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  Y_test: {Y_test.shape}")
    
    # Get dimensions
    num_nodes = X_train.shape[2]
    seq_length = X_train.shape[1]
    horizon = Y_train.shape[1]
    
    print(f"\n⚙️ Configuration:")
    print(f"  Sensors (nodes): {num_nodes}")
    print(f"  History length: {seq_length} steps ({seq_length*15} minutes)")
    print(f"  Prediction horizon: {horizon} steps ({horizon*15} minutes)")
    print(f"  Hidden dimension: {hidden_dim}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {learning_rate}")
    
    # Create data loaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(Y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(Y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(Y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\n📦 DataLoaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Create model
    model = SimpleSTLGRU(
        num_nodes=num_nodes,
        hidden_dim=hidden_dim,
        seq_length=seq_length,
        horizon=horizon
    ).to(device)
    
    summary = model.get_summary()
    print(f"\n🤖 Model created:")
    print(f"  Total parameters: {summary['total_parameters']:,}")
    print(f"  Trainable parameters: {summary['trainable_parameters']:,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Training loop
    print("\n🚀 Starting training...")
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_maes = []
    val_maes = []
    
    for epoch in range(epochs):
        # Train
        train_loss, train_mae = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_maes.append(train_mae)
        
        # Validate
        val_loss, val_mae = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_maes.append(val_mae)
        
        # Scheduler step
        scheduler.step()
        
        # Print progress
        print(f"\n📈 Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.6f}, Train MAE: {train_mae:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}, Val MAE:   {val_mae:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': {
                    'num_nodes': num_nodes,
                    'hidden_dim': hidden_dim,
                    'seq_length': seq_length,
                    'horizon': horizon
                }
            }, 'best_model_simple.pth')
            print(f"  ✅ New best model saved! (Val Loss: {val_loss:.6f})")
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"\nBest validation loss: {best_val_loss:.6f}")
    
    # Load best model for testing
    checkpoint = torch.load('best_model_simple.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\n📊 Loaded best model from epoch {checkpoint['epoch']+1}")
    
    # Test
    test_results = test(model, test_loader, criterion, scaler, device)
    
    # Save results
    results = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_maes': train_maes,
        'val_maes': val_maes,
        'best_val_loss': best_val_loss,
        'test_results': test_results
    }
    
    with open('training_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("\n💾 Results saved to training_results.pkl")
    print("💾 Best model saved to best_model_simple.pth")
    print("\n" + "="*60)

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\n⏱️ Total execution time: {(end_time - start_time):.2f} seconds")
