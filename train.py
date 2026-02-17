import torch
import torch.nn as nn
import torch.nn.functional as F
from stsgcl import STSGCL
from modules import *

class STLGRU(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, seq_length, horizon, adj):
        super(STLGRU, self).__init__()
        
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length
        self.horizon = horizon
        
        # Spatial-Temporal Graph Convolution Layer
        self.stsgcl = STSGCL(
            c_in=input_dim,
            c_out=hidden_dim,
            num_nodes=num_nodes,
            T=seq_length,
            adj=adj
        )
        
        # Gated Recurrent Unit for temporal processing
        self.gru_cell = GRUCell(hidden_dim, hidden_dim)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, horizon * num_nodes)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, seq_length, num_nodes)
        Returns:
            out: Predictions of shape (batch, horizon, num_nodes)
        """
        batch_size, seq_len, num_nodes = x.shape
        
        # Apply STSGCL to each time step
        h = []
        for t in range(seq_len):
            # Take one time step: (batch, num_nodes)
            x_t = x[:, t, :].unsqueeze(-1)  # Shape: (batch, num_nodes, 1)
            
            # Apply graph convolution
            h_t = self.stsgcl(x_t)  # Shape: (batch, num_nodes, hidden_dim)
            h.append(h_t)
        
        # Stack along time dimension
        h = torch.stack(h, dim=1)  # Shape: (batch, seq_len, num_nodes, hidden_dim)
        
        # Initialize hidden state
        hidden_state = torch.zeros(batch_size, self.num_nodes, self.hidden_dim).to(x.device)
        
        # Process through GRU
        outputs = []
        for t in range(seq_len):
            hidden_state = self.gru_cell(h[:, t, :, :], hidden_state)
            
        # Use final hidden state for prediction
        last_hidden = hidden_state  # Shape: (batch, num_nodes, hidden_dim)
        
        # Flatten for output layer
        last_hidden_flat = last_hidden.reshape(batch_size, -1)  # Shape: (batch, num_nodes * hidden_dim)
        
        # Generate predictions
        out = self.output_layer(last_hidden_flat)  # Shape: (batch, horizon * num_nodes)
        out = out.reshape(batch_size, self.horizon, self.num_nodes)  # Shape: (batch, horizon, num_nodes)
        
        return out

# Simple GRU Cell implementation
class GRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GRUCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Reset gate
        self.W_r = nn.Linear(input_dim + hidden_dim, hidden_dim)
        
        # Update gate
        self.W_z = nn.Linear(input_dim + hidden_dim, hidden_dim)
        
        # Candidate hidden state
        self.W_h = nn.Linear(input_dim + hidden_dim, hidden_dim)
        
    def forward(self, x, h_prev):
        """
        Args:
            x: Current input (batch, num_nodes, input_dim)
            h_prev: Previous hidden state (batch, num_nodes, hidden_dim)
        Returns:
            h_next: Next hidden state (batch, num_nodes, hidden_dim)
        """
        # Flatten node dimension for linear layers
        batch_size, num_nodes, _ = x.shape
        x_flat = x.reshape(batch_size * num_nodes, -1)
        h_prev_flat = h_prev.reshape(batch_size * num_nodes, -1)
        
        # Concatenate input and previous hidden state
        combined = torch.cat([x_flat, h_prev_flat], dim=1)
        
        # Gates
        r = torch.sigmoid(self.W_r(combined))
        z = torch.sigmoid(self.W_z(combined))
        
        # Candidate hidden state
        combined_candidate = torch.cat([x_flat, r * h_prev_flat], dim=1)
        h_tilde = torch.tanh(self.W_h(combined_candidate))
        
        # New hidden state
        h_next_flat = (1 - z) * h_prev_flat + z * h_tilde
        
        # Reshape back
        h_next = h_next_flat.reshape(batch_size, num_nodes, -1)
        
        return h_next
