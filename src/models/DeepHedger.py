import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


class DeepHedger(nn.Module):
    def __init__(self, strike: float, n_recur_hidden: int = 3, model_width: int = 16, model_depth: int = 2):
        """
        Uses RNN to produce sequences of hedging decisions from sequences of prices.
        Hidden state is 
        """
        super().__init__()
        self.strike = strike # note: Hedger is option specific. If we want to use for other options, can add more fields here, e.g. put/call or long/short
        self.n_recur_hidden = n_recur_hidden
        self.rnn = nn.RNN(input_size=1, hidden_size=n_recur_hidden, batch_first=True)
        
        layers = []
        layers.append(nn.Linear(n_recur_hidden, model_width))
        layers.append(nn.Sigmoid()) # think Sigmoid is good foro output because we want bounded hedge desicions
        for _ in range(model_depth - 2):
            layers.append(nn.Linear(model_width, model_width))
            layers.append(nn.ReLU()) # could also use sigmoid all the way through, haven't tried that
        layers.append(nn.Linear(model_width, 1))
        self.feedforward = nn.Sequential(*layers)
        
        for m in self.feedforward.modules(): #this is just to make sure we dont get big hedge desicions at the start of training
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -1e-3, 1e-3)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def compute_hedge(self, paths: torch.Tensor) -> torch.Tensor:
        """
        Computes hedge decisions for each time step.
        paths: Tensor of shape (batch_size, n_steps) 
        hedge_decisions: Tensor of shape (batch, n_steps, 1)
        """
        batch_size, n_steps = paths.shape
        x = paths.unsqueeze(2)  # shape: (batch, n_steps, 1)
        hidden_states, _ = self.rnn(x)  # (batch, n_steps, n_recur_hidden)
        hedge_decisions = self.feedforward(hidden_states)  # (batch, n_steps, 1)
        return hedge_decisions

    def compute_loss(self, paths: torch.Tensor) -> torch.Tensor:
        """
        Computes a loss based on semi-quadratic penalty
        paths: Tensor of shape (batch, n_steps) representing price paths.
        loss: Scalar tensor 
        """
        hedges = self.compute_hedge(paths)  # (batch, n_steps, 1)
        price_increments = paths[:, 1:] - paths[:, :-1]  #change here if we want to have data in returns instead
        hedges_trunc = hedges[:, :-1, 0]  # (batch, n_steps-1)
        portfolio = torch.sum(hedges_trunc * price_increments, dim=1) #change here if we want to have data in returns instead
        final_prices = paths[:, -1]
        payoff = torch.clamp(final_prices - self.strike, min=0) #here is hard coded a european call. Can change later if we want more general
        final_values = portfolio - payoff
        penalized = torch.where(final_values < 0, final_values, torch.tensor(0.0, device=final_values.device)) #hard-coded loss. can change later if we want more general
        loss = torch.mean(penalized ** 2)
        return loss

    def forward(self, paths: torch.Tensor) -> torch.Tensor:
        """
        Forwards the paths through the model to compute hedge decisions.
        """
        return self.compute_hedge(paths)

def hedging(df: pd.DataFrame) -> float:
    """
    General purpose hedging function.
    
    Args:
        df (pd.DataFrame): Dataframe with price paths to hedge. Must be of shape (price paths, time steps)
    
    Returns:
        float: validation loss for trained hedging model
    """
    

    # Assume FullyRecurrentDeepHedger is defined in an importable module, e.g., hedger.py
    # from src.models import FullyRecurrentDeepHedger  # make sure hedger.py is in your PYTHONPATH

    # --- Load Pre-Simulated Data ---
    # This file should contain a tensor of shape (n_paths, n_steps)
    # dataset = torch.load('data/processed/gbm_synth_data.csv')
    dataset = torch.tensor(df.values, dtype=torch.float32)

    # Determine device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = dataset.to(device)

    # --- Create Training and Validation Splits ---
    n_samples = dataset.shape[0]
    indices = torch.randperm(n_samples)
    train_split = int(0.8 * n_samples)
    train_indices = indices[:train_split]
    val_indices = indices[train_split:]
    train_data = dataset[train_indices]
    val_data = dataset[val_indices]

    batch_size = 2048
    train_loader = DataLoader(TensorDataset(train_data), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_data), batch_size=batch_size, shuffle=False)

    # --- Hyperparameters for Single Parameter Combination ---
    strike = 1.0
    n_recur_hidden = 4
    model_width = 32
    model_depth = 4
    lr = 1e-4
    num_epochs = 1000
    print_interval = 100

    # --- Instantiate the Model and Optimizer ---
    model = DeepHedger(
        strike=strike,
        n_recur_hidden=n_recur_hidden,
        model_width=model_width,
        model_depth=model_depth
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # --- Training Loop ---
    start_time = time.time()
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_losses = []
        for (batch_tensor,) in train_loader:
            optimizer.zero_grad()
            loss = model.compute_loss(batch_tensor)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        avg_train_loss = np.mean(epoch_losses)

        if epoch % print_interval == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch}, Training Loss: {avg_train_loss:.12f}, Time: {elapsed:.2f}s")

    # --- Evaluation on Validation Set ---
    model.eval()
    with torch.no_grad():
        val_losses = []
        for (batch_tensor,) in val_loader:
            loss = model.compute_loss(batch_tensor)
            val_losses.append(loss.item())
        avg_val_loss = np.mean(val_losses)
    print(f"Validation Loss: {avg_val_loss:.12f}")
    return avg_val_loss