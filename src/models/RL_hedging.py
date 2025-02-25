import numpy as np
import torch.nn as nn
import torch.optim as optim

class RLHedging(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, lr):
        super(RLHedging, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def forward(self, state):
        return self.model(state)

    def train_step(self, state, action, reward):
        self.optimizer.zero_grad()
        action_pred = self.model(state)
        loss = nn.MSELoss()(action_pred, action)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, state):
        return self.model(state)



