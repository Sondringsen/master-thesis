# import pandas as pd
# import pickle
# import matplotlib.pyplot as plt


# model = "heston"

# df = pd.read_csv(f"data/processed/{model}_synth_data.csv", index_col=0)
# # df = 400*(1+df.cumsum(axis=1))

# # Plot the first 30 timeseries
# # print(df.iloc[3, :])
# for i in range(30):
#     plt.plot(df.iloc[i, :])

# plt.xlabel('Datapoints')
# plt.ylabel('Value')
# plt.title('First 30 Timeseries')
# plt.show()


# with open("data/params/heston_params.pkl", "rb") as file:
#     heston_params = pickle.load(file)

# print(heston_params)

#!/usr/bin/env python3
"""
sabr_signature_kernel_fit_iisignature.py

Learn SABR parameters (alpha, beta, rho) along with F0 and sigma0 by matching
the distribution of entire sample paths via a signature kernel–based MMD.
The signature kernel is defined as k_sig(x,y) = ⟨S(x), S(y)⟩,
where S(x) is the truncated signature of the path x computed with iisignature.
We use the unbiased estimator

   Loss = A - (2/(m*n)) Σ_{i=1}^m Σ_{j=1}^n k_sig(x_i, y_j)
          + 1/(n(n-1)) Σ_{i≠j} k_sig(y_i, y_j),

where A = (1/(m(m-1))) Σ_{i≠j} k_sig(x_i, x_j) is precomputed from the real data.
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import minimize
import iisignature  # using iisignature instead of signatory

# ------------------------------
# 1. SABR Simulation (Euler–Maruyama)
# ------------------------------
def sabr_simulate(alpha, beta, rho, F0, sigma0, dt, n_steps, n_paths, seed=None):
    """
    Simulate SABR paths using Euler–Maruyama without in-place updates.

    Dynamics:
       dF_t     = sigma_t * (F_t^beta) * dW_t,
       dsigma_t = alpha * sigma_t * dZ_t,
       with corr(dW_t, dZ_t) = rho.
    
    Returns:
      A tensor of shape (n_paths, n_steps+1) for the forward price.
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    # Ensure parameters are torch tensors.
    alpha = torch.tensor(alpha, dtype=torch.float32) if not isinstance(alpha, torch.Tensor) else alpha
    beta  = torch.tensor(beta,  dtype=torch.float32) if not isinstance(beta, torch.Tensor) else beta
    rho   = torch.tensor(rho,   dtype=torch.float32) if not isinstance(rho, torch.Tensor) else rho
    F0    = torch.tensor(F0,    dtype=torch.float32) if not isinstance(F0, torch.Tensor) else F0
    sigma0= torch.tensor(sigma0,dtype=torch.float32) if not isinstance(sigma0, torch.Tensor) else sigma0

    F_list = [torch.full((n_paths, 1), F0.item(), dtype=torch.float32)]
    sigma_list = [torch.full((n_paths, 1), sigma0.item(), dtype=torch.float32)]
    
    # Generate Brownian increments.
    dW_raw = torch.randn(n_paths, n_steps)
    dZ_raw = torch.randn(n_paths, n_steps)
    dW = dW_raw * (dt**0.5)
    dZ = rho * dW + (((1 - rho**2).clamp(min=1e-12)).sqrt()) * dZ_raw * (dt**0.5)
    
    for t in range(n_steps):
        F_t = F_list[-1]
        sigma_t = sigma_list[-1]
        dF = sigma_t * (F_t.clamp(min=1e-12) ** beta) * dW[:, t:t+1]
        dSig = alpha * sigma_t * dZ[:, t:t+1]
        F_next = F_t + dF
        sigma_next = sigma_t + dSig
        F_list.append(F_next)
        sigma_list.append(sigma_next)
    
    # Stack to get shape (n_paths, n_steps+1)
    F = torch.cat(F_list, dim=1)
    return F

# ------------------------------
# 2. Define the Signature Kernel using iisignature
# ------------------------------
def signature_kernel(x, y, depth):
    """
    Compute the signature kernel between two paths x and y.
    
    x, y: torch.Tensor of shape (T, d), where T is path length and d is channel dimension.
    The function converts the tensor to numpy, computes the truncated signature using iisignature,
    and returns the dot product of the two signatures.
    """
    # Convert torch tensors to numpy arrays.
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    
    # Compute truncated signatures (including the 0th term).
    sig_x = iisignature.sig(x_np, depth)
    sig_y = iisignature.sig(y_np, depth)
    
    # Return dot product.
    return float(np.dot(sig_x, sig_y))

# ------------------------------
# 3. Precompute Real–Real Kernel Term A
# ------------------------------
def compute_A(real_paths, depth):
    """
    Compute A = (1/(m(m-1))) * sum_{i≠j} k_sig(x_i, x_j) for the set of real paths.
    
    real_paths: torch.Tensor of shape (m, T, d)
    """
    m = real_paths.shape[0]
    total = 0.0
    count = 0
    for i in range(m):
        for j in range(i+1, m):
            k_val = signature_kernel(real_paths[i], real_paths[j], depth)
            total += k_val
            count += 1
    return total / count if count > 0 else 0.0

# ------------------------------
# 4. Generate "Real" Data from Known SABR Parameters
# ------------------------------
# True SABR parameters.
true_alpha  = 0.4
true_beta   = 0.5
true_rho    = -0.2
true_F0     = 1.0
true_sigma0 = 0.2

dt      = 1/252
n_steps = 30
n_paths = 300

# Generate real SABR data with a fixed seed.
real_F = sabr_simulate(true_alpha, true_beta, true_rho,
                       true_F0, true_sigma0, dt, n_steps, n_paths,
                       seed=1234)
# Treat each forward price path as a trajectory.
# Add a channel dimension: shape becomes (n_paths, n_steps+1, 1)
real_paths = real_F.unsqueeze(-1)

# Set signature truncation depth.
depth = 5

# Precompute A (real-real kernel term).
m = real_paths.shape[0]
A = compute_A(real_paths, depth)
print("Precomputed A (real-real term):", A)

# ------------------------------
# 5. Define the Objective Function Using the Signature Kernel MMD Estimator
# ------------------------------
def objective(params):
    """
    Given parameters [alpha, beta, rho, F0, sigma0],
    simulate n_paths SABR trajectories and compute the unbiased estimator
    for the signature kernel MMD between the real and simulated distributions:
    
      Loss = A - (2/(m*n)) sum_{i=1}^m sum_{j=1}^n k_sig(real_i, sim_j)
             + (1/(n(n-1))) sum_{i≠j} k_sig(sim_i, sim_j).
    
    Returns the loss as a scalar float.
    """
    alpha, beta, rho, F0, sigma0 = params
    # Impose hard penalties if parameters are out-of-bound.
    if alpha <= 0 or not (0 <= beta <= 1) or not (-1 < rho < 1) or F0 <= 0 or sigma0 <= 0:
        return 1e6
    
    # Use a fixed seed for simulation determinism.
    sim_F = sabr_simulate(alpha, beta, rho, F0, sigma0, dt, n_steps, n_paths, seed=999)
    sim_paths = sim_F.unsqueeze(-1)  # shape: (n_paths, n_steps+1, 1)
    
    n = sim_paths.shape[0]
    
    # Compute cross term: (1/(m*n)) sum_{i=1}^m sum_{j=1}^n k_sig(real_i, sim_j)
    cross_total = 0.0
    for i in range(m):
        for j in range(n):
            cross_total += signature_kernel(real_paths[i], sim_paths[j], depth)
    cross_term = cross_total / (m * n)
    
    # Compute simulated-simulated term: (1/(n(n-1))) sum_{i≠j} k_sig(sim_i, sim_j)
    sim_total = 0.0
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            sim_total += signature_kernel(sim_paths[i], sim_paths[j], depth)
            count += 1
    sim_term = sim_total / count if count > 0 else 0.0
    
    loss = A - 2 * cross_term + sim_term
    return loss

# ------------------------------
# 6. Use SciPy's Optimizer to Learn All Parameters
# ------------------------------
# Initial guess for parameters: [alpha, beta, rho, F0, sigma0]
init_params = [0.1, 0.9, 0.6, 0.8, 0.3]
# Define bounds: alpha > 0, beta in [0,1], rho in (-0.99,0.99), F0 > 0, sigma0 > 0.
bounds = [(1e-6, 5.0), (0.0, 1.0), (-0.99, 0.99), (0.1, 2.0), (1e-6, 2.0)]

result = minimize(objective, x0=init_params, method='Nelder-Mead', bounds=bounds,
                  options={'maxiter': 1000, 'disp': True})

est_alpha, est_beta, est_rho, est_F0, est_sigma0 = result.x

print("\nRecovered parameters:")
print(f"  alpha: true = {true_alpha:.3f}, est = {est_alpha:.3f}")
print(f"  beta : true = {true_beta:.3f}, est = {est_beta:.3f}")
print(f"  rho  : true = {true_rho:.3f}, est = {est_rho:.3f}")
print(f"  F0   : true = {true_F0:.3f}, est = {est_F0:.3f}")
print(f"  sigma0: true = {true_sigma0:.3f}, est = {est_sigma0:.3f}")
