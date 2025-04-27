import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import iisig
import torch

def visualization_signature_tsne(ori_data, generated_data, model_name, compare=3000, signature_level=5):
    """Using path signatures and tSNE for generated and original data visualization.
  
    Args:
        - ori_data: original data
        - generated_data: generated synthetic data
        - model_name: name of the model
        - compare: number of samples to compare
        - signature_level: level of the truncated signature
    """
    # Analysis sample size (for faster computation)
    ori_data = np.array([ori_data[i:i+30] for i in range(len(ori_data)-30 + 1)])
    ori_data = ori_data / ori_data[:, 0:1]

    generated_data = np.array(generated_data.values)
    generated_data = generated_data[..., np.newaxis]

    anal_sample_no = min([compare, ori_data.shape[0]])
    idx = np.random.permutation(ori_data.shape[0])[:anal_sample_no]

    ori_data = ori_data[idx]
    generated_data = generated_data[idx]

    # Add time component
    time = np.linspace(0, 1, ori_data.shape[1])[np.newaxis, :, np.newaxis]
    time = np.repeat(time, ori_data.shape[0], axis=0)
    
    # Combine time and price data
    ori_data_with_time = np.concatenate([time, ori_data], axis=2)
    generated_data_with_time = np.concatenate([time, generated_data], axis=2)

    # Convert to torch tensors for iisig
    ori_data_tensor = torch.tensor(ori_data_with_time, dtype=torch.float32)
    generated_data_tensor = torch.tensor(generated_data_with_time, dtype=torch.float32)

    # Compute signatures using iisig
    # iisig expects paths in format [batch, length, channels]
    ori_signatures = iisig.signature(ori_data_tensor, signature_level)
    generated_signatures = iisig.signature(generated_data_tensor, signature_level)

    # Convert back to numpy
    ori_signatures = ori_signatures.numpy()
    generated_signatures = generated_signatures.numpy()

    # Combine signatures for t-SNE
    combined_signatures = np.concatenate([ori_signatures, generated_signatures], axis=0)

    # Apply t-SNE
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(combined_signatures)

    # Plotting
    f, ax = plt.subplots(1)
    colors = ["black" for i in range(anal_sample_no)] + [model_color[model_name] for i in range(anal_sample_no)]

    plt.scatter(tsne_results[:anal_sample_no, 0], tsne_results[:anal_sample_no, 1],
                c=colors[:anal_sample_no], alpha=0.2, label="Original")
    plt.scatter(tsne_results[anal_sample_no:, 0], tsne_results[anal_sample_no:, 1],
                c=colors[anal_sample_no:], alpha=0.2, label="Synthetic")

    ax.legend()
    plt.title(f't-SNE plot with Path Signatures (Level {signature_level}) for {model_names_pretty[model_name]}')
    plt.xlabel('x-tsne')
    plt.ylabel('y-tsne')
    plt.show()

# Example usage:
# visualization_signature_tsne(original_data, generated_data, "time_vae", compare=3000, signature_level=5) 