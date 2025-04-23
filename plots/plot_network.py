import matplotlib.pyplot as plt
import torch
import os
from pathlib import Path

plt.figure(figsize=(10, 5))
for compression_scheme in ['vector_topk', 'chunk_topk_recompress', 'chunk_topk_single', 'csh', 'cshtopk_actual', 'cshtopk_estimate']:
    results = torch.load(str(Path(f"./results/ringallreduce/{compression_scheme}/exp_CIFAR10_{compression_scheme}_seq_4_4_10.pt")), weights_only=True)
    results = results['results']
    if compression_scheme == 'none':
        plt.plot(results['cumulative_latency'], results['testACCs'], label=f"{compression_scheme}", marker='', linestyle='-', color='b')
    elif compression_scheme == 'chunk_topk_recompress':
        plt.plot(results['cumulative_latency'], results['testACCs'], label=f"{compression_scheme}", marker='', linestyle='-', color='g')
    elif compression_scheme == 'chunk_topk_single':
        plt.plot(results['cumulative_latency'], results['testACCs'], label=f"{compression_scheme}", marker='', linestyle='-', color='r')
    elif compression_scheme == 'vector_topk':
        plt.plot(results['cumulative_latency'], results['testACCs'], label=f"{compression_scheme}", marker='', linestyle='-', color='orange')
    elif compression_scheme == 'csh':
        plt.plot(results['cumulative_latency'], results['testACCs'], label=f"{compression_scheme}", marker='', linestyle='-', color='purple')
    elif compression_scheme == 'cshtopk_actual':
        plt.plot(results['cumulative_latency'], results['testACCs'], label=f"{compression_scheme}", marker='', linestyle='-', color='brown')
    elif compression_scheme == 'cshtopk_estimate':
        plt.plot(results['cumulative_latency'], results['testACCs'], label=f"{compression_scheme}", marker='', linestyle='-', color='pink')

# plt.plot(results['cumulative_bandwidth'], results['trainACCs'], label="Test", marker='', linestyle='-')
plt.xlabel("Cumulative Latency (Number of Floating Points)")
plt.ylabel("Test Accuracy (%)")
# plt.title("Test accuracy Over Cumulative Bandwidth")
plt.legend()
plt.grid(True)
plt.show()