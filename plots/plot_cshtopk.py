import matplotlib.pyplot as plt
import torch
import os
from pathlib import Path

plt.figure(figsize=(10, 5))
results = torch.load(str(Path("./results/ringallreduce/cshtopk_actual/results_7000_CIFAR10_cshtopk_actual_seq_4_4_10.pt")), weights_only=True)
results = results['results']
plt.plot(results['rounds'], results['testACCs'], label="cshtopk_actual test accuracy", marker='', linestyle='-')
plt.plot(results['rounds'], results['trainACCs'], label="cshtopk_actual train accuracy", marker='', linestyle='-')
results = torch.load(str(Path("./results/ringallreduce/cshtopk_estimate/results_7000_CIFAR10_cshtopk_estimate_seq_4_4_10.pt")), weights_only=True)
results = results['results']
plt.plot(results['rounds'], results['testACCs'], label="cshtopk_estimate test accuracy", marker='', linestyle='-')
plt.plot(results['rounds'], results['trainACCs'], label="cshtopk_estimate train accuracy", marker='', linestyle='-')
plt.xlabel("Rounds")
plt.ylabel("Accuracy (%)")
plt.title("Test accuracy Over Training Rounds")
plt.legend()
plt.grid(True)
plt.show()