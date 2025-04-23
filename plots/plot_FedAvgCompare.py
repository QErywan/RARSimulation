import matplotlib.pyplot as plt
import torch
import os
from pathlib import Path

plt.figure(figsize=(10, 5))
results = torch.load(str(Path("./results/ringallreduce/none/results_2500_MNIST_none_seq_4_4_10.pt")), weights_only=True)
plt.plot(results['results']['rounds'], results['results']['trainACCs'], label="Train Accuracy", marker='', linestyle='-', linewidth=2)
plt.plot(results['results']['rounds'], results['results']['testACCs'], label="Test Accuracy ", marker='', linestyle='-', linewidth=2)
plt.xlabel("Rounds")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid(True)
plt.show()