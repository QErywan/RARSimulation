import matplotlib.pyplot as plt
import torch
import os
from pathlib import Path

plt.figure(figsize=(10, 5))
for i in [4, 8, 16, 32, 64]:
    results = torch.load(str(Path(f"./results/ringallreduce/none/scale_CIFAR10_none_seq_{i}_{i}_10.pt")), weights_only=True)
    plt.plot(results['results']['rounds'], results['results']['testACCs'], label=f"{i} Clients", marker='', linestyle='-', linewidth=2)
plt.xlabel("Rounds")
plt.ylabel("Test Accuracy (%)")
# plt.title("Accuracy Over Communication Rounds")
plt.legend()
plt.grid(True)
plt.show()