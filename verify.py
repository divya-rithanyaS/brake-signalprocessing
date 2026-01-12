import numpy as np
import scipy.io as sio

data = sio.loadmat("data/phdiff.mat")
A = data["A"]   # (156, 50001)

# State 3 = signals 104 to 155 (0-based index)
state3 = A[104:156]

print("Checking original instantaneous values for State 3:\n")

for i, sig in enumerate(state3, start=105):
    print(f"Signal {i}:")
    print("  Min:", np.min(sig))
    print("  Max:", np.max(sig))
    print("  Mean:", np.mean(sig))
    print("  Std :", np.std(sig))
    print("  First 10 values:", sig[:10])
    print("  Last 10 values :", sig[-10:])
    print("-"*50)
