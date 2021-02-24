import numpy as np

def t():
    for k in range(0, 100):
        for p in range(0, k//2):
            for s in range(1, 100):
                t = 123 + 2 * p - (k - 1) - 1
                if (np.floor(t/s + 1) == 64):
                    print(f"k {k}, p {p}, s {s}")
                    
t()