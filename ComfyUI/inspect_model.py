from safetensors import safe_open
import sys

path = sys.argv[1]
with safe_open(path, framework="pt") as f:
    print(f"Keys in {path}:")
    for k in f.keys():
        print(f"  {k}: {f.get_tensor(k).shape}")