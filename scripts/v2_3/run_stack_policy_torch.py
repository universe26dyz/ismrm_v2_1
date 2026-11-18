"""Server launcher selecting the same deterministic Torch backend for every ablation."""
import os

if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
    raise RuntimeError('Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting Python')
if os.environ.get('PYTHONHASHSEED') != '20260911':
    raise RuntimeError('Set PYTHONHASHSEED=20260911 before starting Python')

import torch
from nesvor.inr import models as nesvor_models

nesvor_models.USE_TORCH = True
torch.use_deterministic_algorithms(True)
print(f'USE_TORCH = {nesvor_models.USE_TORCH}', flush=True)
print(f'deterministic_algorithms = {torch.are_deterministic_algorithms_enabled()}', flush=True)
print(f'CUBLAS_WORKSPACE_CONFIG = {os.environ["CUBLAS_WORKSPACE_CONFIG"]}', flush=True)
print(f'PYTHONHASHSEED = {os.environ["PYTHONHASHSEED"]}', flush=True)

from ismrm_v2.proposed.q002.run_q002_reconstruction import main

if __name__ == '__main__':
    main()
