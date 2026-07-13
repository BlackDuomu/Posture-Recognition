"""Fail-fast CUDA environment verification for the model service."""

from __future__ import annotations

import json
import platform

import torch


def main() -> None:
    result = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchCuda": torch.version.cuda,
        "cudaAvailable": torch.cuda.is_available(),
        "deviceCount": torch.cuda.device_count(),
    }
    if not torch.cuda.is_available():
        print(json.dumps(result, indent=2))
        raise SystemExit("CUDA is unavailable; GPU verification failed.")

    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)
    free_before, total = torch.cuda.mem_get_info(0)
    left = torch.arange(4096, dtype=torch.float32, device="cuda:0").reshape(64, 64)
    product = left @ left.T
    torch.cuda.synchronize(0)
    free_after, _ = torch.cuda.mem_get_info(0)
    result.update(
        {
            "gpuName": torch.cuda.get_device_name(0),
            "freeMemoryMiBBefore": round(free_before / 1024**2, 2),
            "freeMemoryMiBAfter": round(free_after / 1024**2, 2),
            "totalMemoryMiB": round(total / 1024**2, 2),
            "tensorDevice": str(product.device),
            "tensorChecksum": float(product.sum().item()),
            "peakAllocatedMiB": round(torch.cuda.max_memory_allocated(0) / 1024**2, 2),
        }
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
