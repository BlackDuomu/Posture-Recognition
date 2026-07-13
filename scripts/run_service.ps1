param(
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

if ($env:CONDA_DEFAULT_ENV -ne "mysport-model") {
    throw "Activate the environment first: conda activate mysport-model"
}

python scripts/verify_gpu.py
python -m uvicorn app.main:app --host $HostAddress --port $Port --workers 1
