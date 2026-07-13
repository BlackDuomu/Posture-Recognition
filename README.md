# MySport single-video model service

This repository runs the existing tennis multimodal checkpoint with one phone
video (Cam A), optional IMU CSV input, and an all-zero Cam B placeholder. It
also exposes the same inference path as an internal FastAPI service.

> **Synthetic IMU warning**
>
> `tests/fixtures/synthetic_imu.csv` and `zero_imu.csv` are deterministic,
> explicitly marked **SYNTHETIC** fixtures. They validate parsing, tensor
> shapes, missing-modality handling, fusion code paths, and API integration.
> They are not evidence of recognition accuracy or real-IMU effectiveness.

## Supported scope

- `VIDEO_ONLY`: one required Cam A video, zero IMU and zero Pose B.
- `IMU_ONLY`: code-path test, zero Pose A and Pose B; not exposed by the video
  upload endpoint.
- `VIDEO_WITH_IMU`: one real video plus an IMU CSV and zero Pose B.
- Two-camera inference, training, fine-tuning, ONNX, TensorRT, `torch.compile`,
  and server deployment are intentionally out of scope.

The checkpoint contract is fixed:

| Input | Shape | dtype |
|---|---:|---|
| IMU | `(B, 100, 9)` | `float32` |
| Pose A | `(B, 50, 99)` | `float32` |
| Pose B | `(B, 50, 99)` | `float32` |

Pose input is MediaPipe world landmarks resampled to 50 frames, normalized in
the same way as the dataset generator, then flattened from `(50, 33, 3)` to
`(50, 99)`. Missing modalities use exact all-zero tensors with the same shape,
dtype, batch dimension, and device.

## Windows GPU setup

The verified environment uses Python 3.10. Conda creates and isolates the
environment; pip installs the official PyTorch CUDA wheels.

Set the proxy only in the current PowerShell session when required:

```powershell
$env:HTTP_PROXY="http://127.0.0.1:7897"
$env:HTTPS_PROXY="http://127.0.0.1:7897"
Test-NetConnection 127.0.0.1 -Port 7897
```

Do not disable SSL verification. The proxy is not used or hard-coded by the
application.

Create and activate the environment:

```powershell
cd D:\MySport-Platform\model-service
conda env create -f environment.yml
conda activate mysport-model
python --version
```

Install the current verified Windows CUDA wheels from PyTorch's official wheel
index, then install the application dependencies:

```powershell
python -m pip install -r requirements-gpu.txt
python -m pip install -r requirements.txt
python -m pip check
```

`requirements-gpu.txt` resolves to PyTorch `2.12.0+cu130` and torchvision
`0.27.0+cu130`. The CUDA runtime is carried by the wheel; a separately installed
CUDA Toolkit is not required. See the
[official PyTorch installer](https://pytorch.org/get-started/locally/) when
refreshing versions.

Verify the real GPU and a CUDA tensor operation:

```powershell
python scripts/verify_gpu.py
```

The command must report `cudaAvailable: true` and the expected GPU name. It
exits unsuccessfully rather than silently falling back to CPU.

## Checkpoint

Place the existing state-dict checkpoint at:

```text
checkpoints/tennis_multimodal_transformer.pth
```

It must be a PyTorch checkpoint dictionary containing `model_state_dict` and
the hierarchical head metadata. New weights and large user videos are ignored
by Git and must not be committed.

Override the path at runtime with `MODEL_CHECKPOINT_PATH`.

## Command-line inference

Single real video:

```powershell
python inference/custom_infer.py `
  --cam-a data/videos/video_1.mp4 `
  --output-dir outputs/cli-video
```

IMU-only code-path test using explicitly synthetic data:

```powershell
python inference/custom_infer.py `
  --imu-csv tests/fixtures/synthetic_imu.csv `
  --output-dir outputs/cli-imu-only
```

Real video plus synthetic IMU integration test:

```powershell
python inference/custom_infer.py `
  --cam-a data/videos/video_1.mp4 `
  --imu-csv tests/fixtures/synthetic_imu.csv `
  --output-dir outputs/cli-video-synthetic-imu
```

The CLI prints the same structured JSON used by the API. The
`syntheticImu` field is derived from the fixture's explicit metadata.

## FastAPI service

Configuration is read from environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_CHECKPOINT_PATH` | `checkpoints/tennis_multimodal_transformer.pth` | Existing checkpoint |
| `MODEL_OUTPUT_DIR` | `outputs` | Per-request result root |
| `MODEL_DEVICE` | `cuda` | Required inference device |
| `MODEL_MAX_UPLOAD_MB` | `100` | Upload limit |
| `MODEL_ALLOW_CPU_FALLBACK` | `false` | Explicit opt-in only |

Start one Uvicorn worker:

```powershell
conda activate mysport-model
.\scripts\run_service.ps1
```

Or run the equivalent command:

```powershell
python run.py
```

The model is loaded once in FastAPI lifespan startup. GPU work is protected by
a single-concurrency semaphore; do not increase the worker count on the 8 GB
development GPU.

- Swagger: <http://127.0.0.1:8000/docs>
- Health: <http://127.0.0.1:8000/health>
- Model information: <http://127.0.0.1:8000/api/v1/model/info>

Upload a video with PowerShell/curl:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/v1/inference/video" `
  -F "video=@data/videos/video_1.mp4" `
  -F "cameraDirection=REAR"
```

Upload a video with the synthetic integration fixture:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/v1/inference/video" `
  -F "video=@data/videos/video_1.mp4" `
  -F "imuCsv=@tests/fixtures/synthetic_imu.csv"
```

Uploads are streamed with a size limit, checked by extension, file signature,
and OpenCV decoding, renamed to server-generated safe names, and removed after
the request. Each request uses an isolated UUID output directory. Returned
`clipPath` values are relative and never expose an absolute server path.

Errors use a stable shape:

```json
{
  "errorCode": "INVALID_VIDEO",
  "message": "The uploaded video could not be decoded.",
  "requestId": "optional-request-uuid"
}
```

## Synthetic IMU fixtures

Regenerate both deterministic fixtures:

```powershell
python scripts/generate_synthetic_imu.py
```

The nine feature columns are ordered as:

```text
acc_x, acc_y, acc_z,
gyro_x, gyro_y, gyro_z,
mag_x, mag_y, mag_z
```

The parser and source headers suggest acceleration `g`, angular velocity
`degrees/s`, and a magnetic-field `uT`-like field, but the repository contains
no real IMU sample and performs no unit conversion. Therefore the fixture only
guarantees numerical format and deterministic signal shape, not verified
physical units.

## Tests

Run the normal unit and API suite:

```powershell
python -m pytest -q
```

Run real-checkpoint/real-video GPU tests explicitly:

```powershell
$env:RUN_GPU_TESTS="1"
python -m pytest -m gpu -q
Remove-Item Env:RUN_GPU_TESTS
```

The real GPU verification covers the original CLI-compatible path, the new
single-video function, the FastAPI upload, and real video plus synthetic IMU.
Results from synthetic IMU are reported only as code-path integration evidence.

## Known limitations

- No real IMU file is available, so real-IMU behavior and model accuracy remain
  unverified.
- The repository does not include the original multimodal training dataset or
  its full training script, so the checkpoint cannot be reproduced here.
- Missing tokens are zeroed as in the existing model, but the original fusion
  layer does not pass its available `active_mask`; this architecture is kept
  unchanged to preserve checkpoint behavior.
- MediaPipe pose extraction runs on CPU; the PyTorch forward pass runs on CUDA.
- The three hierarchical heads are thresholded independently at the original
  default `0.50`; low-confidence heads return `UNKNOWN`.
