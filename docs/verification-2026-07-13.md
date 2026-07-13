# Verification record — 2026-07-13

This record distinguishes real-video execution from synthetic-IMU code-path
testing. Synthetic IMU results are not accuracy evidence.

## Git starting point

- Repository: `model-service` only.
- Initial branch: `master`.
- Initial worktree: clean (`git status --short` had no entries).
- Remote: `origin https://github.com/BlackDuomu/Posture-Recognition.git`.
- Initial/latest master commit: `2b54eb2b6a152713892f84ddde6cf7798483f033`.
- `master...origin/master`: `0 0` after `git fetch origin --prune`.
- Working branch: `feature/single-video-model-service`.

No reset, clean, force-push, weight deletion, video deletion, or base Conda
modification was performed.

## Environment

| Item | Verified value |
|---|---|
| Conda | 24.11.3 |
| Environment | `mysport-model` |
| Python | 3.10.20 |
| PyTorch | 2.12.0+cu130 |
| torchvision | 0.27.0+cu130 |
| `torch.version.cuda` | 13.0 |
| `torch.cuda.is_available()` | `True` |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU |
| PyTorch-visible VRAM | 8187.5 MiB |
| Driver | 610.47 |

The current-session proxy `http://127.0.0.1:7897` was reachable. The user
`.condarc` contained only the default channel and no `proxy_servers`, so no
global Conda configuration was changed or restored. SSL verification was not
disabled.

`scripts/verify_gpu.py` successfully created and multiplied CUDA tensors. Its
verification allocation peaked at 8.16 MiB. `python -m pip check` reported no
broken requirements.

## Model and input contract

- Checkpoint: `checkpoints/tennis_multimodal_transformer.pth`.
- Size: 4,517,527 bytes.
- SHA-256: `5641B1372FD2E5AB99E9BD3E9499056E84A5B7614231C75F60E9A1891829133C`.
- Format: checkpoint dictionary containing `model_state_dict`, not a serialized
  full model.
- Strict load: 0 missing keys, 0 unexpected keys.
- Inputs: IMU `(B,100,9)`, Pose A `(B,50,99)`, Pose B `(B,50,99)`, all
  `float32` on the model device.

The repository contains no real IMU capture. Nine parsed features are ordered
`acc_x/y/z`, `gyro_x/y/z`, `mag_x/y/z`. Source headers suggest `g`,
`degrees/s`, and a magnetic `uT`-like field, but there is no unit conversion or
real sample with which to validate units.

## Original script baseline

Input video: `data/videos/video_1.mp4` (148 frames, 30 FPS, 4.933 seconds).

The first run changed only Linux defaults/import-root handling. MediaPipe found
two segments, but the original script then failed after 12,922.8 ms with:

```text
AssertionError: query should be unbatched 2D or batched 3D tensor but received 4-D query tensor
```

This confirmed the existing script's `(1,50,33,3)` inference input conflicted
with the `(1,50,99)` training/checkpoint contract.

After the minimum contract fix (normalization/flattening and zero placeholders),
the original algorithm completed on CUDA:

| Stage | Time |
|---|---:|
| Model load | 178.92 ms |
| MediaPipe pose extraction | 3,675.67 ms |
| Model forward (2 segments) | 309.94 ms |
| Clip export | 578.00 ms |
| Main processing flow | 4,837.59 ms |
| Peak PyTorch GPU allocation | 17.04 MiB |

The two exported baseline clips were independently matched back to source
frames: `[42,80)` and `[124,144)`, or 1,400–2,667 ms and 4,133–4,800 ms.

## New reusable inference

Final CLI validation used the same real checkpoint and real video:

| Stage | Time |
|---|---:|
| Model load | 150 ms |
| MediaPipe pose extraction | 2,562 ms |
| Model forward | 117 ms |
| Clip export | 479 ms |
| Processing including CLI model load | 3,400 ms |
| Peak PyTorch GPU allocation | 12.771 MiB |

Results:

| Segment | Time | Action | Sub-action | Issue | Confidence |
|---:|---|---|---|---|---|
| 0 | 1,400–2,667 ms | BACKHAND | BACKHAND_DRIVE | CONTACT_TOO_CLOSE | 0.986471 / 0.987069 / 0.998580 |
| 1 | 4,133–4,800 ms | BACKHAND | BACKHAND_DRIVE | CONTACT_TOO_CLOSE | 0.978521 / 0.979304 / 0.997089 |

The original-compatible baseline and new function match exactly on segment
count, frame/time ranges, all three class heads, and confidence values (the
original filenames/console rounded those values for display).

## FastAPI real HTTP verification

The actual service was started on `127.0.0.1:8000` with one worker, exercised
over HTTP, and stopped cleanly.

- `GET /health`: HTTP 200, `UP`, CUDA true, expected GPU, model loaded.
- `GET /api/v1/model/info`: HTTP 200, all modes/classes/shapes and `cuda`.
- `POST /api/v1/inference/video`, real video: HTTP 200, two segments exactly
  matching the new function; 3,408 ms total, 2,627 ms pose, 159 ms forward,
  514 ms clip export, 12.771 MiB peak allocation.
- The lifespan-loaded model reported `modelLoadMs: 0` per request and automated
  lifecycle tests confirmed `load_count == 1`.
- Returned clip paths were relative `outputs/<uuid>/segment-XX.mp4` values.
- No `_upload*` or `_imu*` temporary file remained after the requests.

## Synthetic IMU code paths

- Generator seed: `20260713`.
- Default length: 100 rows.
- Fixtures explicitly carry `synthetic=true` and a `SYNTHETIC_*` source.
- Repeated generation is byte-for-byte deterministic.
- Zero IMU fixture is exactly `(100,9)` float32 zeros after parsing.

IMU-only with the real checkpoint completed on CUDA without uninitialized Pose
variables: 404 ms including a 145 ms model load and 151 ms forward. This is a
code-path result only.

Real video plus synthetic IMU completed through the live HTTP service with
`mode=VIDEO_WITH_IMU`, `syntheticImu=true`, and warning
`SYNTHETIC_IMU_CODE_PATH_ONLY`: one mapped clip, 3,596 ms total, 3,169 ms pose,
6 ms forward, 396 ms clip export, and 12.770 MiB peak allocation. The resulting
classification is intentionally not used as evidence of model quality.

## Automated tests

Final commands and results:

```powershell
python -m pytest -q
# 31 passed, 3 skipped

$env:RUN_GPU_TESTS="1"
python -m pytest -q
# 34 passed
```

The full suite covers tensor shape/dtype/device, each zero placeholder,
IMU-only execution, deterministic fixtures, JSON serialization, invalid video,
missing checkpoint, unavailable CUDA, path traversal, isolated outputs,
temporary cleanup, health/info, upload variants, stable failure responses,
lifespan load count, and the real-GPU flows.

## Remaining limitations

- Real IMU parsing/effectiveness and model accuracy are unverified because no
  real IMU capture exists in the repository.
- The original multimodal dataset and complete training script are absent, so
  the checkpoint cannot be reproduced from this repository alone.
- The existing fusion architecture does not pass its available active mask;
  zero placeholders are preserved exactly rather than changing checkpoint
  classification behavior.
- MediaPipe pose extraction is CPU work; only the PyTorch model forward runs on
  CUDA.
- Dual-camera inference remains unsupported by design.
