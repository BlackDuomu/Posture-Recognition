# inference/infer.py
import os
import argparse
import numpy as np
import torch
from models.pose_transformer import PoseTransformer
from utils import Paths


def flatten_keypoints_per_frame(seq: np.ndarray) -> np.ndarray:
    """(T,33,3) -> (T,99)"""
    if seq.ndim != 3 or seq.shape[1:] != (33, 3):
        raise ValueError(f"Expected keypoints shape (T,33,3), got {seq.shape}")
    T = seq.shape[0]
    return seq.reshape(T, 33 * 3)


def pad_truncate(seq_2d: np.ndarray, seq_len: int) -> np.ndarray:
    """Pad/truncate 2D sequence (T,99) to (seq_len,99) with zeros."""
    T, D = seq_2d.shape
    out = np.zeros((seq_len, D), dtype=seq_2d.dtype)
    if T >= seq_len:
        out[:] = seq_2d[:seq_len]
    else:
        out[:T] = seq_2d
    return out


def main():
    parser = argparse.ArgumentParser(description='Pose Transformer Inference')
    parser.add_argument('--video_name', type=str, default='video_1',
                        help='视频基名（不含扩展名），例如 video_1；将从 data/keypoints/<video_name>_keypoints.npy 读取')
    parser.add_argument('--seq_len', type=int, default=100, help='统一的时序长度')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1) 加载模型与权重
    model = PoseTransformer(input_dim=99).to(device)
    ckpt_path = os.path.join(Paths.CHECKPOINTS_DIR, 'pose_transformer.pth')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # 2) 加载关键点并对齐形状
    keypoints_path = Paths.keypoints_file(args.video_name)
    if not os.path.exists(keypoints_path):
        raise FileNotFoundError(f"Keypoints not found: {keypoints_path}")
    seq = np.load(keypoints_path)            # (T,33,3)
    seq = flatten_keypoints_per_frame(seq)   # (T,99)
    seq = pad_truncate(seq, args.seq_len)    # (seq_len,99)

    xb = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)  # (1, seq_len, 99)

    # 3) 推理
    with torch.no_grad():
        logits = model(xb)   # (1,2)
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        pred = torch.argmax(probs).item()

    print(f"Predicted Class: {pred} | probs={probs.cpu().numpy().round(4).tolist()}")


if __name__ == '__main__':
    main()
