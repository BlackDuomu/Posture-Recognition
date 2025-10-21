import os
from typing import Optional


class Paths:
    """
    中央化的项目路径配置。所有模块应通过此类获取目录与文件路径，避免硬编码相对路径。
    路径均为绝对路径，基于此文件(utils.py)所在的项目根目录自动解析。
    """
    # 项目根目录
    PROJECT_ROOT: str = os.path.dirname(os.path.abspath(__file__))

    # 一级目录
    DATA_DIR: str = os.path.join(PROJECT_ROOT, 'data')
    CONFIG_DIR: str = os.path.join(PROJECT_ROOT, 'config')
    CHECKPOINTS_DIR: str = os.path.join(PROJECT_ROOT, 'checkpoints')
    LOGS_DIR: str = os.path.join(PROJECT_ROOT, 'logs')

    # data 子目录
    VIDEOS_DIR: str = os.path.join(DATA_DIR, 'videos')
    KEYPOINTS_DIR: str = os.path.join(DATA_DIR, 'keypoints')
    LABELS_DIR: str = os.path.join(DATA_DIR, 'labels')
    PREPROCESSED_DIR: str = os.path.join(DATA_DIR, 'preprocessed')
    RAW_DIR: str = os.path.join(DATA_DIR, 'raw')

    @staticmethod
    def ensure_dir(path: str) -> str:
        """确保目录存在，若不存在则创建；返回该目录绝对路径。"""
        if path and not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def keypoints_file(video_basename_no_ext: str) -> str:
        """
        根据视频名（不含扩展名）给出关键点文件的绝对路径。
        例如: video_1 -> <PROJECT_ROOT>/data/keypoints/video_1_keypoints.npy
        """
        return os.path.join(Paths.KEYPOINTS_DIR, f"{video_basename_no_ext}_keypoints.npy")

    @staticmethod
    def labels_file(filename: str = 'labels.npy') -> str:
        """labels.npy 的绝对路径。"""
        return os.path.join(Paths.LABELS_DIR, filename)


def ensure_dir(path: Optional[str]) -> str:
    """便捷函数，等价于 Paths.ensure_dir。"""
    return Paths.ensure_dir(path or '')

