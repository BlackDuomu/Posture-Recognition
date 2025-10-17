# Posture-Recognition
AI-powered posture correction system using video skeleton recognition and wearable IMU sensors. Combines Pose Transformer & IMU Transformer models with a cross-modal attention mechanism for enhanced motion analysis accuracy. Provides real-time feedback for dynamic activities, ensuring efficient movement and injury prevention.

## Project Structure

```
AI_Pose_Correction_Model/
├── data/ # 数据处理相关文件
│   ├── raw/ # 原始数据存储文件夹（视频、IMU原始数据）
│   ├── preprocessed/ # 处理后数据（归一化、对齐后的数据）
│   ├── labels/ # 数据标签文件
│   └── data_loader.py # 数据加载与预处理代码
├── models/ # AI模型相关文件
│   ├── pose_transformer.py # Pose Transformer 模型实现
│   ├── imu_transformer.py # IMU Transformer 模型实现
│   ├── cross_modal_attention.py # 跨模态注意力机制实现
│   ├── model.py # 模型集成（融合Pose Transformer + IMU Transformer）
│   ├── dropout_mechanism.py # 模态Dropout机制实现
│   └── utils.py # 常用工具函数（如模型初始化、权重加载等）
├── training/ # 训练相关文件
│   ├── train.py # 训练主文件（包含训练循环、损失函数、优化器等）
│   ├── loss.py # 定义损失函数
│   ├── optimizer.py # 定义优化器
│   └── callbacks.py # 训练回调函数（如早停、模型保存等）
├── inference/ # 推理相关文件
│   ├── infer.py # 推理主文件（加载训练好的模型，进行推理）
│   ├── preprocess_input.py # 输入数据预处理（包括视频骨架和IMU数据）
│   └── postprocess_output.py # 输出后处理（包括识别结果的分析、改进建议）
├── config/ # 配置文件
│   ├── config.yaml # 项目配置文件（例如训练参数、数据路径等）
│   ├── model_config.yaml # 模型超参数配置（如Transformer层数、Dropout率等）
│   └── training_config.yaml # 训练过程配置（如批量大小、学习率等）
├── checkpoints/ # 保存训练好的模型
│   └── model_checkpoint.ckpt # 模型检查点（训练过程中的模型保存文件）
├── logs/ # 训练日志
│   ├── training_log.txt # 训练日志文件（记录训练过程中的信息）
│   └── evaluation_log.txt # 模型评估日志
├── requirements.txt # 项目依赖库（如MindSpore、NumPy、PyTorch等）
├── README.md # 项目说明文档
└── run.py # 项目入口文件，包含训练、评估和推理等操作的选择
```
