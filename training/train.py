import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import csv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.data_loader import create_tennis_dataloaders
from models.model import TennisMultimodalTransformer


def unwrap_dataset(dataset):
    while hasattr(dataset, 'dataset'):
        dataset = dataset.dataset
    return dataset


def move_batch_to_device(batch, device):
    return {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }


def run_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0

    # 新增：单项准确率计数器
    total_major = 0
    total_action = 0
    total_quality = 0
    total = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad()
        outputs = model(batch)

        if isinstance(outputs, dict):
            loss_major = criterion(outputs['major'], batch['major_label'])
            loss_action = criterion(outputs['action'], batch['action_label'])
            loss_quality = criterion(outputs['quality'], batch['quality_label'])

            # 采用精调的 0.1 / 0.1 / 0.8 级联权重
            loss = 0.3 * loss_major + 0.3 * loss_action + 0.4 * loss_quality

            pred_major = outputs['major'].argmax(dim=1)
            pred_action = outputs['action'].argmax(dim=1)
            pred_quality = outputs['quality'].argmax(dim=1)

            # 严格联合准确率
            correct = (pred_major == batch['major_label']) & \
                      (pred_action == batch['action_label']) & \
                      (pred_quality == batch['quality_label'])
            total_correct += correct.sum().item()

            # 记录各层单项正确数
            total_major += (pred_major == batch['major_label']).sum().item()
            total_action += (pred_action == batch['action_label']).sum().item()
            total_quality += (pred_quality == batch['quality_label']).sum().item()
        else:
            labels = batch['label']
            loss = criterion(outputs, labels)
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch['label'].size(0)
        total += batch['label'].size(0)

    # 返回字典包含各项指标
    metrics = {
        'loss': total_loss / max(1, total),
        'strict_acc': total_correct / max(1, total),
        'major_acc': total_major / max(1, total) if total_major > 0 else 0.0,
        'action_acc': total_action / max(1, total) if total_action > 0 else 0.0,
        'quality_acc': total_quality / max(1, total) if total_quality > 0 else 0.0
    }
    return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    if loader is None:
        return None
    model.eval()
    total_loss = 0.0
    total_correct = 0

    # 新增：单项准确率计数器
    total_major = 0
    total_action = 0
    total_quality = 0
    total = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        outputs = model(batch)

        if isinstance(outputs, dict):
            loss_major = criterion(outputs['major'], batch['major_label'])
            loss_action = criterion(outputs['action'], batch['action_label'])
            loss_quality = criterion(outputs['quality'], batch['quality_label'])

            # 采用精调的 0.1 / 0.1 / 0.8 级联权重
            loss = 0.1 * loss_major + 0.1 * loss_action + 0.8 * loss_quality

            pred_major = outputs['major'].argmax(dim=1)
            pred_action = outputs['action'].argmax(dim=1)
            pred_quality = outputs['quality'].argmax(dim=1)

            correct = (pred_major == batch['major_label']) & \
                      (pred_action == batch['action_label']) & \
                      (pred_quality == batch['quality_label'])
            total_correct += correct.sum().item()

            # 👈 记录各层单项正确数
            total_major += (pred_major == batch['major_label']).sum().item()
            total_action += (pred_action == batch['action_label']).sum().item()
            total_quality += (pred_quality == batch['quality_label']).sum().item()
        else:
            labels = batch['label']
            loss = criterion(outputs, labels)
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()

        total_loss += loss.item() * batch['label'].size(0)
        total += batch['label'].size(0)

    # 返回字典包含各项指标
    metrics = {
        'loss': total_loss / max(1, total),
        'strict_acc': total_correct / max(1, total),
        'major_acc': total_major / max(1, total) if total_major > 0 else 0.0,
        'action_acc': total_action / max(1, total) if total_action > 0 else 0.0,
        'quality_acc': total_quality / max(1, total) if total_quality > 0 else 0.0
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train Tennis multimodal transformer (Hierarchical)')
    parser.add_argument('--dataset', default='/home/yezi/Desktop/Professions/BJUT/Posture-Recognition/data_utils/tennis_dataset_v1.pkl')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--label-key', default='class_id')
    parser.add_argument('--hierarchical', action='store_true', default=True, help="Use Hierarchical MTL architecture")
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, _ = create_tennis_dataloaders(
        args.dataset,
        batch_size=args.batch_size,
        label_key=args.label_key,
        split_by='subject_id',
        is_training=True
    )

    base_dataset = unwrap_dataset(train_loader.dataset)
    mapping = base_dataset.get_label_mapping()

    num_major_classes = int(max([s.get('major_label', 0) for s in base_dataset.samples])) + 1
    num_action_classes = int(max([s.get('action_label', 0) for s in base_dataset.samples])) + 1
    num_quality_classes = int(max([s.get('quality_label', s.get('label', 0)) for s in base_dataset.samples])) + 1

    print(f"=== [训练配置初始化] ===")
    print(f"模式: {'分层多任务(Hierarchical)' if args.hierarchical else '经典单分类(Flat)'}")
    print(f"大类数: {num_major_classes} | 小类数: {num_action_classes} | 纠错类数: {num_quality_classes}")
    print(f"=========================")

    # 初始化支持分层的网络
    model = TennisMultimodalTransformer(
        hierarchical=args.hierarchical,
        num_major_classes=num_major_classes,
        num_action_classes=num_action_classes,
        num_quality_classes=num_quality_classes
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    if args.smoke_test:
        model.train()
        batch = move_batch_to_device(next(iter(train_loader)), device)
        optimizer.zero_grad()
        outputs = model(batch)
        if isinstance(outputs, dict):
            loss = criterion(outputs['quality'], batch['quality_label'])
        else:
            loss = criterion(outputs, batch['label'])
        loss.backward()
        optimizer.step()
        print(f"Smoke test OK | Loss: {loss.item():.4f}")
        return

    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    for epoch in range(1, args.epochs + 1):
        # 调用新版函数，返回指标字典
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        # 记录画图和 CSV 历史（兼容老格式，使用主严格准确率）
        history['epoch'].append(epoch)
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['strict_acc'])

        if val_metrics is not None:
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['strict_acc'])
        else:
            history['val_loss'].append(None)
            history['val_acc'].append(None)

        # 各层独立指标和学习率
        current_lr = optimizer.param_groups[0]['lr']

        epoch_str = f"Epoch {epoch:03d}/{args.epochs:03d} | lr={current_lr:.6f}"

        train_str = (f"Train -> Loss: {train_metrics['loss']:.4f}, StrictAcc: {train_metrics['strict_acc']:.2%} "
                     f"(Major: {train_metrics['major_acc']:.1%} | Action: {train_metrics['action_acc']:.1%} | Quality: {train_metrics['quality_acc']:.1%})")

        if val_metrics is not None:
            val_str = (f"Val -> Loss: {val_metrics['loss']:.4f}, StrictAcc: {val_metrics['strict_acc']:.2%} "
                       f"(Major: {val_metrics['major_acc']:.1%} | Action: {val_metrics['action_acc']:.1%} | Quality: {val_metrics['quality_acc']:.1%})")
            print(f"{epoch_str}\n  {train_str}\n  {val_str}\n" + "-" * 80)
        else:
            print(f"{epoch_str}\n  {train_str}\n" + "-" * 80)

        # 更新学习率退火
        scheduler.step()

    os.makedirs('checkpoints', exist_ok=True)

    csv_path = '/home/yezi/Desktop/Professions/BJUT/Posture-Recognition/checkpoints/training_history.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
        for i in range(len(history['epoch'])):
            writer.writerow([
                history['epoch'][i],
                f"{history['train_loss'][i]:.6f}",
                f"{history['train_acc'][i]:.6f}",
                f"{history['val_loss'][i]:.6f}" if history['val_loss'][i] is not None else 'null',
                f"{history['val_acc'][i]:.6f}" if history['val_acc'][i] is not None else 'null'
            ])

    epochs_range = history['epoch']
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='Train Loss', color='blue', linestyle='-', marker='o',
             markersize=3)
    if any(v is not None for v in history['val_loss']):
        plt.plot(epochs_range, history['val_loss'], label='Val Loss', color='red', linestyle='--', marker='s',
                 markersize=3)
    plt.title('Loss Convergence Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_acc'], label='Train Acc', color='blue', linestyle='-', marker='o',
             markersize=3)
    if any(v is not None for v in history['val_acc']):
        plt.plot(epochs_range, history['val_acc'], label='Val Acc', color='red', linestyle='--', marker='s',
                 markersize=3)
    plt.title('Accuracy Evaluation Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()

    plt.tight_layout()
    curve_image_path = '/home/yezi/Desktop/Professions/BJUT/Posture-Recognition/checkpoints/training_curves.png'
    plt.savefig(curve_image_path, dpi=300)
    plt.close()
    ckpt_path = '/home/yezi/Desktop/Professions/BJUT/Posture-Recognition/checkpoints/tennis_multimodal_transformer.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'args': vars(args),
        'hierarchical': args.hierarchical,
        'num_major_classes': num_major_classes,
        'num_action_classes': num_action_classes,
        'num_quality_classes': num_quality_classes,
        'class_id_to_code': mapping['class_id_to_code'],
        'code_to_class_id': mapping['code_to_class_id'],
    }, ckpt_path)
    print(f"Model saved to {ckpt_path}")


if __name__ == '__main__':
    main()