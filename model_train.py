import pickle
import numpy as np
import sys
import time
import argparse
from typing import Optional

import cv2
import paddle
import paddle.nn as nn
from paddle.io import Dataset, DataLoader

BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
NUM_CLASSES = 7
BALANCE_CAP = 3000


class EmotionDataset(Dataset):
    def __init__(self, data, labels, augment: bool = False):
        self.data = data
        self.labels = labels
        self.augment = augment

    def __getitem__(self, idx):
        img = self.data[idx]

        if img.ndim == 2:
            img = img[:, :, None]
        elif img.ndim == 3 and img.shape[0] == 1 and img.shape[-1] != 1:
            img = np.transpose(img, (1, 2, 0))

        img = img.astype("float32")
        if img.max() > 1.5:
            img = img / 255.0
            img = (img - 0.5) / 0.5

        if self.augment:
            if np.random.rand() < 0.5:
                img = img[:, ::-1, :]

            angle = float(np.random.uniform(-10.0, 10.0))
            h, w = img.shape[:2]
            m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
            img = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            if img.ndim == 2:
                img = img[:, :, None]

        img = np.transpose(img, (2, 0, 1))

        label = np.int64(self.labels[idx])
        return img, label

    def __len__(self):
        return len(self.data)


class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2D(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias_attr=False
        )
        self.bn1 = nn.BatchNorm2D(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2D(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias_attr=False
        )
        self.bn2 = nn.BatchNorm2D(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2D(in_channels, out_channels, kernel_size=1, stride=stride, bias_attr=False),
                nn.BatchNorm2D(out_channels),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class EmotionResNetSmall(nn.Layer):
    def __init__(self, num_classes=7):
        super().__init__()
        self.in_channels = 64

        self.stem = nn.Sequential(
            nn.Conv2D(1, 64, kernel_size=3, stride=1, padding=1, bias_attr=False),
            nn.BatchNorm2D(64),
            nn.ReLU(),
        )

        self.layer1 = self._make_layer(64, blocks=2, stride=1)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2D((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, blocks, stride):
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride=stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = paddle.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def _accuracy_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.size == 0:
        return 0.0
    return float((y_true == y_pred).mean())


def train_model(
    dataset_pkl: str = "emotion_dataset.pkl",
    log_interval: int = 10,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0,
    device: Optional[str] = None,
):
    if device is None:
        device = "gpu" if paddle.is_compiled_with_cuda() else "cpu"
    paddle.set_device(device)

    with open(dataset_pkl, "rb") as f:
        dataset = pickle.load(f)

    X_train = dataset["X_train"]
    X_test = dataset["X_test"]
    y_train = dataset["y_train"]
    y_test = dataset["y_test"]

    # 类别均衡（方案1）：对大类下采样，对小类过采样到相同上限
    rng = np.random.default_rng(42)
    y_train = np.asarray(y_train)
    indices_by_class = {i: np.where(y_train == i)[0] for i in range(NUM_CLASSES)}
    balanced_indices = []
    for cls in range(NUM_CLASSES):
        idxs = indices_by_class.get(cls)
        if idxs is None or len(idxs) == 0:
            continue

        if len(idxs) >= BALANCE_CAP:
            chosen = rng.choice(idxs, size=BALANCE_CAP, replace=False)
        else:
            chosen = rng.choice(idxs, size=BALANCE_CAP, replace=True)
        balanced_indices.append(chosen)

    if len(balanced_indices) > 0:
        balanced_indices = np.concatenate(balanced_indices)
        rng.shuffle(balanced_indices)
        X_train = X_train[balanced_indices]
        y_train = y_train[balanced_indices]

    train_dataset = EmotionDataset(X_train, y_train, augment=True)
    test_dataset = EmotionDataset(X_test, y_test, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    model = EmotionResNetSmall(NUM_CLASSES)
    try:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    except TypeError:
        criterion = nn.CrossEntropyLoss()
    lr_scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
        learning_rate=LEARNING_RATE, T_max=int(epochs), eta_min=1e-5
    )
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=1e-4,
    )

    best_acc = 0.0

    for epoch in range(int(epochs)):
        model.train()
        train_loss = 0.0
        train_acc = 0.0

        epoch_start = time.time()

        for step, (data, label) in enumerate(train_loader, start=1):
            logits = model(data)
            loss = criterion(logits, label)

            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

            train_loss += float(loss.numpy().item())
            pred = paddle.argmax(logits, axis=1)
            train_acc += _accuracy_np(label.numpy(), pred.numpy())

            if log_interval and (step == 1 or step % int(log_interval) == 0 or step == len(train_loader)):
                elapsed = max(1e-9, time.time() - epoch_start)
                it_per_sec = step / elapsed
                remaining = max(0, len(train_loader) - step)
                eta = remaining / it_per_sec if it_per_sec > 0 else 0.0
                avg_loss = train_loss / step
                avg_acc = train_acc / step
                msg = (
                    f"Epoch [{epoch+1}/{int(epochs)}] "
                    f"Step {step}/{len(train_loader)} "
                    f"Loss {avg_loss:.4f} "
                    f"Acc {avg_acc:.4f} "
                    f"{it_per_sec:.2f} it/s "
                    f"ETA {eta:.0f}s"
                )
                print(msg, flush=True)

        model.eval()
        test_loss = 0.0
        test_acc = 0.0

        with paddle.no_grad():
            for data, label in test_loader:
                logits = model(data)
                loss = criterion(logits, label)
                test_loss += float(loss.numpy().item())
                pred = paddle.argmax(logits, axis=1)
                test_acc += _accuracy_np(label.numpy(), pred.numpy())

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc / len(train_loader)
        avg_test_loss = test_loss / len(test_loader)
        avg_test_acc = test_acc / len(test_loader)

        print(f"Epoch [{epoch+1}/{int(epochs)}]")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}")
        print(f"Test Loss: {avg_test_loss:.4f}, Test Acc: {avg_test_acc:.4f}")
        print("-" * 50)

        if avg_test_acc > best_acc:
            best_acc = avg_test_acc
            paddle.save(model.state_dict(), "best_emotion_model.pdparams")
            print(f"保存最佳模型，准确率: {best_acc:.4f}")

        lr_scheduler.step()

    print(f"训练完成！最佳测试准确率: {best_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_pkl", nargs="?", default="emotion_dataset.pkl")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    train_model(
        dataset_pkl=args.dataset_pkl,
        log_interval=args.log_interval,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )
