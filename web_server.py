import cv2
import numpy as np
import os
import uuid
import time
import tempfile
import shutil
import base64
import json
import urllib.request
import urllib.error
import urllib.parse
from flask import Flask, render_template, request, jsonify

try:
    import paddle
    import paddle.nn as nn
except Exception:
    paddle = None
    nn = None

app = Flask(__name__)

EMOTIONS = ["happy", "sad", "angry", "surprise", "fear", "disgust", "neutral"]
IMG_SIZE = (48, 48)
NUM_CLASSES = 7

FER_SERVICE_URL = os.environ.get("FER_SERVICE_URL", "http://127.0.0.1:5002/predict")
USE_FER_SERVICE = os.environ.get("USE_FER_SERVICE", "0").strip() not in {"0", "false", "False", "no"}

if paddle is not None and nn is not None:
    class BasicBlock(nn.Layer):
        expansion = 1

        def __init__(self, in_channels, out_channels, stride=1):
            super().__init__()
            self.conv1 = nn.Conv2D(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias_attr=False,
            )
            self.bn1 = nn.BatchNorm2D(out_channels)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2D(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias_attr=False,
            )
            self.bn2 = nn.BatchNorm2D(out_channels)

            self.downsample = None
            if stride != 1 or in_channels != out_channels:
                self.downsample = nn.Sequential(
                    nn.Conv2D(
                        in_channels, out_channels, kernel_size=1, stride=stride, bias_attr=False
                    ),
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


    def load_model():
        model = EmotionResNetSmall(NUM_CLASSES)
        state = paddle.load("best_emotion_model.pdparams")
        model.set_state_dict(state)
        model.eval()
        return model
else:
    BasicBlock = None
    EmotionResNetSmall = None

    def load_model():
        raise RuntimeError(
            "Paddle 不可用，无法加载自训练模型。请设置 USE_FER_SERVICE=1 使用 fer 推理，或修复 paddle 环境。"
        )

_PADDLE_MODEL = None


def _get_paddle_model():
    global _PADDLE_MODEL
    if _PADDLE_MODEL is None:
        _PADDLE_MODEL = load_model()
    return _PADDLE_MODEL


def _multipart_request(url: str, field_name: str, filename: str, content_type: str, data: bytes, timeout: float = 10.0):
    boundary = "----WebKitFormBoundary" + uuid.uuid4().hex
    body = (
        f"--{boundary}\r\n"
        f"Content-Disposition: form-data; name=\"{field_name}\"; filename=\"{filename}\"\r\n"
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode("utf-8") + data + f"\r\n--{boundary}--\r\n".encode("utf-8")

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _fer_predict(img_bytes: bytes):
    try:
        raw = _multipart_request(
            FER_SERVICE_URL,
            field_name="image",
            filename="frame.jpg",
            content_type="image/jpeg",
            data=img_bytes,
            timeout=15.0,
        )
        data = json.loads(raw.decode("utf-8", errors="ignore") or "{}")
        if data.get("error"):
            return None, data.get("error"), None
        return data, None, None
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError) as e:
        return None, None, str(e)

def _get_cascade_path():
    env_path = (os.environ.get("OPENCV_HAAR_CASCADE_PATH") or "").strip()
    if env_path and os.path.exists(env_path):
        src = env_path
    else:
        candidates = []

        # OpenCV 官方 wheels 通常提供 cv2.data.haarcascades
        haar_root = None
        try:
            haar_root = getattr(getattr(cv2, "data", None), "haarcascades", None)
        except Exception:
            haar_root = None

        if haar_root:
            candidates.append(os.path.join(haar_root, "haarcascade_frontalface_default.xml"))

        # 兼容没有 cv2.data 的版本：从 cv2 安装位置附近寻找
        try:
            cv2_dir = os.path.dirname(getattr(cv2, "__file__", "") or "")
        except Exception:
            cv2_dir = ""

        if cv2_dir:
            candidates.extend(
                [
                    os.path.join(cv2_dir, "data", "haarcascade_frontalface_default.xml"),
                    os.path.join(os.path.dirname(cv2_dir), "data", "haarcascade_frontalface_default.xml"),
                ]
            )

        # 兜底：当前工作目录/项目目录（允许你手动放一份 xml）
        candidates.extend(
            [
                os.path.join(os.getcwd(), "haarcascade_frontalface_default.xml"),
                os.path.join(os.getcwd(), "models", "haarcascade_frontalface_default.xml"),
            ]
        )

        src = next((p for p in candidates if p and os.path.exists(p)), "")

    if not src:
        raise FileNotFoundError(
            "找不到人脸检测模型文件 haarcascade_frontalface_default.xml。"
            "\n你可以："
            "\n- 安装/修复 opencv-python（建议 4.x）"
            "\n- 或下载 haarcascade_frontalface_default.xml 并设置环境变量 OPENCV_HAAR_CASCADE_PATH 指向该文件"
        )

    dst_dir = os.path.join(tempfile.gettempdir(), "face_emotion_cascades")
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, "haarcascade_frontalface_default.xml")

    if not os.path.exists(dst):
        shutil.copyfile(src, dst)

    return dst


face_cascade = None
if not USE_FER_SERVICE:
    cascade_path = _get_cascade_path()
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise RuntimeError(f"人脸检测器加载失败: {cascade_path}")


SESSIONS_DIR = "emotion_dataset_sessions"
os.makedirs(SESSIONS_DIR, exist_ok=True)
PENDING_DIR = os.path.join(SESSIONS_DIR, "_pending")
os.makedirs(PENDING_DIR, exist_ok=True)
PENDING = {}
MAX_PENDING_PER_SESSION = 4


def preprocess_image(img):
    if paddle is None:
        raise RuntimeError("Paddle 不可用，无法执行 Paddle 推理预处理。请设置 USE_FER_SERVICE=1 使用 fer 推理。")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None, None, None

    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
    face_roi = gray[y : y + h, x : x + w]
    face_roi = cv2.resize(face_roi, IMG_SIZE)
    face_u8 = face_roi.copy()
    face_roi = face_roi.astype("float32") / 255.0
    face_roi = (face_roi - 0.5) / 0.5
    face_roi = np.expand_dims(face_roi, axis=(0, 1))
    face_roi = paddle.to_tensor(face_roi)
    return face_roi, (x, y, w, h), face_u8


def predict_emotion(img):
    if USE_FER_SERVICE:
        ok, buf = cv2.imencode(".jpg", img)
        if not ok:
            return "Invalid image", None
        data, infer_error, conn_error = _fer_predict(buf.tobytes())
        if conn_error:
            return f"FER service error: {conn_error}", None
        if infer_error:
            return infer_error, None
        return data.get("pred_emotion", "unknown"), float(data.get("confidence") or 0.0)

    face_roi, _, _ = preprocess_image(img)
    if face_roi is None:
        return "No face detected", None

    model = _get_paddle_model()
    logits = model(face_roi)
    pred_idx = int(paddle.argmax(logits, axis=1).numpy()[0])
    emotion = EMOTIONS[pred_idx]
    probs = paddle.nn.functional.softmax(logits, axis=1).numpy()[0]
    confidence = float(probs[pred_idx])
    return emotion, confidence


def _safe_emotion_name(name: str) -> str:
    if name not in EMOTIONS:
        raise ValueError("invalid emotion")
    return name


def _session_dir(session_id: str) -> str:
    return os.path.join(SESSIONS_DIR, session_id)


def _ensure_session_dirs(session_id: str):
    root = _session_dir(session_id)
    os.makedirs(root, exist_ok=True)
    for e in EMOTIONS:
        os.makedirs(os.path.join(root, e), exist_ok=True)
    return root


def _encode_preview(face_u8: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", face_u8)
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _save_face_to_session(session_id: str, target_emotion: str, face_u8: np.ndarray) -> str:
    _ensure_session_dirs(session_id)
    dst_dir = os.path.join(_session_dir(session_id), target_emotion)
    ts = int(time.time() * 1000)
    fn = f"{target_emotion}_{ts}_{uuid.uuid4().hex[:6]}.jpg"
    path = os.path.join(dst_dir, fn)
    cv2.imwrite(path, face_u8)
    return fn


@app.route("/start_session", methods=["POST"])
def start_session():
    session_id = uuid.uuid4().hex
    _ensure_session_dirs(session_id)
    return jsonify({"session_id": session_id})


@app.route("/capture", methods=["POST"])
def capture():
    file = request.files.get("image")
    if file is None:
        return jsonify({"error": "No image uploaded"})

    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Invalid image"})

    emotion, confidence = predict_emotion(img)
    return jsonify(
        {
            "emotion": emotion,
            "confidence": round(confidence * 100, 2) if confidence else 0,
        }
    )


@app.route("/confirm", methods=["POST"])
def confirm():
    data = request.get_json(silent=True) or {}
    pending_id = (data.get("pending_id") or "").strip()
    action = (data.get("action") or "").strip()
    if pending_id not in PENDING:
        return jsonify({"error": "Invalid pending_id"})

    meta = PENDING.pop(pending_id)
    path = meta.get("path")
    if not path or not os.path.exists(path):
        return jsonify({"error": "Missing pending file"})

    if action != "save":
        try:
            os.remove(path)
        except OSError:
            pass
        return jsonify({"ok": True, "saved": False})

    session_id = meta["session_id"]
    target_emotion = meta["target_emotion"]
    _ensure_session_dirs(session_id)
    dst_dir = os.path.join(_session_dir(session_id), target_emotion)
    ts = int(time.time() * 1000)
    fn = f"{target_emotion}_{ts}_{uuid.uuid4().hex[:6]}.jpg"
    dst = os.path.join(dst_dir, fn)
    shutil.move(path, dst)
    return jsonify({"ok": True, "saved": True, "filename": fn})


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if file is None:
        return jsonify({"error": "No image uploaded"})

    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Invalid image"})

    emotion, confidence = predict_emotion(img)
    return jsonify(
        {
            "emotion": emotion,
            "confidence": round(confidence * 100, 2) if confidence else 0,
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5001"))
    app.run(host="127.0.0.1", port=port, debug=True, use_reloader=False)
