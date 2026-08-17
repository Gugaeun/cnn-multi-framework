# ============================================================
# model_loader.py
# Flask / FastAPI 양쪽에서 공통으로 쓰는 모델 로딩 + 추론 유틸리티
# ============================================================

import numpy as np
import onnxruntime as ort

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

# ONNX 모델 하나로 통일해서 서빙 (PyTorch/Keras 어느 쪽에서 학습했든
# ONNX로 변환해두면 서빙 코드는 프레임워크에 상관없이 동일하게 짤 수 있음
# -> 이게 ONNX를 쓰는 실질적인 이유 중 하나)

_session = None


def load_model(onnx_path="cnn_model.onnx"):
    global _session
    if _session is None:
        _session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    return _session


def preprocess_image(image: "PIL.Image.Image") -> np.ndarray:
    """
    PIL Image -> 모델 입력 형태 (1, 3, 32, 32) float32 0~1 정규화
    """
    image = image.convert("RGB").resize((32, 32))
    arr = np.array(image).astype("float32") / 255.0   # (32, 32, 3)
    arr = arr.transpose(2, 0, 1)                        # (3, 32, 32)
    arr = np.expand_dims(arr, axis=0)                   # (1, 3, 32, 32)
    return arr


def predict(image: "PIL.Image.Image") -> dict:
    session = load_model()
    input_array = preprocess_image(image)

    logits = session.run(None, {"input": input_array})[0]  # (1, 10)

    # softmax
    exp = np.exp(logits - np.max(logits))
    probs = exp / np.sum(exp)
    probs = probs.flatten()

    predicted_idx = int(np.argmax(probs))

    return {
        "predicted_class": CLASS_NAMES[predicted_idx],
        "confidence": float(probs[predicted_idx]),
        "all_probabilities": {
            CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))
        },
    }
