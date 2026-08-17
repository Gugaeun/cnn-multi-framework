# ============================================================
# Flask 버전 모델 서빙 API (FastAPI와 동일 기능 -> 비교용)
# 실행: python main_flask.py
# ============================================================

from flask import Flask, request, jsonify
from PIL import Image
import io
import time

from model_loader import predict, load_model

app = Flask(__name__)

# Flask는 FastAPI처럼 startup 이벤트가 내장되어 있지 않아서,
# 앱 생성 직후 바로 모델을 로드해둠 (수동으로 처리해야 하는 부분 -- 비교 포인트)
load_model()
print("모델 로드 완료 (Flask)")


@app.route("/")
def root():
    return jsonify({"message": "CNN 이미지 분류 API (Flask)"})


@app.route("/health")
def health_check():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict_image():
    if "file" not in request.files:
        return jsonify({"error": "file 필드로 이미지를 업로드해주세요."}), 400

    file = request.files["file"]

    # FastAPI는 UploadFile.content_type으로 자동 검증했지만,
    # Flask는 이런 검증을 직접 짜야 함 (수동 처리 -> 코드량 차이 포인트)
    if not file.content_type or not file.content_type.startswith("image/"):
        return jsonify({"error": "이미지 파일만 업로드 가능합니다."}), 400

    contents = file.read()
    image = Image.open(io.BytesIO(contents))

    start = time.time()
    result = predict(image)
    elapsed = time.time() - start

    result["inference_time_ms"] = round(elapsed * 1000, 2)
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
