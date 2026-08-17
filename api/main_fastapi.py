# ============================================================
# FastAPI 버전 모델 서빙 API
# 실행: uvicorn main_fastapi:app --reload --port 8000
# 문서 확인: http://localhost:8000/docs (자동 생성됨)
# ============================================================

from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import io
import time

from model_loader import predict, load_model

app = FastAPI(
    title="CNN 이미지 분류 API (FastAPI)",
    description="CIFAR-10 클래스로 이미지를 분류하는 API",
    version="1.0.0",
)


@app.on_event("startup")
def startup_event():
    # 서버 시작 시 모델을 미리 로드해서, 첫 요청이 느려지지 않게 함
    load_model()
    print("모델 로드 완료 (FastAPI)")


@app.get("/")
def root():
    return {"message": "CNN 이미지 분류 API (FastAPI). /docs 에서 API 문서를 확인하세요."}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    start = time.time()
    result = predict(image)
    elapsed = time.time() - start

    result["inference_time_ms"] = round(elapsed * 1000, 2)
    return result
