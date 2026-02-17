from __future__ import annotations

"""
แอป FastAPI สำหรับเปิดให้เรียกใช้งานโมเดลทำนายราคาบ้านผ่าน REST API

ไฟล์นี้อยู่ที่ root ของโปรเจกต์ เพื่อให้รันได้ง่ายด้วยคำสั่ง:

    uvicorn main:app --host 127.0.0.1 --port 8002

ส่วนที่โหลดโมเดลและฟังก์ชันทำนายจริง ๆ จะอยู่ใน `src/model.py`
"""

import json
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.model import DEFAULT_MODEL_INFO_PATH, build_default_service


# สร้างอินสแตนซ์หลักของ FastAPI
app = FastAPI(title="California Housing Model API", version="1.0.0")

# เปิดใช้ CORS เพื่อให้ frontend ที่อยู่คนละพอร์ต/โดเมนเรียก API นี้ได้
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    """
    โครงสร้างข้อมูลที่รับเข้ามาที่ /predict

    ฟิลด์ทั้งหมดตรงกับชื่อฟีเจอร์ของ model2 ในไฟล์ model_info.json
    """

    MedInc: float = Field(..., description="รายได้เฉลี่ยของครัวเรือน (MedInc)")
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float


class PredictResponse(BaseModel):
    """โครงสร้างข้อมูลที่ตอบกลับจาก /predict"""

    prediction: float
    units: str = "target (California housing dataset units)"
    features_used: Dict[str, float]


def _load_model_info() -> Dict[str, Any]:
    """อ่านข้อมูลฟีเจอร์และ metric ของโมเดลจากไฟล์ model_info.json"""
    with DEFAULT_MODEL_INFO_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


# สร้างอินสแตนซ์ ModelService หนึ่งตัวตอนแอปเริ่มทำงาน (ใช้ซ้ำทุกคำขอ)
service = build_default_service()


@app.get("/health")
def health() -> Dict[str, str]:
    """เอ็นด์พอยต์สำหรับตรวจว่าเซิร์ฟเวอร์ยังทำงานปกติหรือไม่"""
    return {"status": "ok"}


@app.get("/metadata")
def metadata() -> Dict[str, Any]:
    """คืนค่า metadata พื้นฐานของโมเดล (ฟีเจอร์ + metric)"""
    info = _load_model_info()
    return {
        "model": "model2_linear_regression.pkl",
        "features": service.features,
        "metrics": {
            "r2": info.get("model2_r2"),
            "rmse": info.get("model2_rmse"),
        },
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, debug: Optional[bool] = False) -> PredictResponse:
    """
    ทำนายราคาบ้านจากฟีเจอร์ที่ส่งมาในคำขอ

    ระหว่างพัฒนา สามารถส่ง query param `debug=true`
    เพื่อให้เห็น error จริง ๆ ของ Python แทนการตอบกลับเป็น HTTP 400 เฉย ๆ
    """
    try:
        # แปลงออบเจ็กต์ Pydantic ให้กลายเป็น dict แบบปกติ
        payload = req.model_dump()
        # ส่งต่อให้ ModelService เป็นคนเรียก predict จริง ๆ
        pred = service.predict_one(payload)
        return PredictResponse(prediction=pred, features_used=payload)
    except Exception as e:
        if debug:
            # ถ้าอยู่ในโหมด debug ให้โยน exception เดิมทิ้งไปเลยเพื่อดู stack trace เต็ม ๆ
            raise
        # ถ้าเป็นการใช้งานปกติ ให้ตอบกลับเป็น HTTP 400 พร้อมข้อความ error แบบอ่านเข้าใจง่าย
        raise HTTPException(status_code=400, detail=str(e))

