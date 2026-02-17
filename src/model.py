from __future__ import annotations

"""
ยูทิลิตี้หลักของโมเดลสำหรับโปรเจกต์ California housing

หน้าที่ของโมดูลนี้:
- หา path ที่ถูกต้องของไฟล์โมเดล sklearn (.pkl) และไฟล์ model_info.json
- โหลดโมเดลจากดิสก์
- ให้คลาส `ModelService` ที่มีเมธอด `predict_one()` สำหรับทำนายสะดวก ๆ
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd


# โฟลเดอร์ root ของโปรเจกต์ = โฟลเดอร์ที่อยู่เหนือ `src/` ขึ้นไป 1 ชั้น
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ตำแหน่งหลักของไฟล์โมเดลภายใต้โฟลเดอร์ `models/` (ที่อยู่ใต้ git)
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "model2_linear_regression.pkl"
DEFAULT_MODEL_INFO_PATH = PROJECT_ROOT / "models" / "model_info.json"

# ตำแหน่งสำรองใต้โฟลเดอร์ notebooks (กรณีรันเซฟใหม่จาก Jupyter)
ALT_MODEL_PATH = PROJECT_ROOT / "notebooks" / "models" / "model2_linear_regression.pkl"
ALT_MODEL_INFO_PATH = PROJECT_ROOT / "notebooks" / "models" / "model_info.json"


def resolve_existing_path(*candidates: Path) -> Path:
    """
    คืน path ตัวแรกที่มีอยู่จริงจากลิสต์ของ candidates

    ทำให้ service ใช้งานได้ทั้งกรณีที่โมเดลถูกเซฟใต้ `models/`
    หรือใต้ `notebooks/models/` ถ้าไม่เจอเลยสักไฟล์
    จะคืนตัวแรกเพื่อให้ผู้ใช้เจอ FileNotFoundError ที่ชัดเจน
    """
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def load_model_info(path: Path = DEFAULT_MODEL_INFO_PATH) -> Dict[str, Any]:
    """โหลด metadata ของโมเดล (ฟีเจอร์ + metric) จากไฟล์ JSON"""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_model(path: Path = DEFAULT_MODEL_PATH) -> Any:
    """โหลดออบเจกต์โมเดล sklearn ที่ถูกเซฟด้วย joblib"""
    return joblib.load(path)


@dataclass(frozen=True)
class ModelService:
    """
    คลาสห่อ (wrapper) บาง ๆ รอบโมเดล sklearn เพื่อให้รูปแบบ input/output ชัดเจน

    Attributes
    ----------
    model:
        โมเดล sklearn ที่โหลดมาแล้ว (ในโปรเจกต์นี้คือ LinearRegression)
    features:
        ลิสต์ชื่อฟีเจอร์ (ตามลำดับ) ที่โมเดลคาดหวัง
    """

    model: Any
    features: List[str]

    def predict_one(self, payload: Dict[str, float]) -> float:
        """
        ทำนาย 1 แถวข้อมูล ที่ส่งมาในรูป dict

        Parameters
        ----------
        payload:
            mapping จากชื่อฟีเจอร์ → ค่าเชิงตัวเลข
            จะต้องมีทุกฟีเจอร์ตามที่ระบุใน self.features
        """
        # ตรวจว่ามีฟีเจอร์ที่จำเป็นครบทุกตัวหรือไม่
        missing = [f for f in self.features if f not in payload]
        if missing:
            raise ValueError(f"Missing features: {missing}")

        # เก็บเฉพาะฟีเจอร์ที่รู้จัก และบังคับให้เป็น float
        row = {f: float(payload[f]) for f in self.features}

        # สร้าง DataFrame 1 แถว พร้อมเรียงคอลัมน์ให้ตรงตามลำดับฟีเจอร์
        X = pd.DataFrame([row], columns=self.features)

        # เรียกโมเดล sklearn จริง ๆ และแปลงผลลัพธ์ให้เป็น float ธรรมดา
        y_pred = self.model.predict(X)
        return float(y_pred[0])


def build_default_service(
    model_path: Path = resolve_existing_path(DEFAULT_MODEL_PATH, ALT_MODEL_PATH),
    model_info_path: Path = resolve_existing_path(DEFAULT_MODEL_INFO_PATH, ALT_MODEL_INFO_PATH),
) -> ModelService:
    """
    ฟังก์ชัน factory สำหรับสร้าง ModelService จากไฟล์ดีฟอลต์ของโปรเจกต์

    อ่านไฟล์ `model_info.json` เพื่อดูว่าฟีเจอร์ไหนเป็นของ model2
    แล้วค่อยโหลดไฟล์ .pkl ของ model2 จากดิสก์
    """
    info = load_model_info(model_info_path)

    # ตรงนี้เราใช้ลิสต์ฟีเจอร์ของ "model2" โดยเฉพาะ
    features = info.get("model2_features")
    if not isinstance(features, list) or not all(isinstance(x, str) for x in features):
        raise ValueError("model_info.json missing valid 'model2_features' list")

    model = load_model(model_path)
    return ModelService(model=model, features=list(features))

