FROM python:3.11-slim

WORKDIR /app

# PYTHONDONTWRITEBYTECODE=1 บอก Python ว่าไม่ต้องสร้างไฟล์ .pyc (bytecode) เวลา import โมดูล → ลดไฟล์ขยะใน image
# PYTHONUNBUFFERED=1 ทำให้ output ของ Python (stdout/stderr) ไม่ถูก buffer → log แสดงทันทีใน console/docker logs เหมาะกับรันแอปใน container
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ติดตั้ง dependency สำหรับ build บางแพ็กเกจ (ถ้าไม่ต้องใช้สามารถลบ build-essential ได้)
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# ติดตั้ง Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# คัดลอกซอร์สโค้ดทั้งหมด (ยกเว้นไฟล์ที่ถูก ignore ใน .dockerignore)
COPY . .

# เปิดพอร์ตของ FastAPI และ static frontend
EXPOSE 8002
EXPOSE 5500

# ดีฟอลต์ให้รัน FastAPI backend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]

