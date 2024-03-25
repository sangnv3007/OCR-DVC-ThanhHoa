from fastapi import FastAPI, File, UploadFile, status, Body
from pydantic import BaseModel
from typing import Union
from process import ReturnInfoNew, yolo_load, OCRFile
from fastapi.middleware.cors import CORSMiddleware
import json
import uvicorn
import os
import shutil
import base64
import time
import datetime
import asyncio
import fitz
# from thongke import create_connection, select_one_tasks, update_record, select_one_tasks_with_time, insert_record, select_time_last_record
from starlette import status
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp
class Item(BaseModel):
    name: Union[str, None] = None
    stringbase64: Union[str, None] = None
class LimitUploadSize(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, max_upload_size: int) -> None:
        super().__init__(app)
        self.max_upload_size = max_upload_size

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.method == 'POST':
            if 'content-length' not in request.headers:
                return Response(status_code=status.HTTP_411_LENGTH_REQUIRED)
            content_length = int(request.headers['content-length'])
            if content_length > self.max_upload_size:
                return Response(status_code=status.HTTP_431_REQUEST_HEADER_FIELDS_TOO_LARGE)
        return await call_next(request)

app = FastAPI(
    title="OCR-DVCThanhHoa",
    description="""Copyright 2024 for TAN DAN ., JSC. All right reserved\n
    MVB1 - Chung chi tu bo, hoi phuc di tich 
    MVB2 - Giay chung nhan HDV du lich noi dia, HDV Quoc Te
    MVB3 - Ly lich tu phap
    MVB4 - Ban sao van bang THPT
    MVB5 - Quyet dinh khen thuong
    MVB6 - Quyet dinh cong nhan truong Chuan quoc gia
    MVB7 - Thu tuc thanh lap hoi
    MVB8 - Giay phep khai thac thuy san
    MVB9 - Thu tuc cham dut kinh doanh
    MVB10 - Thu tuc an toan thuc pham san xuat kinh doanh
    MVB11 - Thu tuc an toan thuc pham cap Huyen
    MVB12 - Thu tuc cai chinh ho tich
    MVB13 - Thu tuc phu hieu xe oto kinh doanh van tai(LoaiCu)
    MVB14 - Thu tu cap phep giay phep xe tap lai
    MVB15 - Thu tuc giay buon ban thuoc thu y
    MVB16 - Thu tuc cap phu hieu xe o to kinh doanh van tai(LoaiMoi)
    MVB17 - Thu tuc giay phep lai xe quoc te
    MVB18 - Thu tuc cap moi giay phep lai xe
    MVB19 - Thu tuc dang ky chi nhanh cua hop tac xa
    MVB20 - Thu tuc chung chi nang luc thau 
    MVB21 - Thu tuc dang ky noi quy lao dong
    MVB22 - Thu tuc xac nhan tinh trang hon nha
    MVB23 - Thu tuc khai bao an toan lao dong
    MVB24 - Thu tuc giay chung nhan cua hang du dieu kien ban le xang dau
    MVB25 - Thu tuc ho so cong bo hop quy san pham, hang hoa VLXD
    MVB26 - Thu tuc bao cao lao dong nguoi nuoc ngoai
    MVB27 - Thu tuc cap giay phep lao dong nuoc ngoai
    \n""",
    version="beta-0.0.1"
    )

origins = [
    "http://192.168.2.70:3011",
    "http://dangkyvaora.hanhchinhcong.org",
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(LimitUploadSize, max_upload_size=10000000)  # ~10MB

''' Load mo hinh '''

# Chung chi tu bo, hoi phuc di tich
net_MVB1 = yolo_load(f'./models/MVB1/best.pt')

# Giay chung nhan HDV du lich noi dia, HDV Quoc Te
net_MVB2 = yolo_load(f'./models/MVB2/best.pt')

# Ly lich tu phap
net_MVB3 = yolo_load(f'./models/MVB3/best_rename.pt')

# Ban sao van bang THPT
net_MVB4 = yolo_load(f'./models/MVB4/best.pt')

# Quyet dinh khen thuong
net_MVB5 = yolo_load(f'./models/MVB5/best.pt')

# Quyet dinh cong nhan truong Chuan quoc gia(Can xem lai)
net_MVB6 = yolo_load(f'./models/MVB6/best.pt')

# Thu tuc thanh lap hoi
net_MVB7 = yolo_load(f'./models/MVB7/best.pt')

# Giay phep khai thac thuy san (Chua bo sung du lieu)
net_MVB8 = yolo_load(f'./models/MVB8/best.pt')

# Thu tuc cham dut kinh doanh (Chua bo sung du lieu)
net_MVB9 = yolo_load(f'./models/MVB9/best.pt')

# Thu tuc an toan thuc pham san xuat kinh doanh (Chua bo sung du lieu)
net_MVB10 = yolo_load(f'./models/MVB10/best.pt')

# Thu tuc an toan thuc pham cap Huyen (Chua bo sung du lieu)
net_MVB11 = yolo_load(f'./models/MVB11/best.pt')

# Thu tuc cai chinh ho tich (Chua bo sung du lieu)
net_MVB12 = yolo_load(f'./models/MVB12/best.pt')

# Thu tuc phu hieu xe oto kinh doanh van tai(LoaiCu)
net_MVB13 = yolo_load(f'./models/MVB13/best.pt')

# Thu tu cap phep giay phep xe tu lai
net_MVB14 = yolo_load(f'./models/MVB14/best.pt')

# Thu tuc giay nhan buon ban thuoc thu y
net_MVB15 = yolo_load(f'./models/MVB15/best.pt')

# Thu tuc cap phu hieu xe o to kinh doanh van tai(LoaiMoi)
net_MVB16 = yolo_load(f'./models/MVB16/best.pt')

# Thu tuc giay phep lai xe quoc te
net_MVB17 = yolo_load(f'./models/MVB17/best.pt')

# Thu tuc cap moi giay phep lai xe
net_MVB18 = yolo_load(f'./models/MVB18/best.pt')

# Thu tuc dang ky chi nhanh cua hop tac xa
net_MVB19 = yolo_load(f'./models/MVB19/best.pt')

# Thu tuc chung chi nang luc thau 
net_MVB20 = yolo_load(f'./models/MVB20/best.pt')

# Thu tuc dang ky noi quy lao dong
net_MVB21 = yolo_load(f'./models/MVB21/best.pt')

# Thu tuc xac nhan tinh trang hon nhan
net_MVB22 = yolo_load(f'./models/MVB22/best.pt')

# Thu tuc khai bao an toan lao dong
net_MVB23 = yolo_load(f'./models/MVB23/best.pt')

# Thu tuc giay chung nhan cua hang du dieu kien ban le xang dau
net_MVB24 = yolo_load(f'./models/MVB24/best.pt')

# Thu tuc ho so cong bo hop quy san pham, hang hoa VLXD
net_MVB25 = yolo_load(f'./models/MVB25/best.pt')

# Thu tuc bao cao lao dong nguoi nuoc ngoai
net_MVB26 = yolo_load(f'./models/MVB26/best.pt')

# Thu tuc cap giay phep lao dong nuoc ngoai
net_MVB27 = yolo_load(f'./models/MVB27/best.pt')

@app.post("/DVC/uploadFile")
async def uploadFile(textCode: Union[str, None] = None, file: UploadFile = File(...)):
    try:
        formatted_code = textCode.upper()
        model_name = f'net_{formatted_code}'
        if (model_name in globals()):
            print(f'Mo hinh {model_name}')
            pathSave = os.getcwd() + '/files'
            if (os.path.exists(pathSave)):
                with open(f'files/{file.filename}', 'wb') as buffer:
                    shutil.copyfileobj(file.file, buffer)
            else:
                os.mkdir(pathSave)
                with open(f'files/{file.filename}', 'wb') as buffer:
                    shutil.copyfileobj(file.file, buffer)
            
            return await ReturnInfoNew(f'./files/{file.filename}', formatted_code, globals()[model_name])
        else:
            rs = {
                "errorCode": 2,
                "errorMessage": "Lỗi! Mã văn bản không hợp lệ.",
                "results": []
            }
            return rs
    except Exception as e:
        save_directory = os.getcwd() + '/error_files'
        os.makedirs(save_directory,exist_ok=True)
        with open(f'error_files/{file.filename}', 'wb') as buffer:
            shutil.copyfileobj(file.file, buffer)
        rs = {
                "errorCode": 3,
                "errorMessage": str(e),
                "results": []
            }
        return rs

@app.post("/OCR/test")
def VietOCR(file: UploadFile = File(...)):
    pathSave = os.getcwd() + '/ocr_files'
    if (os.path.exists(pathSave)):
        with open(f'ocr_files/{file.filename}', 'wb') as buffer:
            shutil.copyfileobj(file.file, buffer)
    else:
        os.mkdir(pathSave)
        with open(f'ocr_files/{file.filename}', 'wb') as buffer:
            shutil.copyfileobj(file.file, buffer)
    return OCRFile(f'./ocr_files/{file.filename}')
# if __name__ == "__main__":
#     uvicorn.run(app,host='192.168.2.167', port=8005)
