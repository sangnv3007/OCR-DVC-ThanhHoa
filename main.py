from fastapi import FastAPI, File, UploadFile, status, Body
from pydantic import BaseModel
from typing import Union
from process import ReturnInfoCard, load_model_init
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

app = FastAPI()
app.add_middleware(LimitUploadSize, max_upload_size=10000000)  # ~10MB

# Load mo hinh 


@app.post("/DVC/uploadFile")
def uploadFile(code: Union[str, None] = None, file: UploadFile = File(...)):
    pathSave = os.getcwd() + '/file'
    if (os.path.exists(pathSave)):
        with open(f'file/{file.filename}', 'wb') as buffer:
            shutil.copyfileobj(file.file, buffer)
    else:
        os.mkdir(pathSave)
        with open(f'file/{file.filename}', 'wb') as buffer:
            shutil.copyfileobj(file.file, buffer)
    load_model_init('MVB1')
    return ReturnInfoCard(f'./file/{file.filename}', code)

# if __name__ == "__main__":
#     uvicorn.run(app,host='192.168.2.167', port=8005)
