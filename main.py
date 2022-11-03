from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
from typing import Union
from process import ReturnInfoCard
import json
import os
import shutil
import base64
from starlette import status
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp
class Item(BaseModel):
    name: str
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
app.add_middleware(LimitUploadSize, max_upload_size=4000000)  # ~3MB

@app.post("/MembershipCard/uploadBase64")
def uploadBase64(item: Item):
    try:
        image_as_bytes = str.encode(item.stringbase64)  # convert string to bytes
        img_recovered = base64.b64decode(image_as_bytes)  # decode base64string
        pathSave = os.getcwd() +'\\'+ 'anhthe'
        if (os.path.exists(pathSave)):
            with open(f'anhthe\\{item.name}', "wb") as f:
                f.write(img_recovered)
        else:
            os.mkdir(pathSave)
            with open(f'anhthe\\{item.name}', "wb") as f:
                f.write(img_recovered)
        obj = ReturnInfoCard(f'anhthe\\{item.name}')
        if (obj.errorCode == 0):
            return {"errorCode": obj.errorCode, "errorMessage": obj.errorMessage,
                    "data": [{"id": obj.id, "name": obj.name, "dob": obj.dob, "home": obj.home, "join_date": obj.join_date,
                              "official_date": obj.official_date, "issued_by": obj.issued_by, "issue_date": obj.issue_date, "image": obj.image}]}
        else:
            return {"errorCode": obj.errorCode, "errorMessage": obj.errorMessage, "data": []}
    except Exception:
        return {"message": "There was an error uploading the file"}
    return {"message": f"Successfuly uploaded {item.name}"}
@app.post("/MembershipCard/uploadFile")
def uploadFile(file: UploadFile = File(...)):
    try:
        pathSave = os.getcwd() + '/anhthe'
        if (os.path.exists(pathSave)):
            with open(f'anhthe/{file.filename}', 'wb') as buffer:
                shutil.copyfileobj(file.file, buffer)
        else:
            os.mkdir(pathSave)
            with open(f'anhthe/{file.filename}', 'wb') as buffer:
                shutil.copyfileobj(file.file, buffer)
        obj = ReturnInfoCard(f'anhthe/{file.filename}')
        if (obj.errorCode == 0):
            return {"errorCode": obj.errorCode, "errorMessage": obj.errorMessage,
                    "data": [{"id": obj.id, "name": obj.name, "dob": obj.dob, "home": obj.home, "join_date": obj.join_date,
                              "official_date": obj.official_date, "issued_by": obj.issued_by, "issue_date": obj.issue_date, "image": obj.image}]}
        else:
            return {"errorCode": obj.errorCode, "errorMessage": obj.errorMessage, "data": []}
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()
    return {"message": f"Successfuly uploaded {file.filename}"}