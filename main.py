from fastapi import FastAPI, File, UploadFile
from process import ReturnInfoCard, ExtractCard, MessageInfo
import json
import os
import shutil
from starlette import status
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

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
app.add_middleware(LimitUploadSize, max_upload_size=3000000)  # ~3MB
@app.post("/MembershipCard/upload")
def upload(file: UploadFile = File(...)):
    try:
        pathSave = os.getcwd() + '\\anhthe'
        if (os.path.exists(pathSave)):
            with open(f'anhthe/{file.filename}','wb') as buffer:
                shutil.copyfileobj(file.file, buffer)
        else:
            os.mkdir(pathSave)
            with open(f'anhthe/{file.filename}','wb') as buffer:
                shutil.copyfileobj(file.file, buffer)
        obj = ReturnInfoCard(f'anhthe/{file.filename}')
        if(obj.errorCode==0):
            return {"status": obj.errorCode,"message": obj.errorMessage, 
                    "data": [{"id": obj.id,"name": obj.name,"dob": obj.dob,"home": obj.home,"join_date":obj.join_date,
                    "issued_by":obj.issued_by,"issue_date":obj.issue_date,"image":obj.image}]}
        else:
            return {"status": obj.errorCode,"message": obj.errorMessage, "data": []}
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()
    return {"message": f"Successfuly uploaded {file.filename}"}
