from fastapi import FastAPI, File, UploadFile, status, Body
from pydantic import BaseModel
from typing import Union
from process import ReturnInfoCard
import json
import uvicorn
import os
import shutil
import base64
import time
import datetime
import asyncio
from thongke import create_connection, select_one_tasks, update_record, select_one_tasks_with_time, insert_record, select_time_last_record
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
counter_lock_total = asyncio.Lock()
counter_lock_successful = asyncio.Lock()
counter_lock_failed = asyncio.Lock()
counter_lock_invalid = asyncio.Lock()
conn = create_connection('./model/thongke.pb')
# total_request = 0
# successful = 0
# failed = 0
# invalid_image = 0
@app.get("/MembershipCard/thongke")
async def thongke():
    data = select_one_tasks(conn)
    return data
@app.get("/MembershipCard/thongketheongay")
async def thongketheongay(datetime: Union[str, None] = datetime.datetime.now().date()):
    #print(datetime)
    time,total_request, successful, failed, invalid_image= select_one_tasks_with_time(conn, str(datetime))
    return {"time":time,"total_request": total_request, "successful": successful, "failed": failed, 'invalid_image': invalid_image}
@app.post("/MembershipCard/uploadBase64")
async def uploadBase64(item: Item): 
    global total_request
    global successful
    global failed
    global invalid_image
    last_time_in_db = select_time_last_record(conn)
    date_now = str(datetime.datetime.now().date())
    if(date_now != last_time_in_db):
        insert_record(conn,date_now, 0, 0, 0, 0)
    time_record, total_request, successful, failed, invalid_image= select_one_tasks_with_time(conn, date_now)
    try:
        async with counter_lock_total:
            total_request += 1           
        title, etx = os.path.splitext(os.path.basename(item.name))      
        if(item.name is None): item.name= 'dangvien' +'_'+ str(time.time()) + '.jpg'
        image_as_bytes = str.encode(item.stringbase64)# convert string to bytes
        img_recovered = base64.b64decode(image_as_bytes)  # decode base64string
        pathSave = os.getcwd() +'/'+ 'anhthe'
        nameSave = title +'.jpg'
        if (os.path.exists(pathSave)):
            with open(f'anhthe/{nameSave}', "wb") as f:
                f.write(img_recovered)
        else:
            os.mkdir(pathSave)
            with open(f'anhthe/{nameSave}', "wb") as f:
                f.write(img_recovered)
        obj = await ReturnInfoCard(f'anhthe/{nameSave}')
        if (obj.errorCode == 0):
            async with counter_lock_successful:
                successful += 1
            return {"errorCode": obj.errorCode, "errorMessage": obj.errorMessage,
                    "data": [{"id": obj.id, "name": obj.name.upper(), "dob": obj.dob, "home": obj.home, "join_date": obj.join_date,
                            "official_date": obj.official_date, "issued_by": obj.issued_by, "issue_date": obj.issue_date,
                            "image": obj.image}]}
        else:
            async with counter_lock_invalid:
                invalid_image += 1
                shutil.move(f'anhthe/{nameSave}', f'invalid-image/{nameSave}')
            return {"errorCode": obj.errorCode, "errorMessage": obj.errorMessage, "data": []}
    except:
        async with counter_lock_failed:
                failed += 1
    finally:
        update_record(conn,total_request, successful, failed, invalid_image,date_now)

@app.post("/MembershipCard/uploadFile")
def uploadFile(file: UploadFile = File(...)):
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
#if __name__ == "__main__":
    #uvicorn.run(app,host='192.168.2.167', port=8002)
    
