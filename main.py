from fastapi import FastAPI, File, UploadFile
from process import ReturnInfoCard, ExtractCard, MessageInfo
import json
import os
import shutil
app = FastAPI()

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
