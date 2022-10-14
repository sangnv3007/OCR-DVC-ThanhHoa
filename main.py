from fastapi import FastAPI, File, UploadFile
from process import ReturnInfoCard, ExtractCard, MessageInfo
import json
app = FastAPI()

@app.post("/MembershipCard/upload")
def upload(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        with open(file.filename, 'wb') as f:
            f.write(contents)
            obj = ReturnInfoCard(file.filename, True)
            if(obj.status!="failed"):
                return {"id": obj.id,"name": obj.name,"dob": obj.dob,"home": obj.home,"join_date":obj.join_date,
                        "issued_by":obj.issued_by,"issue_date":obj.issue_date,"image":obj.image,"status": obj.status,"message": obj.message}
            else:
                return {"status": obj.status,"message": obj.message}
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()
    return {"message": f"Successfuly uploaded {file.filename}"}
