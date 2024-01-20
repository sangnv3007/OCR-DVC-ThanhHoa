from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator  # ultralytics.yolo.utils.plotting is deprecated

# Load a model
model = YOLO('./weights/best.pt')  # pretrained YOLOv8n model
img = cv2.imread('/home/polaris/ml/DVC/DVC_ThanhHoa/MVB3/image/1M84bQQJ8kO7oORL (1).pdf_0.jpg')
results = model.predict(img)

def draw_results(img_source):
    for r in results:
        annotator = Annotator(img_source)
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
            c = box.cls
            annotator.box_label(b, model.names[int(c)])
        
    img_source = annotator.result()  
    return img_source

for result in results:                                         # iterate results
    boxes = result.boxes.cpu().numpy()                        # get boxes on cpu in numpy
    for box in boxes: 
        class_id = result.names[box.cls[0].item()]  
        print(class_id)                                       # iterate boxes
        r = box.xyxy[0].astype(int)                            # get corner points as int 
        crop = img[r[1]:r[3], r[0]:r[2]]
        cv2.imshow('Cropped', crop)
        cv2.waitKey(0)

# for result in results:
#     boxes = result.boxes.cpu().numpy()
#     for i, box in enumerate(boxes):
#         r = box.xyxy[0].astype(int)
#         crop = img[r[1]:r[3], r[0]:r[2]]
#         cv2.imwrite(str(i) + ".jpg", crop)

# cv2.imshow('YOLO V8 Detection', im1)
# cv2.waitKey(0)