import cv2
import numpy as np
from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import time
import os
import re
import glob
import fitz
import json
from pathlib import Path
from ultralytics import YOLO
import spacy
from custom import handle_textcode
''' Funtions '''

# Ham check dinh dang dau vao cua anh
def check_type_image(path):
    imgName = str(path)
    imgName = imgName[imgName.rindex('.')+1:]
    imgName = imgName.lower()
    return imgName

# Ham ve cac boxes len anh
def draw_prediction(img, classes, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes)
    color = (0, 0, 255)
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 1)
    cv2.putText(img, label, (x-5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# Ham load model vietOCr recognition
def ReturnCrop(pathImage):
    image = cv2.imread(pathImage)
    #image = resize_image(image, height=960)
    indices, boxes, classes, class_ids, image, confidences = getIndices(
        image, net, classes)
    list_boxes = []
    label = []
    for i in indices:
        #i = i[0]
        box = boxes[i]
        # print(box,str(classes[class_ids[i]]))
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        list_boxes.append([x+w/2, y+h/2])
        #draw_prediction(image, classes[class_ids[i]], confidences[i], round(x), round(y), round(x + w), round(y + h))
        label.append(str(classes[class_ids[i]]))
    #cv2.imshow('rec', resize_image(image, height=720))
    #cv2.waitKey()
    label_boxes = dict(zip(label, list_boxes))
    label_miss = find_miss_corner(label_boxes, classes)
    #Noi suy goc neu thieu 1 goc cua CCCD
    if len(label_miss) == 1:
        calculate_missed_coord_corner(label_miss, label_boxes)
        source_points = np.float32([label_boxes['top_left'], label_boxes['bottom_left'],
                                    label_boxes['bottom_right'], label_boxes['top_right']])
        crop = perspective_transoform(image, source_points)
        return crop
    elif len(label_miss)==0:
        source_points = np.float32([label_boxes['top_left'], label_boxes['bottom_left'],
                                    label_boxes['bottom_right'], label_boxes['top_right']])
        crop = perspective_transoform(image, source_points)
        return crop

# Ham load thu vien vietOCR
def vietocr_load():
    config = Cfg.load_config_from_name('vgg_transformer')
    config['weights'] = './models/transformerocr.pth'
    config['cnn']['pretrained'] = False
    config['device'] = 'cuda:0'
    config['predictor']['beamsearch'] = False
    detector = Predictor(config)
    return detector

# Ham load mo hinh Yolo
def load_model_new(yolo_path, ner_path = None):
    # Load a model
    model_yolo = YOLO(yolo_path)  # pretrained YOLOv8n model
    print(f"Load model YOLO from file {yolo_path} susscessfully..")
    model_ner = None
    if ner_path is not None:
        model_ner = spacy.load(ner_path) # Load pretrained NER model
        print(f"Load model NER from file {yolo_path} susscessfully..")
    return model_yolo, model_ner

# Ham chuyen dinh dang pdf sang dinh dang anh
def pdf_to_image(pdf_path, code):
    pdf_document = fitz.open(pdf_path)
    file_name = os.path.basename(pdf_path)
    image_url = []
    code_path = os.path.join(os.getcwd(),'pdf2img',code)
    zoom_x = 2.0  # horizontal zoom
    zoom_y = 2.0  # vertical zoom
    mat = fitz.Matrix(zoom_x, zoom_y)  # zoom factor 2 in each dimension

    for page_number in range(pdf_document.page_count):
        page = pdf_document.load_page(page_number)
        image = page.get_pixmap(matrix=mat)
        image_folder = os.path.join(code_path,'image')
        #print(image_folder)
        os.makedirs(image_folder,exist_ok=True)
        save_path = f'{image_folder}/{file_name}_{page_number}.jpg'
        #print(save_path)
        image.save(save_path)
        image_url.append(save_path)
    return image_url

# Load model, class
def load_model_old(path_weights_yolo, path_clf_yolo, path_to_class):
    weights_yolo = path_weights_yolo
    clf_yolo = path_clf_yolo
    net = cv2.dnn.readNet(weights_yolo, clf_yolo)
    with open(path_to_class, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    print(f"Load model from file {path_weights_yolo} susscessfully..")
    return net, classes

# Ham xu ly tra ve du liẹu dang dictionary
def process_image(img, engine, label_dict):
    # Thuc hien phat hien vung chua du lieu
    confidence = 0.65 # TODO : Fix cung nguong phat hien doi tuong
    start_time_yolo = time.time()
    results = engine.predict(img,conf=confidence)
    end_time_yolo = time.time()
    print(f'elapsed_time yolo: {end_time_yolo-start_time_yolo}[sec]')
    for result in results:
        boxes = result.boxes.cpu().numpy()
        start_time = time.time()
        for box in boxes:
            class_id = result.names[box.cls[0].item()]
            r = box.xyxy[0].astype(int)
            image_crop = img[r[1]:r[3], r[0]:r[2]]
            y = (r[1] + r[3]) / 2.0
            s = detector.predict(Image.fromarray(image_crop))
            label_dict[class_id].update({s: y})
        end_time = time.time()
        elapsed_time = end_time - start_time
        print ("elapsed_time ocr:{0}".format(elapsed_time) + "[sec]")
    return label_dict

# Ham tra ve thong tin tren mo hinh moi
async def ReturnInfoNew(path, text_code, engine, ner):
    # Tinh thoi gian tai thoi diem bat dau
    start_time = time.time()

    typeimage = check_type_image(path)
    classes = list(engine.names.values())
    label_dict = {key: {} for key in classes}

    # Dau vao la anh co dinh dang(*.jpg,*.jpeg,*.png,...)
    if (typeimage == 'jpg' or typeimage == 'png' or typeimage == 'jpeg'):
        img = cv2.imread(path)
        label_dict.update(process_image(img, engine, label_dict))                                    
    
    # Dau vao la file pdf dang(*.pdf)
    elif(typeimage == 'pdf'):
        image_url = pdf_to_image(path, text_code)
        for url in image_url:
            img = cv2.imread(url)
            label_dict.update(process_image(img, engine, label_dict))
    
    # Dau vao khong dung dinh dang
    else:
        rs = {
            "errorCode": 1,
            "errorMessage": "Lỗi! File không đúng định dạng.",
            "results": []
        }
        return rs
    
    # Gop cac thong tin vao tu dien
    for key, value in label_dict.items():
        if len(value) >=2: # Gop du lieu
            sorted_items = sorted(value.items(), key = lambda x:x[1])
            merged_value = ' '.join([k for k,v in sorted_items])
            label_dict[key] = merged_value
        elif not value: # Du lieu Null
            label_dict[key] = None
        else:
            label_dict[key] = list(value.keys())[0]
    
    # Ham xu ly custom du lieu theo ma van ban 
    label_dict = handle_textcode(label_dict, text_code, ner)

    # Tinh thoi gian tai thoi diem ket thuc thuat toan
    end_time = time.time()
    
    # Tinh tong thoi gian chay
    elapsed_time = end_time - start_time

    # Tra ve ket qua sau khi duyet qua tat ca cac anh
    rs = {
        "errorCode": 0,
        "errorMessage": "",
        "executionTime": round(elapsed_time,2),
        "results": [label_dict]
    }
    return rs

# Ham tra ve thong tin OCR voi dau vao la 1 anh
def OCRFile(file_path):
    if not os.path.exists(file_path):
        return f'Duong dan file {file_path} khong hop le !'
    im_gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    #thresh, im_bw = cv2.threshold(im_gray, 10, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #cv2.imshow('image_bw', im_bw)
    cv2.waitKey(0)
    s = detector.predict(Image.fromarray(im_gray))
    # print(s)
    return f'OCR results: {s}'

detector = vietocr_load()
# engine = yolo_load('./models/MVB1/best.pt')

# if __name__ == "__main__":
    # path = '/home/polaris/ml/DVC/OCR-DVC-ThanhHoa/ocr_files/Screenshot_2.png'
    # OCRFile(path)