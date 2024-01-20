from turtle import rt
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
''' Funtions '''

# Ham resize image
def resize_image(inputImg, width=0, height=0):
    (new_w, new_h) = (0, 0)
    (w, h) = (inputImg.shape[1], inputImg.shape[0])
    if (width == 0 and height == 0):
        return inputImg
    if (width == 0):
        r = height / float(h)
        new_w = int(w * r)
        new_h = height
    else:
        r = width / float(w)
        new_w = width
        new_h = int(h * r)
    imageResize = cv2.resize(inputImg, (new_w, new_h),
                             interpolation=cv2.INTER_AREA)
    return imageResize

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

# Ham get output_layer
def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1]
                     for i in net.getUnconnectedOutLayers()]
    return output_layers

# Ham getIndices
async def getIndices(image, net, classes):
    (Width, Height) = (image.shape[1], image.shape[0])
    boxes = []
    class_ids = []
    confidences = []
    conf_threshold = 0.5
    nms_threshold = 0.5
    scale = 1/255
    # print(classes)
    # (416,416) img target size, swapRB=True,  # BGR -> RGB, center crop = False
    blob = cv2.dnn.blobFromImage(
        image, scale, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    indices = cv2.dnn.NMSBoxes(
        boxes, confidences, conf_threshold, nms_threshold)
    return indices, boxes, classes, class_ids, image, confidences
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
    config['weights'] = 'transformerocr.pth'
    config['cnn']['pretrained'] = False
    config['device'] = 'cuda:0'
    config['predictor']['beamsearch'] = False
    detector = Predictor(config)
    return detector

# Ham load mo hinh Yolo
def yolo_load():
    # Load a model
    model = YOLO('./weights/best.pt')  # pretrained YOLOv8n model
    return model

# Ham chuyen dinh dang pdf sang dinh dang anh
def pdf_to_image(pdf_path, code):
    pdf_document = fitz.open(pdf_path)
    file_name = os.path.basename(pdf_path)
    image_url = []
    code_path = os.path.join(os.getcwd(), code)
    if (not os.path.exists(code_path)):
        return image_url
    for page_number in range(pdf_document.page_count):
        page = pdf_document.load_page(page_number)
        image = page.get_pixmap()
        image_folder = os.path.join(code_path,'image')
        if(not os.path.exists(image_folder)):
            os.mkdir(image_folder)
        save_path = f'{code}/image/{file_name}_{page_number}.jpg'
        image.save(save_path)
        image_url.append(save_path)
    return image_url

# Load model, class
def load_model(path_weights_yolo, path_clf_yolo, path_to_class):
    weights_yolo = path_weights_yolo
    clf_yolo = path_clf_yolo
    net = cv2.dnn.readNet(weights_yolo, clf_yolo)
    with open(path_to_class, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    print(f"Load model from file {path_weights_yolo} susscessfully..")
    return net, classes

def get_data_model(folder_name):
    # Ly lich tu phap
    net_MVB3, classes_MVB3 = load_model(f'./MVB3/model/yolov4-custom.weights',
                                  f'./MVB3/model/yolov4-custom.cfg',
                                  f'./MVB3/model/obj.names')
    for filename in glob.glob(os.path.join(folder_name, '*.pdf')):
        print(f'Processing file: {filename}')
        file_name = Path(filename).stem
        rs = ReturnInfDataNER(filename, 'MVB3', net_MVB3, classes_MVB3)
        # Đường dẫn tới file .txt bạn muốn lưu dữ liệu vào
        file_path = f"{filename}_label.txt"
        # Mở file .txt trong chế độ ghi ('w' là chế độ ghi, 'a' để thêm vào cuối file)
        with open(file_path, 'w') as file:
            if (rs != ''):
                file.write(rs)  # Ghi từng dòng dữ liệu vào file,
        print(f'=> Done file: {filename}')
        print('----------------------------------------------------------------')
    print('Completed !!!')

# Ham test mo hinh
def ReturnInfoTest(path, text_code, net, classes):
    typeimage = check_type_image(path)
    if (typeimage != 'pdf'):
        rs = {
            "errorCode": 1,
            "errorMessage": "Lỗi! File không đúng định dạng.",
            "results": []
        }
        return rs
    else:
        image_url = pdf_to_image(path, text_code)
        label_dict = {key: {} for key in classes}
        rs = ''
        for url in image_url:
            file_name = Path(url).stem
            image_data = cv2.imread(url)
            indices, boxes, labels, class_ids, image, confidences = getIndices(
            image_data, net, classes)
            if len(indices) > 0:
                for i in indices:
                    # i = i[0]
                    box = boxes[i]
                    x, y, w, h = box[0], box[1], box[2], box[3]
                    draw_prediction(image, classes[class_ids[i]], confidences[i], round(x), round(y), round(x + w), round(y + h))
                    imageCrop = image[round(y): round(y + h), round(x):round(x + w)]
                    s = detector.predict(Image.fromarray(imageCrop))
                    # cv2.imshow('rec', imageCrop)
                    # cv2.waitKey(0)
                    print(s)
                    # rs = rs + s + '\n'
                    #label_dict[classes[class_ids[i]]].update({s: y})
            cv2.imshow('rec', image)
            cv2.waitKey()
        return rs

# Ham test mo hinh
def ReturnInfDataNER(path, text_code, net, classes):
    typeimage = check_type_image(path)
    if (typeimage != 'pdf'):
        rs = {
            "errorCode": 1,
            "errorMessage": "Lỗi! File không đúng định dạng.",
            "results": []
        }
        return rs
    else:
        image_url = pdf_to_image(path, text_code)
        label_dict = {key: {} for key in classes}
        rs = ''
        for url in image_url:
            file_name = Path(url).stem
            image_data = cv2.imread(url)
            indices, boxes, labels, class_ids, image, confidences = getIndices(
            image_data, net, classes)
            if len(indices) > 0:
                for i in indices:
                    # i = i[0]
                    box = boxes[i]
                    x, y, w, h = box[0], box[1], box[2], box[3]
                    #draw_prediction(image, classes[class_ids[i]], confidences[i], round(x), round(y), round(x + w), round(y + h))
                    imageCrop = image[round(y): round(y + h), round(x):round(x + w)]
                    s = detector.predict(Image.fromarray(imageCrop))
                    label_dict[classes[class_ids[i]]].update({s: y})
                # Gop cac thong tin vao tu dien
        for key, value in label_dict.items():
            if len(value) >=2: # Gop du lieu
                sorted_items = sorted(value.items(), key = lambda x:x[1])
                merged_value = ' '.join([k for k,v in sorted_items])
                label_dict[key] = merged_value
            elif not value: # Du lieu Null
                label_dict[key] = ""
            else:
                label_dict[key] = list(value.keys())[0]
        for key, value in label_dict.items():
            if (value != ""):
                rs += value + '\n'
        return rs

# Hma tra ve thong tin anh
async def ReturnInfo(path, text_code, net, classes):
    typeimage = check_type_image(path)
    if (typeimage != 'pdf'):
        rs = {
            "errorCode": 1,
            "errorMessage": "Lỗi! File không đúng định dạng.",
            "results": []
        }
        return rs
    else:
        image_url = pdf_to_image(path, text_code)
        label_dict = {key: {} for key in classes}
        for url in image_url:
            file_name = Path(url).stem
            image_data = cv2.imread(url)
            indices, boxes, labels, class_ids, image, confidences = await getIndices(
            image_data, net, classes)
            if len(indices) > 0:
                for i in indices:
                    # i = i[0]
                    box = boxes[i]
                    x, y, w, h = box[0], box[1], box[2], box[3]
                    #draw_prediction(image, classes[class_ids[i]], confidences[i], round(x), round(y), round(x + w), round(y + h))
                    imageCrop = image[round(y): round(y + h), round(x):round(x + w)]
                    s = detector.predict(Image.fromarray(imageCrop))
                    label_dict[classes[class_ids[i]]].update({s: y})
        # Gop cac thong tin vao tu dien
        for key, value in label_dict.items():
            if len(value) >=2: # Gop du lieu
                sorted_items = sorted(value.items(), key = lambda x:x[1])
                merged_value = ' '.join([k for k,v in sorted_items])
                label_dict[key] = merged_value
            elif not value: # Du lieu Null
                label_dict[key] = "NaN"
            else:
                label_dict[key] = list(value.keys())[0]
        # Tra ve ket qua sau khi duyet qua tat ca cac anh
        rs = {
            "errorCode": 0,
            "errorMessage": "",
            "results": [label_dict]
        }
        return rs
def ReturnInfoNew(path):
    typeimage = check_type_image(path)
    if (typeimage != 'pdf'):
        rs = {
            "errorCode": 1,
            "errorMessage": "Lỗi! File không đúng định dạng.",
            "results": []
        }
        return rs
    else:
        img = cv2.imread(path)
        results = engine.predict(img)
        class_names = results.names.values()
        for result in results:                                      
            boxes = result.boxes.cpu().numpy()                        # get boxes on cpu in numpy
            for box in boxes: 
                class_id = result.names[box.cls[0].item()]  
                print(class_id)                                       # iterate boxes
                r = box.xyxy[0].astype(int)                            # get corner points as int 
                crop = img[r[1]:r[3], r[0]:r[2]]

                # cv2.imshow('Cropped', crop)
                # cv2.waitKey(0)

detector = vietocr_load()
engine = yolo_load()

if __name__ == "__main__":
    path = './MVB3/image/1M84bQQJ8kO7oORL (1).pdf_0.jpg'
    ReturnInfoNew(path)