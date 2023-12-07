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
# Funtions

#Ham check miss_conner
def find_miss_corner(labels, classes):
    labels_miss = []
    for i in classes:
        bool = i in labels
        if(bool == False):
            labels_miss.append(i)
    return labels_miss
#Ham tinh toan miss_conner
def calculate_missed_coord_corner(label_missed, coordinate_dict):
    thresh = 0
    if(label_missed[0]=='top_left'):
        midpoint = np.add(coordinate_dict['top_right'], coordinate_dict['bottom_left']) / 2
        y = 2 * midpoint[1] - coordinate_dict['bottom_right'][1] - thresh
        x = 2 * midpoint[0] - coordinate_dict['bottom_right'][0] - thresh
        coordinate_dict['top_left'] = (x, y)
    elif(label_missed[0]=='top_right'):
        midpoint = np.add(coordinate_dict['top_left'], coordinate_dict['bottom_right']) / 2
        y = 2 * midpoint[1] - coordinate_dict['bottom_left'][1] - thresh
        x = 2 * midpoint[0] - coordinate_dict['bottom_left'][0] - thresh
        coordinate_dict['top_right'] = (x, y)
    elif(label_missed[0]=='bottom_left'):
        midpoint = np.add(coordinate_dict['top_left'], coordinate_dict['bottom_right']) / 2
        y = 2 * midpoint[1] - coordinate_dict['top_right'][1] - thresh
        x = 2 * midpoint[0] - coordinate_dict['top_right'][0] - thresh
        coordinate_dict['bottom_left'] = (x, y)
    elif(label_missed[0]=='bottom_right'):
        midpoint = np.add(coordinate_dict['bottom_left'], coordinate_dict['top_right']) / 2
        y = 2 * midpoint[1] - coordinate_dict['top_left'][1] - thresh
        x = 2 * midpoint[0] - coordinate_dict['top_left'][0] - thresh
        coordinate_dict['bottom_right'] = (x, y)
    return coordinate_dict
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
# Transform sang toa do dich


def perspective_transoform(image, points):
    # Use L2 norm
    width_AD = np.sqrt(
        ((points[0][0] - points[3][0]) ** 2) + ((points[0][1] - points[3][1]) ** 2))
    width_BC = np.sqrt(
        ((points[1][0] - points[2][0]) ** 2) + ((points[1][1] - points[2][1]) ** 2))
    maxWidth = max(int(width_AD), int(width_BC))  # Get maxWidth
    height_AB = np.sqrt(
        ((points[0][0] - points[1][0]) ** 2) + ((points[0][1] - points[1][1]) ** 2))
    height_CD = np.sqrt(
        ((points[2][0] - points[3][0]) ** 2) + ((points[2][1] - points[3][1]) ** 2))
    maxHeight = max(int(height_AB), int(height_CD))  # Get maxHeight

    output_pts = np.float32([[0, 0],
                             [0, maxHeight - 1],
                             [maxWidth - 1, maxHeight - 1],
                             [maxWidth - 1, 0]])
    # Compute the perspective transform M
    M = cv2.getPerspectiveTransform(points, output_pts)
    out = cv2.warpPerspective(
        image, M, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)
    return out
# Ham check classes


def check_enough_labels(labels, classes):
    for i in classes:
        bool = i in labels
        if bool == False:
            return False
    return True

# Ham khoi tao model
def load_model_init(text_code):
    global net, classes, object_labels
    if (net is None):
        net, classes, object_labels = load_model(f'./{text_code}/model/yolov4-custom.weights',
                                  f'./{text_code}/model/yolov4-custom.cfg',
                                  f'./{text_code}/model/obj.names')

# Ham load model yolo
def load_model(path_weights_yolo, path_clf_yolo, path_to_class):
    weights_yolo = path_weights_yolo
    clf_yolo = path_clf_yolo
    net = cv2.dnn.readNet(weights_yolo, clf_yolo)
    object_labels, classes = read_labels(path_to_class)
    return net, classes, object_labels

# Ham getIndices
def getIndices(image, net, classes):
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
    config['device'] = 'cpu'
    config['predictor']['beamsearch'] = False
    detector = Predictor(config)
    return detector

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
        save_path = f'{code}/image/{file_name}_{page_number}.jpg'
        image.save(save_path)
        image_url.append(save_path)
    return image_url

# Init object label
def read_labels(file_name):
    classes = []
    labels = ExtractedInformation()
    with open(file_name, 'r') as file:
        lines = file.readlines()
        for line in lines:
            label_name = line.strip()
            classes.append(label_name)
            setattr(labels, label_name, None)
        #Init Messsage Information
        setattr(labels, "errorCode", None)
        setattr(labels, "errorMessage", None)
    return labels, classes

# Crop image tu cac boxes
def ReturnInfoCard(path, text_code):
    print(path)
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
        for url in image_url:    
            file_name = Path(url).stem
            image_data = cv2.imread(url)
            indices, boxes, labels, class_ids, image, confidences = getIndices(
            image_data, net, classes)
            label_dict = {key: {} for key in classes}
            if len(indices) > 0:
                for i in indices:
                    # i = i[0]
                    box = boxes[i]
                    x, y, w, h = box[0], box[1], box[2], box[3]
                    #draw_prediction(image, classes[class_ids[i]], confidences[i], round(x), round(y), round(x + w), round(y + h))
                    imageCrop = image[round(y): round(y + h), round(x):round(x + w)]    
                    # #start = time.time()       
                    s = detector.predict(Image.fromarray(imageCrop))
                    #print(s)
                    # # end = time.time()
                    # # total_time = end - start
                    # # print(str(round(total_time, 2)) + ' [sec]')
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
                # for key,value in label_dict.items():
                #     if hasattr(object_labels, key):  # Kiểm tra xem thuộc tính tồn tại trong đối tượng không
                #         setattr(object_labels, key, value)  # Gán giá trị từ từ điển vào thuộc tính của đối tượng
                # setattr(object_labels, "errorCode", 0)
                # setattr(object_labels, "errorMessage", "")
                rs = {
                    "errorCode": 0,
                    "errorMessage": "",
                    "results": [label_dict]
                }
                return rs

net = None
classes = None
object_labels = None
detector = vietocr_load()

class ExtractedInformation:
    def __init__(self):
        pass

class MessageInfo:
    def __init__(self, errorCode, errorMessage):
        self.errorCode = errorCode
        self.errorMessage = errorMessage

# if(obj.errorCode==0): print('Load model successful !')
# Crop anh
# path = 'D:\Download Chorme\Members\Detect_edge\obj'
# i=199
# for filename in glob.glob(os.path.join(path, '*.jpg')):
#     print(filename)
#     imageCrop = ReturnInfoCard(filename)
#     if(imageCrop is not None):
#         cv2.imwrite('D:\Download Chorme\Members\Detect_text\CropMCVR\MembershipCrop'+str(i)+'.jpg', imageCrop)
#         i = i + 1

if __name__ == "__main__":
    load_model_init('MVB1')
    # load_model_init('MVB1')
    obj = ReturnInfoCard('MVB1/CCTBDT (91).pdf', 'MVB1')