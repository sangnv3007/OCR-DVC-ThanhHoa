import cv2
import numpy as np
from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import time
import os
import base64
import re
import glob
### Funtions
# Ham decode, endecode
def EncodeImage(pathImageEncode):
    with open(pathImageEncode, 'rb') as binary_file:
        binary_file_data = binary_file.read()
        base64_encoded_data = base64.b64encode(binary_file_data)
        base64_message = base64_encoded_data.decode('utf-8')
        return base64_message
def EndecodeImage(base64_img):
    base64_img_bytes = base64_img.encode('utf-8')
    with open('decoded_image.png', 'wb') as file_to_save:
        decoded_image_data = base64.decodebytes(base64_img_bytes)
        file_to_save.write(decoded_image_data)
def resize_image(inputImg, width = 0, height = 0):
    (new_w, new_h)  = (0,0)
    (w, h) = (inputImg.shape[1], inputImg.shape[0])
    if(width == 0 and height == 0):
        return inputImg
    if (width == 0):
        r = height / float(h)
        new_w = int(w * r)
        new_h = height
    else:
        r = width / float(w)
        new_w = width
        new_h = int(h * r)
    imageResize = cv2.resize(inputImg, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
    return imageResize
# Ham check dinh dang dau vao cua anh
def check_type_image(path):
    imgName = str(path)
    imgName = imgName[imgName.rindex('.')+1:]
    imgName = imgName.lower()
    return imgName
#Ham ve cac boxes len anh
def draw_prediction(img, classes, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes)
    color = (0, 0, 255)
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x-5 , y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
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
# Ham load model yolo
def load_model(path_weights_yolo, path_clf_yolo, path_to_class):
    weights_yolo = path_weights_yolo
    clf_yolo = path_clf_yolo
    net = cv2.dnn.readNet(weights_yolo, clf_yolo)
    with open(path_to_class, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return net, classes
# Ham getIndices
def getIndices(image, net, classes):
    (Width, Height) = (image.shape[1], image.shape[0])
    boxes = []
    class_ids = []
    confidences = []
    conf_threshold = 0.5
    nms_threshold = 0.4
    scale = 0.00392
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
def vietocr_load():
    config = Cfg.load_config_from_name('vgg_transformer')
    config['weights'] = './model/transformerocr.pth'
    config['cnn']['pretrained'] = False
    config['device'] = 'cpu'
    config['predictor']['beamsearch'] = False
    detector = Predictor(config)
    return detector
# Crop image tu cac boxes
def ReturnInfoCard(path):
    typeimage = check_type_image(path) 
    if(typeimage!='png' and typeimage!='jpeg' and typeimage!='jpg' and typeimage != 'bmp'):
        obj = MessageInfo(1, 'Invalid image file! Please try again.')
        return obj
    else:
        image = cv2.imread(path)
        if(image is not None):
            image = resize_image(image, height=1080)
            #print(image.shape[1], image.shape[0])
            # cv2.imshow('image_input', image)
            # cv2.waitKey()
            indices, boxes, classes, class_ids, image, confidences  = getIndices(image, net_det, classes_det)
            # print(indices)
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
                list_boxes.append([x+ w/2, y + h/2])
                label.append(str(classes[class_ids[i]]))
                draw_prediction(image, classes[class_ids[i]], confidences[i], round(x), round(y), round(x + w), round(y + h)) #Ve cac class len anh
            #cv2.imshow('image', image)
            #cv2.waitKey()
            label_boxes = dict(zip(label, list_boxes))
            # print(label_boxes)
            if (check_enough_labels(label_boxes, classes)):
                source_points = np.float32([label_boxes['top_left'], label_boxes['bottom_left'],
                                        label_boxes['bottom_right'], label_boxes['top_right']])
                crop = perspective_transoform(image, source_points)
                indices, boxes, classes, class_ids, image, confidences = getIndices(crop, net_rec, classes_rec)
                home_text, issued_by_text = [], []
                label_boxes = []
                imgCrop = np.zeros((100, 100, 3), dtype=np.uint8)
                dict_var = {'id':{}, 'name': {},'dob':{}, 'home':{},
                            'join_date':{},'official_date':{}, 'issued_by':{}, 'issue_date':{}}
                for i in indices:
                    #i = i[0]
                    box = boxes[i]
                    x = box[0]
                    y = box[1]
                    w = box[2]
                    h = box[3]
                    label_boxes.append(classes[class_ids[i]])
                    #draw_prediction(crop, classes[class_ids[i]], confidences[i], round(x), round(y), round(x + w), round(y + h))
                    imageCrop = image[round(y): round(y + h), round(x):round(x + w)]
                    s = detector.predict(Image.fromarray(imageCrop))
                    if(class_ids[i] == 8):
                        imgCrop = imageCrop
                    else: dict_var[classes[class_ids[i]]].update({s:y})
                #cv2.imshow('crop', crop)
                #cv2.waitKey()
                for i in classes:
                    bool = i in label_boxes
                    if(bool == False):
                        dict_var[i].update({'N/A':0})
                errorCode = 0
                errorMessage = ""
                for i in sorted(dict_var['home'].items(),
                                key=lambda item: item[1]): home_text.append(i[0])
                for i in sorted(dict_var['issued_by'].items(
                ), key=lambda item: item[1]): issued_by_text.append(i[0])
                home_text = " ".join(home_text)
                issued_by_text = " ".join(issued_by_text)
                pathSave = os.getcwd() + '\\dangvien\\'
                stringImage = "dangvien" + '_' + str(time.time()) + ".jpg"
                if (os.path.exists(pathSave)):
                    cv2.imwrite(pathSave + stringImage, imgCrop)
                else:
                    os.mkdir(pathSave)
                    cv2.imwrite(pathSave + stringImage, imgCrop)
                obj = ExtractCard(list(dict_var['id'].keys())[0], list(dict_var['name'].keys())[0], list(dict_var['dob'].keys())[0], home_text,
                                    list(dict_var['join_date'].keys())[0], list(dict_var['official_date'].keys())[0], issued_by_text, 
                                    list(dict_var['issue_date'].keys())[0], stringImage, errorCode, errorMessage)
                return obj
            else:
                obj = MessageInfo(4, "Error! Membership Card not found !")
                return obj
        else:
            obj = MessageInfo(2, "Input image not found! Check and try again.")
            return obj
detector = vietocr_load()
net_det, classes_det = load_model('./model/det/yolov4-tiny-custom_det.weights', './model/det/yolov4-tiny-custom_det.cfg', './model/det/obj.names')
net_rec, classes_rec= load_model('./model/rec/yolov4-custom_rec.weights', './model/rec/yolov4-custom_rec.cfg', './model/rec/obj.names')
class ExtractCard:
    def __init__(self, id, name, dob, home, join_date, official_date, issued_by, issue_date, image, errorCode, errorMessage):
        self.id = id
        self.name = name
        self.dob = dob
        self.home = home
        self.join_date = join_date
        self.official_date = official_date
        self.issued_by = issued_by
        self.issue_date = issue_date
        self.image = image
        self.errorCode = errorCode
        self.errorMessage = errorMessage
class MessageInfo:
    def __init__(self, errorCode, errorMessage):
        self.errorCode = errorCode
        self.errorMessage = errorMessage
#obj = ReturnInfoCard('C:\\Users\\Sang TL\\Downloads\\Telegram Desktop\\photo_2022-11-05_14-59-29.jpg')
#print(obj.errorCode, obj.errorMessage)
#Crop anh 
# path = 'D:\Download Chorme\Members\Detect_edge\obj'
# i=199
# for filename in glob.glob(os.path.join(path, '*.jpg')):
#     print(filename)
#     imageCrop = ReturnInfoCard(filename)
#     if(imageCrop is not None):
#         cv2.imwrite('D:\Download Chorme\Members\Detect_text\CropMCVR\MembershipCrop'+str(i)+'.jpg', imageCrop)
#         i = i + 1