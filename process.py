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
    config['weights'] = './models/transformerocr.pth'
    config['cnn']['pretrained'] = False
    config['device'] = 'cuda:0'
    config['predictor']['beamsearch'] = False
    detector = Predictor(config)
    return detector

# Ham load mo hinh Yolo
def yolo_load(model_path):
    # Load a model
    model = YOLO(model_path)  # pretrained YOLOv8n model
    print(f"Load model from file {model_path} susscessfully..")
    return model

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
def load_model(path_weights_yolo, path_clf_yolo, path_to_class):
    weights_yolo = path_weights_yolo
    clf_yolo = path_clf_yolo
    net = cv2.dnn.readNet(weights_yolo, clf_yolo)
    with open(path_to_class, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    print(f"Load model from file {path_weights_yolo} susscessfully..")
    return net, classes

# Ham lay ket qua detect dua vao file *.txt
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

# Ham loai bo dau tieng viet
def no_accent_vietnamese(s):
    s = re.sub('[áàảãạăắằẳẵặâấầẩẫậ]', 'a', s)
    s = re.sub('[ÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬ]', 'A', s)
    s = re.sub('[éèẻẽẹêếềểễệ]', 'e', s)
    s = re.sub('[ÉÈẺẼẸÊẾỀỂỄỆ]', 'E', s)
    s = re.sub('[óòỏõọôốồổỗộơớờởỡợ]', 'o', s)
    s = re.sub('[ÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢ]', 'O', s)
    s = re.sub('[íìỉĩị]', 'i', s)
    s = re.sub('[ÍÌỈĨỊ]', 'I', s)
    s = re.sub('[úùủũụưứừửữự]', 'u', s)
    s = re.sub('[ÚÙỦŨỤƯỨỪỬỮỰ]', 'U', s)
    s = re.sub('[ýỳỷỹỵ]', 'y', s)
    s = re.sub('[ÝỲỶỸỴ]', 'Y', s)
    s = re.sub('đ', 'd', s)
    s = re.sub('Đ', 'D', s)
    return s

# Ham chinh sua thong tin OCR tu chuoi patterns
def adjust_ocr_data(ocr_data, patterns):
    for pattern in patterns:
        # Convert ocr_data, pattern to vietnamese without accents
        text1 = no_accent_vietnamese(ocr_data)
        text2 = no_accent_vietnamese(pattern)
        if re.search(text1, text2, re.IGNORECASE | re.UNICODE):
            return pattern
    
    return ocr_data

# Ham tra ve thong tin tren mo hinh moi
async def ReturnInfoNew(path, text_code, engine):
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
    
    ''' Customize thong tin theo ma van ban '''

    # TODO Custom ThoiGianHieuLuc - Ma van ban 6(GCN Truong Dat Chuan Quoc Gia)
    if (text_code == 'MVB6'):
        key = 'ThoiHanHieuLuc'
        if key in label_dict:
            text = label_dict[key]
            pattern1 = r"c[óo] th[oồờ]i h[aàạậâăặ]n(.*)"
            pattern2 = r"l[àa](.*)"
            combined_pattern = f"{pattern1}|{pattern2}"
            if (text is not None):
                match = re.search(combined_pattern, text, re.IGNORECASE | re.UNICODE)
                if match:
                    content = match.group(1) or match.group(2)
                    label_dict[key] = content.strip()
    
    # TODO: Custom TenHoi - Ma van ban 7(Thanh Lap Hoi)
    if (text_code == 'MVB7'):
        key = 'TenHoi'
        pattern = r"th[aà]nh l[aàạậâăặ]p(.*)"
        text = label_dict['TrichYeu']
        if (text is not None):
            match = re.search(pattern, text, re.IGNORECASE | re.UNICODE)
            if match:
                content = match.group(1)
                label_dict[key] = content.strip()
    
    # TODO: Custom ThoiHanCuaGiayPhep - Ma van ban 8(GPKTTS)
    if (text_code == 'MVB8'):
        key = 'ThoiHanCuaGiayPhep'
        pattern = r"[dđ][eếê]n(.*)"
        text = label_dict[key]
        print(text)
        if (text is not None):
                match = re.search(pattern, text, re.IGNORECASE | re.UNICODE)
                if match:
                    content = match.group(1)
                    label_dict[key] = content.strip()

    # TODO: Custom ChucVu,CoQuanBanHanh,LoaiKetQua - Ma van ban 13(CPKDVT)
    if (text_code == 'MVB13'):
        # Chinh sua thong tin ChucVu, CoQuanBanHanh
        key1 = 'ChucVu'
        key2 = 'CoQuanBanHanh'
        label_dict[key1] = 'TRƯỞNG PHÒNG QUẢN LÝ VẬN TẢI'
        label_dict[key2] = 'SỞ GTVT '+ label_dict[key2]

        # Chinh sua thong tin OCR LoaiKetQua tu chuoi patterns
        key3 = 'LoaiKetQua'
        patterns = [
        "XE CHẠY TUYẾN CỐ ĐỊNH",
        "XE TAXI",
        "XE ĐẦU KÉO",
        "XE HỢP ĐỒNG",
        "XE TẢI",
        "XE CÔNG-TEN-NƠ"
        ]
        label_dict[key3] = adjust_ocr_data(label_dict[key3], patterns)
    
    # TODO: Custom ChucVu,CoQuanBanHanh,LoaiKetQua - Ma van ban 15(GPXTL)
    if (text_code == 'MVB14'):
        key = 'LoaiKetQua'
        label_dict[key] = 'GIẤY PHÉP XE TẬP LÁI'

    # TODO: Custom LoaiKetQua - Ma van ban 15(GPXTL)
    if (text_code == 'MVB15'):
        key1 = 'ThoiGianCoHieuLuc'
        pattern = r"[dđ][eếêé]n(.*)"
        text = label_dict[key1]
        # Custom CoQuanBanHanh
        key2 = 'CoQuanBanHanh'
        label_dict[key2] = 'CHI CỤC CHĂN NUÔI VÀ THÚ Y'
        # Custom LoaiKetQua
        if (text is not None):
            match = re.search(pattern, text, re.IGNORECASE | re.UNICODE)
            if match:
                content = match.group(1)
                label_dict[key1] = content.strip()

    # TODO: Custom LoaiKetQua - Ma van ban 16(CPHXOTKDVT)
    if (text_code == 'MVB16'):
        key = 'LoaiXe'
        patterns = [
        "XE CHẠY TUYẾN CỐ ĐỊNH",
        "XE TAXI",
        "XE ĐẦU KÉO",
        "XE HỢP ĐỒNG",
        "XE TẢI",
        "XE CÔNG-TEN-NƠ"
        ]
        if key in label_dict:
            label_dict[key] = adjust_ocr_data(label_dict[key], patterns)

    # TODO: Custom Hang Xe - Ma van ban 17(GPLXQT)
    if (text_code == 'MVB17'):
        key1 = 'Hang'
        key2 = 'LoaiKetQua'
        label_dict[key1] = None
        label_dict[key2] = 'GIẤY PHÉP LÁI XE QUỐC TẾ'
        
    # TODO: Custom LoaiKetQua,NgayBanHanh,MoTa (CMGPLX)
    if (text_code == 'MVB18'):
        # Key customize
        keys = ['LoaiKetQua', 'NgayBanHanh', 'MoTa', 'SoSeri']
        # Hanlde OCR result
        for key in keys:
            if key in label_dict and label_dict[key] is not None:
                # TODO: Custom LoaiKetQua
                if key == "LoaiKetQua":
                    label_dict[key] = 'GIẤY PHÉP LÁI XE' # Nhan co dinh cua thu tuc
                # TODO: Custom NgayBanHanh
                elif key == "NgayBanHanh":
                    # Tim cac doan so lien tiep (ngay,thang,nam)
                    pattern = re.compile(r"(\d+)")
                    # Tim kiem cac doan so trong chuoi
                    matches = pattern.findall(label_dict[key])
                    if len(matches) >= 3:
                        day, month, year = matches[0],matches[1], matches[2]
                        label_dict[key] = f'{day}/{month}/{year}'
                # TODO: Custom MoTa
                elif key == "MoTa":
                    # Mô tả text cac hạng xe trong GPLX
                    dict_vehicle_of_class = {
                    "den duoi 175cm3":"Xe môtô 2 bánh có dung tích xilanh từ 50 đến dưới 175cm3", #Hang A1
                    "175cm3 tro lên":"Xe môtô 2 bánh có dung tích xilanh từ 175cm3 trở lên và xe hạng A1", #Hang A2
                    "xich lo may":"Xe lam, môtô 3 bánh, xích lô máy", #Hang A3
                    "tai den 1000 kg":"Máy kéo có trọng tải đến 1000kg",# Hang A4
                    "khong chuyen nghiep":"Ôtô chở người đến 9 chỗ ngồi; ô tô tải, máy kéo kéo rơmooc có trọng tải dưới 3500 kg (không chuyên nghiệp)", #Hang B1
                    "3500 kg va xe hang B1":"Ôtô chở người đến 9 chỗ ngồi; ô tô tải, máy kéo kéo rơmooc có trọng tải dưới 3500 kg và xe hạng B1", #Hang B2
                    "3500 kg tro len":"Ôtô tải, máy kéo kéo rơmooc, có trọng tải từ 3500 kg trở lên và xe hạng B1, B2", #Hang C
                    "10 den 30 cho ngoi":"Ôtô chở từ 10 đến 30 chỗ ngồi và xe hang B1, B2, C", #Hang D
                    "tren 30 cho ngoi":"Ôtô chở người trên 30 chỗ ngồi và xe hạng B1, B2, C, D", #Hang E
                    "O to hang C keo romooc":"Ô tô hàng C kéo rơmooc, đầu kéo kéo sơmi rơmooc và xe hạng B1, B2, C, FB2" #Hang FC
                    }
                    result = ''
                    if label_dict[key] is not None:
                        text_ocr = no_accent_vietnamese(label_dict[key])
                        print(text_ocr)
                        for k,v in dict_vehicle_of_class.items():
                            if re.search(k, text_ocr, re.IGNORECASE | re.UNICODE):
                                result += v
                        if(result != ''): label_dict[key] = result
                # Bo qua so seri OCR chua chinh xac
                elif key == "SoSeri":
                    label_dict['SoSeri'] = None     
        key
    
    # TODO: Custom LoaiKetQua,CoQuanBanHanh,DangKyLanDau,DangKyThayDoi (HTX)
    if (text_code == 'MVB19'):
        # Key customize
        keys = ['LoaiKetQua', 'CoQuanBanHanh', 'NgayDangKyLanDau', 'NgayThayDoiCuoiCung']
        # Hanlde OCR result
        for key in keys:
            if key in label_dict:
                # TODO: Custom LoaiKetQua
                if key == "LoaiKetQua":
                    label_dict[key] = 'GIẤY CHỨNG NHẬN' # Nhan co dinh cua thu tuc
                # TODO: Custom CoQuanBanHanh
                elif key == "CoQuanBanHanh":
                    label_dict[key] = 'SỞ KẾ HOẠCH VÀ ĐẦU TƯ TỈNH THANH HOÁ' # Nhan co dinh cua thu tuc
                # TODO: Custom NgayDangKyLanDau va NgayThayDoiCuoiCung
                elif key == "NgayDangKyLanDau" or key == "NgayThayDoiCuoiCung" and label_dict[key] is not None:
                    # Tim cac doan so lien tiep (ngay,thang,nam)
                    pattern = re.compile(r"(\d+)")
                    # Tim kiem cac doan so trong chuoi
                    matches = pattern.findall(label_dict[key])
                    expected_matches = len(matches)
                    if expected_matches >= 3:
                        if expected_matches == 3:
                            day, month, year = matches[:3]
                            label_dict[key] = f'{day}/{month}/{year}'
                        elif expected_matches == 4:
                            times, day, month, year = matches
                            label_dict['SoLanThayDoi'] = int(times)
                            label_dict[key] = f'{day}/{month}/{year}'
    
    # TODO: Customize LoaiGiayTo, NgayBanHanh
    if (text_code == 'MVB20'):
        # Key customize
        keys = ['NgayBanHanh','LoaiGiayTo','CoQuanBanHanh']
        # Hanlde OCR result
        for key in keys:
            if key in label_dict:
                if key == "NgayBanHanh" and label_dict[key] is not None:
                    # Tim cac doan so lien tiep (ngay,thang,nam)
                    pattern = re.compile(r"(\d+)")
                    # Tim kiem cac doan so trong chuoi
                    matches = pattern.findall(label_dict[key])
                    expected_matches = len(matches)
                    if expected_matches >= 3:
                        day, month, year = matches
                        label_dict[key] = f'{day}/{month}/{year}'
                elif key == "LoaiGiayTo":
                    label_dict[key] = f'CHỨNG CHỈ'
                elif key == "CoQuanBanHanh":
                    label_dict[key] = f'SỞ XÂY DỰNG'

    # TODO: Custom TenDoanhNghiep (DKNQLD)
    if (text_code == 'MVB21'):
        key = 'TenDoanhNghiep'
        if label_dict[key] is not None and key in label_dict:
            text_ocr = label_dict[key]
            label_dict[key]= re.sub(r'[-;]', '',text_ocr).strip()
    
    # TODO: Custom NgayHetHan,ChucVu (XNTTHN)
    if (text_code == 'MVB22'):
        # Key customize
        keys = ['NgayHetHan', 'ChucVu', 'LoaiGiayTo']
        # Hanlde OCR result
        for key in keys:
            if key in label_dict and label_dict[key] is not None:
                if key == "NgayHetHan":
                    # Tim cac doan so lien tiep (ngay,thang,nam)
                    pattern = re.compile(r"(\d+)")
                    # Tim kiem cac doan so trong chuoi
                    matches = pattern.findall(label_dict[key])
                    expected_matches = len(matches)
                    if expected_matches >= 1:
                        expiration_date = matches[0]
                        label_dict[key] = expiration_date + " tháng"
                elif key == "ChucVu":
                    patterns = [
                    "PHÓ CHỦ TỊCH",
                    "CHỦ TỊCH"]
                    key_words = 'PHO'
                    text_ocr = no_accent_vietnamese(label_dict[key])
                    if(re.search(key_words, text_ocr, re.IGNORECASE | re.UNICODE)):
                        label_dict[key] = patterns[0]
                    else:
                        label_dict[key] = patterns[1]
                elif key == "LoaiGiayTo":
                    label_dict[key] = f'GIẤY XÁC NHẬN TÌNH TRẠNG HÔN NHÂN'

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

if __name__ == "__main__":
    path = '/home/polaris/ml/DVC/OCR-DVC-ThanhHoa/ocr_files/Screenshot_2.png'
    OCRFile(path)