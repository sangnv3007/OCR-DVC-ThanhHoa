import re
import cv2

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

# Ham lay ket qua detect dua vao file *.txt
def get_data_model(folder_name):
    # Ly lich tu phap
    net_MVB3, classes_MVB3 = load_model_old(f'./MVB3/model/yolov4-custom.weights',
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
def adjust_ocr_data(ocr_data, patterns, reverse = True):
    for pattern in patterns:
        # Convert ocr_data, pattern to vietnamese without accents
        text1 = no_accent_vietnamese(ocr_data)
        text2 = no_accent_vietnamese(pattern)
        if reverse:
            if re.search(text1, text2, re.IGNORECASE | re.UNICODE):
                return pattern
        else:
            if re.search(text2, text1, re.IGNORECASE | re.UNICODE):
                return pattern
    return ocr_data

# Ham xu ly chuoi van ban tren NER
def handle_ner(raw_text, ner):
    doc = ner(raw_text)
    # Tạo từ điển từ kết quả của NER
    result_dict = {}
    current_label = None
    current_text = ''
    for ent in doc.ents:
        word, label = ent.text, ent.label_
        if label.startswith('B-'):  # Nếu là nhãn bắt đầu (Begin)
            if current_label:  # Nếu đã có nhãn trước đó, lưu lại kết quả
                result_dict[current_label] = current_text.strip()
            current_label = label[2:]
            current_text = word
        elif label.startswith('I-'):  # Nếu là nhãn tiếp tục (Inside)
            if current_label == label[2:]:  # Nếu nhãn tiếp tục trùng với nhãn hiện tại
                current_text += ' ' + word
            else:  # Nếu nhãn tiếp tục không trùng, lưu lại kết quả và bắt đầu nhãn mới
                if current_label is not None:
                    result_dict[current_label] = current_text.strip()
                    current_label = label[2:]
                    current_text = word

    # Lưu kết quả cho nhãn cuối cùng
    if current_label:
        result_dict[current_label] = current_text.strip()

    return result_dict

# Ham check dinh dang ngay, thang, nam co hop le
def is_valid_date(day, month, year):
    if len(str(year)) != 4:
        return False

    # Kiểm tra tính hợp lệ của ngày, tháng, năm
    if month < 1 or month > 12:
        return False
    if day < 1 or day > 31:
        return False
    if month in [4, 6, 9, 11] and day > 30:
        return False
    if month == 2:
        if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
            if day > 29:
                return False
        elif day > 28:
            return False

    return True