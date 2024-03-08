# Ham load thu vien vietOCR
def vietocr_load():
    config = Cfg.load_config_from_name('vgg_transformer')
    config['weights'] = './models/transformerocr.pth'
    config['cnn']['pretrained'] = False
    config['device'] = 'cpu'
    config['predictor']['beamsearch'] = False
    detector = Predictor(config)
    return detector

from paddleocr import PaddleOCR
import cv2
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image
import os, glob
from pathlib import Path
import time
from export import get_workbook
import numpy as np

'''
Export file train PaddleOCR
'''
def ExportPaddleTrainDet():
    with open("Label.txt", 'a', encoding='utf-8') as file:
        for idx, file_path in enumerate(file_list, start=1):
            result = ocr.ocr(file_path, rec=False, cls=False)
            # Sap xep ket qua tu trn duong duoi trai qua phai

            image = cv2.imread(file_path)
            start_time = time.time()
            ocr_text = []
            text_ocr = ''
            for line in result:    
                image_crop = image[int(line[0][1]): int(line[2][1]), int(line[0][0]):int(line[2][0])]
                transcription_value = detector.predict(Image.fromarray(image_crop))
                format_value= transcription_value.replace('"', "")
                ocr_text.append(f'{{"transcription": "{format_value}", "points": {line}, "difficult": false}}')
            
            end_time = time.time()
            #tính thời gian chạy của thuật toán Python
            elapsed_time = end_time - start_time
            print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

            lst_ocr_text = ', '.join(ocr_text)
            name_file = f'data_train/{Path(file_path).name}'
            new_content = f'{name_file}\t[{lst_ocr_text}]\n'
            file.write(new_content)
            
            print(f'Done file {idx}: {name_file}')
            print(f'================================================================')

def ExportPaddleSingleDet(file):
    print(f'Processing file: {file}')
    file_name = Path(file).stem

    text_ocr = ''
    result = ocr.ocr(file, rec=False, cls=False)
    # Sap xep ket qua cac box theo thu tu trai qua phai, tu tren xuong duoi
    result = sorted_boxes(result)

    image = cv2.imread(file)
    for line in result:    
        image_crop = image[int(line[0][1]): int(line[2][1]), int(line[0][0]):int(line[2][0])]
        transcription_value = detector.predict(Image.fromarray(image_crop))
        format_value= transcription_value.replace('"', "")
        text_ocr += f'{format_value}\t{line}\n'
    # Đường dẫn tới file .txt bạn muốn lưu dữ liệu vào
    file_path = f"{file}_label_test.txt"
    # Mở file .txt trong chế độ ghi ('w' là chế độ ghi, 'a' để thêm vào cuối file)
    with open(file_path, 'w') as file:
        if (text_ocr != ''):
            file.write(text_ocr)  # Ghi từng dòng dữ liệu vào file,
    print(f'=> Done file: {file}')
    print('----------------------------------------------------------------')
def ExportNERTrain(folder_name):
    for file in glob.glob(os.path.join(folder_name, '*.jpg')):
        print(f'Processing file: {file}')
        file_name = Path(file).stem
        
        # Thong tin trich xuat 
        text_ocr = ''
        result = ocr.ocr(file, rec=False, cls=False)

        # Sap xep ket qua cac box theo thu tu tren xuong duoi, tu trai qua phai, 
        result = sorted_boxes(result)
        image = cv2.imread(file)
        for line in result:    
            image_crop = image[int(line[0][1]): int(line[2][1]), int(line[0][0]):int(line[2][0])]
            transcription_value = detector.predict(Image.fromarray(image_crop))
            format_value= transcription_value.replace('"', "")
            text_ocr += f'{format_value}\n'
        # Đường dẫn tới file .txt bạn muốn lưu dữ liệu vào
        file_path = f"{file}_label.txt"
        # Mở file .txt trong chế độ ghi ('w' là chế độ ghi, 'a' để thêm vào cuối file)
        with open(file_path, 'w') as file:
            if (text_ocr != ''):
                file.write(text_ocr)  # Ghi từng dòng dữ liệu vào file,
        print(f'=> Done file: {file}')
        print('----------------------------------------------------------------')
    print('Completed !!!')

def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    """
    num_boxes = len(dt_boxes)
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                (_boxes[i + 1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes

if __name__ == "__main__":
    detector = vietocr_load()
    ocr = PaddleOCR(lang="en", use_gpu=False)

    # folder_path = f'./MVB1/image/'
    # ExportNERTrain(folder_path)

    #get_workbook(folder_path)
    # file_path = './MVB3/AnhGoc/Untitled.FR12 - 0047.pdf'
    # rs = ReturnInfoTest(file_path, 'MVB3', net_MVB3, classes_MVB3)

    # Export single file
    file = f'./pdf2img/files/Untitled.FR12 - 0001.jpg'
    ExportPaddleSingleDet(file)