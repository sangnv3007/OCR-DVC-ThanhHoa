from process import ReturnInfoTest, load_model, ReturnInfDataNER, get_data_model
import json
import os

''' Load mo hinh '''

# Ly lich tu phap
net_MVB3, classes_MVB3 = load_model(f'./MVB3/model/yolov4-custom.weights',
                                  f'./MVB3/model/yolov4-custom.cfg',
                                  f'./MVB3/model/obj.names')
if __name__ == "__main__":
    file_path = './MVB3/AnhGoc/Untitled.FR12 - 0047.pdf'
    rs = ReturnInfoTest(file_path, 'MVB3', net_MVB3, classes_MVB3)
    # folder_path = './MVB3/AnhGoc'
    # get_data_model(folder_path)