# Ham load thu vien vietOCR
def vietocr_load():
    config = Cfg.load_config_from_name('vgg_transformer')
    config['weights'] = './models/transformerocr.pth'
    config['cnn']['pretrained'] = False
    config['device'] = 'cpu'
    config['predictor']['beamsearch'] = False
    detector = Predictor(config)
    return detector

if __name__ == "__main__":
    detector = vietocr_load()
    
    file = f'./pdf2img/files/Untitled.FR12 - 0001.jpg'
    ExportPaddleSingleDet(file)