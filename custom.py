import re
from utils import adjust_ocr_data, no_accent_vietnamese, handle_ner, is_valid_date
''' Customize thong tin theo ma van ban '''
def handle_textcode(label_dict, text_code, ner):
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
                        for k,v in dict_vehicle_of_class.items():
                            if re.search(k, text_ocr, re.IGNORECASE | re.UNICODE):
                                result += v
                        if(result != ''): label_dict[key] = result
                # Bo qua so seri OCR chua chinh xac
                elif key == "SoSeri":
                    label_dict['SoSeri'] = None
    
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
        keys = ['NgayHetHan', 'ChucVu', 'LoaiGiayTo', 'GiayToTuyThan', 'MucDichSuDung']
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
                elif key == "MucDichSuDung":
                    dict_uses = {"ket hon voi":"Làm thủ tục đăng ký kết hôn"}
                    text_ocr = no_accent_vietnamese(label_dict[key])
                    for k,v in dict_uses.items():
                        if re.search(k, text_ocr, re.IGNORECASE | re.UNICODE):
                            label_dict[key] = v
                elif key == "GiayToTuyThan":
                    keywords = 'LoaiGiayTo'
                    raw_text = label_dict[key]
                    if ner is not None:
                        rs = handle_ner(raw_text, ner)
                        label_dict[key] = rs

    # TODO: Custom LoaiGiayTo, TrichYeu, Email, SDT (KBATLD)
    if (text_code == 'MVB23'):
        # Key customize
        keys = ['LoaiGiayTo', 'TrichYeu', 'Email', 'SoDienThoai']
        # Hanlde OCR result
        for key in keys:
            if key in label_dict and label_dict[key] is not None:
                text_ocr = label_dict[key]
                if key == "LoaiGiayTo":
                    label_dict[key] = 'GIẤY XÁC NHẬN KHAI BÁO'
                elif key == "TrichYeu":
                    label_dict[key] = 'SỬ DỤNG MÁY, THIẾT BỊ, VẬT TƯ CÓ YÊU CẦU NGHIÊM NGHẶT VỀ AN TOÀN LAO ĐỘNG'
                elif key == "Email":
                    label_dict[key] = text_ocr.replace(' ','')
                elif key == "SoDienThoai":
                    label_dict[key]= re.sub(r'[-;]', '',text_ocr).strip()

    # TODO: Custom SoGiayChungNhanDKDN, DiaChiTruSoCungCapXangDau, NgayHetHieuLuc (GCNCHDDKBLXD)
    if (text_code == 'MVB24'):
        # Key customize
        keys = ['SoGiayChungNhanDKDN','NgayBanHanh', 'DiaChiTruSoCungCapXangDau', 'NgayHetHieuLuc']
        # Hanlde OCR result
        for key in keys:
            if key in label_dict and label_dict[key] is not None:
                if key == "SoGiayChungNhanDKDN":
                    # Tim cac doan so lien tiep (ngay,thang,nam)
                    pattern = re.compile(r"(\d+)")
                    # Tim kiem cac doan so trong chuoi
                    matches = pattern.findall(label_dict[key])
                    expected_matches = len(matches)
                    print(expected_matches)
                    if expected_matches >= 3:
                        label_dict[key] = f'{matches[0]}'
                elif key == "DiaChiTruSoCungCapXangDau":
                    text_raw = label_dict[key]
                    pattern = r"(.*), Số điện thoại"
                    match = re.search(pattern, text_raw, re.IGNORECASE | re.UNICODE)
                    if match:
                        content = match.group(1)
                        label_dict[key] = content
                elif key == "NgayHetHieuLuc" or key == 'NgayBanHanh':
                    # Tim cac doan so lien tiep (ngay,thang,nam)
                    pattern = re.compile(r"(\d+)")
                    # Tim kiem cac doan so trong chuoi
                    matches = pattern.findall(label_dict[key])
                    expected_matches = len(matches)
                    if expected_matches >= 3:
                        day, month, year = matches
                        label_dict[key] = f'{day}/{month}/{year}'

    # TODO: Custom LoaiGiayTo, NgayBanHanh, CoQuanBanHanh, DienThoai, Email
    if (text_code == "MVB25"):
        # Key customize
        keys = ['LoaiGiayTo','NgayBanHanh', 'CoQuanBanHanh', 'DienThoai', 'Email']
        # Hanlde OCR result
        for key in keys:
            if key in label_dict:
                text_ocr = label_dict[key]
                if key == "LoaiGiayTo" and label_dict[key] is not None:
                    label_dict[key] = 'THÔNG BÁO'
                elif key == "NgayBanHanh" and label_dict[key] is not None:
                    # Tim cac doan so lien tiep (ngay,thang,nam)
                    pattern = re.compile(r"(\d+)")
                    # Tim kiem cac doan so trong chuoi
                    matches = pattern.findall(label_dict[key])
                    expected_matches = len(matches)
                    if expected_matches >= 3:
                        day, month, year = matches
                        label_dict[key] = f'{day}/{month}/{year}'
                elif key == "CoQuanBanHanh":
                    label_dict[key] = "SỞ XÂY DỰNG"
                elif key == "DienThoai":
                    label_dict[key]= re.sub(r'[-;]', '',text_ocr).strip()
                elif key == "Email":
                    label_dict[key] = text_ocr.replace(' ','')

    # TODO: Custom LoaiGiayTo, SoQuyetDinh, ViTriCongViec,TinhTrangCap, NgayBatDauLamViecTu, NgayBatDauLamViecDen, CoQuanBanHanh
    if (text_code == "MVB27"):
        # Key customize
        keys = ['LoaiGiayTo', 'SoQuyetDinh', 'ViTriCongViec', 'TinhTrangCap','NgayBatDauLamViecTu', 'NgayBatDauLamViecDen', 'CoQuanBanHanh', 'NgayBanHanh']
        # Hanlde OCR result
        for key in keys:
            if key in label_dict and label_dict[key] is not None:
                text_ocr = label_dict[key]
                if key == "LoaiGiayTo":
                    label_dict[key] = "GIẤY PHÉP LAO ĐỘNG"
                elif key == "SoQuyetDinh":
                    # Tim cac doan so lien tiep (ngay,thang,nam)
                    pattern = re.compile(r"(\d+)")
                    # Tim kiem cac doan so trong chuoi
                    matches = pattern.findall(label_dict[key])
                    expected_matches = len(matches)
                    if expected_matches >= 1:
                        label_dict[key] = matches[0]
                elif key == "ViTriCongViec":
                    patterns = [
                    "Nhà quản lý",
                    "Giám đốc điều hành",
                    "Chuyên gia",
                    "Lao động kỹ thuật"]
                    label_dict[key] = adjust_ocr_data(label_dict[key], patterns, reverse= False)
                elif key == "TinhTrangCap":
                    patterns = [
                    "Cấp mới",
                    "Cấp lại",
                    "Gia hạn"]
                    label_dict[key] = adjust_ocr_data(label_dict[key], patterns, reverse= False)
                elif key == "NgayBanHanh" or key == "NgayBatDauLamViecTu" or key == "NgayBatDauLamViecDen":
                    # Tim cac doan so lien tiep (ngay,thang,nam)
                    pattern = re.compile(r"(\d+)")
                    # Tim kiem cac doan so trong chuoi
                    matches = pattern.findall(label_dict[key])
                    expected_matches = len(matches)
                    if expected_matches >= 3:
                        day, month, year = matches[0],matches[1], matches[2]
                        if is_valid_date(int(day), int(month), int(year)):
                            label_dict[key] = f'{day}/{month}/{year}'
                elif key == "CoQuanBanHanh":
                    label_dict[key] = "SỞ LAO ĐỘNG THƯƠNG BINH VÀ XÃ HỘI"
    return label_dict
