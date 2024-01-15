from openpyxl import Workbook
import os

def get_workbook(path_folder):
    # Tạo danh sách các file .txt có trong thư mục
    txt_files = [file for file in os.listdir(path_folder) if file.endswith('.txt')]

    data = []

    # Đọc từng file .txt và tạo dữ liệu
    for file_index, file_name in enumerate(txt_files):
        with open(os.path.join(path_folder, file_name), 'r', encoding='utf-8') as file:
            lines = file.read().splitlines()  # Đọc từng dòng trong file
            # Sử dụng list comprehension để tạo dữ liệu
            words = [word for line in lines for word in line.strip().split()]
            data.extend([{'id': file_index + 1, 'text': w, 'tag': ''} for w in words])

    # Tạo file Excel và ghi dữ liệu
    wb = Workbook()
    sheet = wb.active

    # Đặt tên cho các cột
    sheet['A1'] = 'id'
    sheet['B1'] = 'text'
    sheet['C1'] = 'tag'

    # Ghi dữ liệu vào các cột tương ứng
    for row_index, row_data in enumerate(data, start=2):
        sheet[f'A{row_index}'] = row_data['id']
        sheet[f'B{row_index}'] = row_data['text']
        sheet[f'C{row_index}'] = row_data['tag']

    # Lưu file Excel
    file_path = 'data.xlsx'  # Đặt tên và đường dẫn cho file Excel
    wb.save(file_path)

    print(f"File Excel đã được tạo: '{file_path}'")