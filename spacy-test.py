import spacy

NER = spacy.load(f'./models/MVB22/ner/model-best')

raw_text="Xác nhận thông tin về cư trú, Công an xã Trung Đông cấp ngày 14/06/2023"

def handle_ner(raw_text):
    doc = NER(raw_text)
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
    # # In kết quả
    # for label, text in result_dict.items():
    #     print(f"{label}: {text}")

rs = handle_ner(raw_text)
print(rs)