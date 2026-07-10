import json
import os

def convert(img_size, box):
    dw = 1. / img_size[0]
    dh = 1. / img_size[1]
    x = (box[0] + box[2]) / 2.0 - 1
    y = (box[1] + box[3]) / 2.0 - 1
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = abs(x * dw)
    w = abs(w * dw)
    y = abs(y * dh)
    h = abs(h * dh)
    return x, y, w, h

def decode_json(json_path, txt_output_folder, name2id):
    txt_name = os.path.join(txt_output_folder, os.path.basename(json_path)[:-5] + '.txt')
    os.makedirs(os.path.dirname(txt_name), exist_ok=True)

    with open(txt_name, 'w') as txt_file:
        try:
            data = json.load(open(json_path, 'r', encoding='gb2312'))
        except UnicodeDecodeError:
            data = json.load(open(json_path, 'r', encoding='utf-8'))

        img_w = data['imageWidth']
        img_h = data['imageHeight']

        for i in data['shapes']:
            label_name = i['label']
            if label_name not in name2id:
                print(f"Skipping label: {label_name}")
                continue

            if i['shape_type'] == 'rectangle':
                x1 = int(i['points'][0][0])
                y1 = int(i['points'][0][1])
                x2 = int(i['points'][1][0])
                y2 = int(i['points'][1][1])

                bb = (x1, y1, x2, y2)
                bbox = convert((img_w, img_h), bb)
                txt_file.write(str(name2id[label_name]) + " " + " ".join([str(a) for a in bbox]) + '\n')

if __name__ == "__main__":
    # （需修改）类别名称：索引
    # {"Cavity": 0, "Crack": 1, "Efflorescence": 2, "Erosion": 3 }
    name2id = {"Cavity": 0, "Crack": 1, "Efflorescence": 2, "Erosion": 3 }

    # （需修改）Labelme的JSON文件夹路径
    json_folder_path = r'F:\data\ZSH_DataSets\zshtest\labels_JSON'
    # （需修改）YOLO的TXT文件夹路径（相对）
    txt_output_folder = r'F:\data\ZSH_DataSets\zshtest\labels_TXT'

    for root, _, files in os.walk(json_folder_path):
        for file in files:
            if file.endswith('.json'):
                json_path = os.path.join(root, file)
                print(f'Processing {json_path}')
                decode_json(json_path, txt_output_folder, name2id)
