import os
import xml.etree.ElementTree as ET
import glob

# 1. 类别映射字典
classes_mapping = {
    'D00': 0,  # Longitudinal Crack
    'D10': 1,  # Transverse Crack
    'D20': 2,  # Aligator Crack
    'D40': 3  # Pothole
}


def convert_box(size, box):
    """将 VOC 的 [xmin, ymin, xmax, ymax] 转换为 YOLO 的 [x_center, y_center, w, h]"""
    dw = 1. / size[0]
    dh = 1. / size[1]
    x_center = (box[0] + box[1]) / 2.0
    y_center = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]

    x_center = x_center * dw
    w = w * dw
    y_center = y_center * dh
    h = h * dh
    return (x_center, y_center, w, h)


def convert_annotation(xml_file, txt_dir):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    yolo_labels = []

    for obj in root.iter('object'):
        # 很多 VOC 格式可能没有 difficult 标签，这里做个安全获取
        difficult_elem = obj.find('difficult')
        difficult = difficult_elem.text if difficult_elem is not None else '0'
        cls_name = obj.find('name').text

        # 过滤掉不在字典中的类别 或 极难样本
        if cls_name not in classes_mapping or int(difficult) == 1:
            continue

        cls_id = classes_mapping[cls_name]
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
             float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert_box((w, h), b)

        yolo_labels.append(f"{cls_id} {' '.join([str(a) for a in bb])}")

    # 【重点修改】无论有没有提取到目标，都生成 txt 文件！
    # 没有目标的将生成 0 字节的空文件，作为 YOLO 训练的背景负样本
    txt_filename = os.path.basename(xml_file).replace('.xml', '.txt')
    txt_filepath = os.path.join(txt_dir, txt_filename)

    with open(txt_filepath, 'w') as f:
        if yolo_labels:
            f.write('\n'.join(yolo_labels))


def main():
    # ---------- 请修改以下路径 ----------
    xml_dir = 'D:\ZSHProject\DATA\YYB_YOLO\RDD2022-China\labels_xml'  # 你的 XML 文件夹路径
    txt_dir = 'D:\ZSHProject\DATA\YYB_YOLO\RDD2022-China\labels_txt'  # 输出的 YOLO TXT 文件夹路径
    # ------------------------------------

    os.makedirs(txt_dir, exist_ok=True)
    xml_files = glob.glob(os.path.join(xml_dir, '*.xml'))

    print(f"找到 {len(xml_files)} 个 XML 文件，开始转换...")
    for xml_file in xml_files:
        convert_annotation(xml_file, txt_dir)
    print("XML 转换 YOLO TXT 完成！背景负样本已保留。")


if __name__ == '__main__':
    main()