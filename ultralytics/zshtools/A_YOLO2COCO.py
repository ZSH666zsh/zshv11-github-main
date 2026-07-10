import json
import os
import shutil

import cv2

# COCO数据集instances_train/val/test.json中有5个大字段，info和licenses不重要。categories、images、annotations为必填字段。
info = {
    "year": 2025,
    "version": '1.0',
    "date_created": "2025-08-31"
}

licenses = {
    "id": 1,
    "name": "null",
    "url": "null",
}

# （需修改）yolo原来的类别0、1、2
categories = [
    {
        "id": 1,
        "name": 'Crack',
        "supercategory": 'lines',
    },
    # {
    #     "id": 2,
    #     "name": 'Crack',
    #     "supercategory": 'lines',
    # },
    # {
    #     "id": 3,
    #     "name": 'Efflorescence',
    #     "supercategory": 'lines',
    # },
    # {
    #     "id": 4,
    #     "name": 'Erosion',
    #     "supercategory": 'lines',
    # }

]

# 初始化train、test、valid 数据字典。info、licenses、categories在train和test里一致；
train_data = {'info': info, 'licenses': licenses, 'categories': categories, 'images': [], 'annotations': []}
test_data = {'info': info, 'licenses': licenses, 'categories': categories, 'images': [], 'annotations': []}
valid_data = {'info': info, 'licenses': licenses, 'categories': categories, 'images': [], 'annotations': []}


def yolo_covert_coco_format(image_path, label_path):
    images = []
    annotations = []
    for index, img_file in enumerate(os.listdir(image_path)):
        if img_file.endswith('.jpg'):
            image_info = {}
            img = cv2.imread(os.path.join(image_path, img_file))
            height, width, channel = img.shape
            image_info['id'] = index
            image_info['file_name'] = img_file
            image_info['width'], image_info['height'] = width, height
        else:
            continue
        if image_info != {}:
            images.append(image_info)
        # 处理label信息-------
        label_file = os.path.join(label_path, img_file.replace('.jpg', '.txt'))
        with open(label_file, 'r') as f:
            for idx, line in enumerate(f.readlines()):
                info_annotation = {}
                class_num, xs, ys, ws, hs = line.strip().split(' ')
                class_id, xc, yc, w, h = int(class_num), float(xs), float(ys), float(ws), float(hs)
                xmin = (xc - w / 2) * width
                ymin = (yc - h / 2) * height
                xmax = (xc + w / 2) * width
                ymax = (yc + h / 2) * height
                bbox_w = int(width * w)
                bbox_h = int(height * h)
                img_copy = img[int(ymin):int(ymax), int(xmin):int(xmax)].copy()

                info_annotation["category_id"] = class_id + 1  # 类别的id，代码给classId+1，训练后结果也是+1的，比如yolo中0是人。那coco后1才是人。
                info_annotation['bbox'] = [xmin, ymin, bbox_w, bbox_h]  ## bbox的坐标
                info_annotation['area'] = bbox_h * bbox_w
                info_annotation['image_id'] = index
                info_annotation['id'] = index * 100 + idx
                # cv2.imwrite(f"./temp/{info_annotation['id']}.jpg", img_copy)
                info_annotation['segmentation'] = [[xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]]  # 四个点的坐标
                info_annotation['iscrowd'] = 0
                annotations.append(info_annotation)
    return images, annotations


def gen_json_file(yolov8_data_path, coco_format_path, key):
    # JSON文件路径
    json_path = os.path.join(coco_format_path, f'annotations/instances_{key}.json')
    dst_path = os.path.join(coco_format_path, f'{key}')
    if not os.path.exists(os.path.dirname(json_path)):
        os.makedirs(os.path.dirname(json_path), exist_ok=True)

    # YOLO数据路径
    data_path = os.path.join(yolov8_data_path, f'images/{key}')
    label_path = os.path.join(yolov8_data_path, f'labels/{key}')

    # 转换数据，返回images图像信息列表、annotations标注信息列表
    images, anns = yolo_covert_coco_format(data_path, label_path)

    # 根据数据集类型更新对应的数据字典
    if key == 'train':
        train_data['images'] = images
        train_data['annotations'] = anns
        with open(json_path, 'w') as f:
            json.dump(train_data, f, indent=2)
    elif key == 'test':
        test_data['images'] = images
        test_data['annotations'] = anns
        with open(json_path, 'w') as f:
            json.dump(test_data, f, indent=2)
    elif key == 'val':
        valid_data['images'] = images
        valid_data['annotations'] = anns
        with open(json_path, 'w') as f:
            json.dump(valid_data, f, indent=2)
    else:
        print(f'key is {key}')
    print(f'generate {key} json success!')
    return

def src_add_image(src):
    # 只移动images文件夹下面的图片
    src = os.path.join(src, 'images')
    return src

def copy_directory_contents(src, dest):
    # 如果目标目录不存在，则创建它
    if not os.path.exists(dest):
        os.makedirs(dest)
    # 遍历源目录中的所有文件和目录
    for item in os.listdir(src):
        src_path = os.path.join(src, item)
        dest_path = os.path.join(dest, item)
        # 如果是目录，递归调用这个函数
        if os.path.isdir(src_path):
            copy_directory_contents(src_path, dest_path)
        # 如果是文件，则复制文件
        else:
            shutil.copy2(src_path, dest_path)
        print("图片复制成功!")


if __name__ == '__main__':
    yolov8_data_path = r'D:\ZSHProject\DATA\ZSHCMHB_YOLO'  # YOLO数据集路径（不具体到images和labels）
    coco_format_path = r'D:\ZSHProject\DATA\ZSHCMHB_COCO'  # COCO数据集输出路径
    gen_json_file(yolov8_data_path, coco_format_path, key='train')  # key为数据集类型 (instances_train/val/test.json)
    gen_json_file(yolov8_data_path, coco_format_path, key='val')
    gen_json_file(yolov8_data_path, coco_format_path, key='test')
    # 复制图片到COCO数据集
    src = src_add_image(yolov8_data_path)
    copy_directory_contents(src, coco_format_path)
    print("COCO数据集已建立!")
