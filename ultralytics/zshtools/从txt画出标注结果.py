import os
import cv2
import numpy as np

# --- 配置区域 ---
# True: 显示名称 (如 Crack)
# False: 显示序号 (如 1)
USE_NAME_LABEL = True

visdrone_classes = [
    'D00',        # 0
    'D10',         # 1
    'D20', # 2
    'D40'        # 3
]
# visdrone_classes = [
#     'Cavity',        # 0
#     'Crack',         # 1
#     'Efflorescence', # 2
#     'Erosion'        # 3
# ]

# 标签颜色 (BGR 格式)
# "0" (Cavity): 红色
# "1" (Crack): 绿色
# "2" (Efflorescence): 黄色
# "3" (Erosion): 蓝色
class_colors = {
    "0": (0, 0, 255),    # Red
    "1": (0, 255, 0),    # Green
    "2": (0, 255, 255),  # Yellow
    "3": (255, 0, 0)     # Blue
}

def show_label_from_txt(img_path, txt_path, save_dir, save_filename, use_names=True):
    # 读取图片 (支持中文路径)
    src_img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)

    if src_img is None:
        print(f"Error: Could not load image {img_path}")
        return

    h, w = src_img.shape[:2]

    # 检查标签文件是否存在
    if not os.path.exists(txt_path):
        print(f"Error: Label file {txt_path} not found")
        return

    with open(txt_path, "r", encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        data = line.strip().split(' ')
        if len(data) < 5:
            continue

        # 1. 解析YOLO格式数据
        class_id_str = data[0]
        x_center = float(data[1]) * w
        y_center = float(data[2]) * h
        box_width = float(data[3]) * w
        box_height = float(data[4]) * h

        # 2. 计算边界框坐标
        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)
        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)

        p1 = (x1, y1)
        p2 = (x2, y2)

        # 3. 设置样式
        thickness = 2
        font_scale = 0.8
        font_thickness = 2
        font_face = cv2.FONT_HERSHEY_SIMPLEX

        # 获取颜色 (根据ID获取)
        color = class_colors.get(class_id_str, (255, 255, 255))

        # --- 核心修改：决定显示文本是 名称 还是 ID ---
        if use_names:
            try:
                class_id_int = int(class_id_str)
                # 确保ID在列表范围内
                if 0 <= class_id_int < len(visdrone_classes):
                    text = visdrone_classes[class_id_int]
                else:
                    text = class_id_str # 如果ID越界，回退显示数字
            except ValueError:
                text = class_id_str     # 如果转换失败，回退显示原始字符
        else:
            text = class_id_str         # 强制显示数字ID
        # ----------------------------------------

        # 4. 绘制矩形框
        cv2.rectangle(src_img, p1, p2, color, thickness)

        # 5. 绘制文字标签
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font_face, font_scale, font_thickness
        )

        # 防止文字跑出图片上边缘
        y_text = y1 - 5 if y1 - text_height - 5 > 0 else y1 + text_height + 5

        # 绘制文字背景条 (填充)
        cv2.rectangle(
            src_img,
            (x1, y_text - text_height - baseline),
            (x1 + text_width, y_text + baseline),
            color,
            -1
        )

        # 绘制白色文字
        cv2.putText(
            src_img, text, (x1, y_text),
            font_face, font_scale, (255, 255, 255), font_thickness
        )

    save_path = os.path.join(save_dir, save_filename)

    # 保存图片 (支持中文路径)
    is_success, im_buf_arr = cv2.imencode(os.path.splitext(save_filename)[1], src_img)
    if is_success:
        im_buf_arr.tofile(save_path)
        print(f"模式: {'[显示名称]' if use_names else '[显示序号]'} -> 结果已保存至: {save_path}")
    else:
        print(f"Error: Could not save to {save_path}")
    return

# --- 这里是你的路径设置 ---
img_path = r'D:\ZSHProject\DATA\YYB_YOLO\RDD2022-China\RDD2022-China-Final\images\test\China_Drone_000004.jpg'
txt_path = r"D:\ZSHProject\DATA\YYB_YOLO\RDD2022-China\RDD2022-China-Final\labels\test\China_Drone_000004.txt"
save_folder = r"D:\ZSHProject\ultralytics-main\zsh_tools\EI\results"
output_filename = "1478NEW1可视化.jpg"

os.makedirs(save_folder, exist_ok=True)

# 调用函数，use_names参数由顶部的配置控制
show_label_from_txt(img_path, txt_path, save_folder, output_filename, use_names=USE_NAME_LABEL)