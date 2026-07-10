import cv2
import numpy as np
import os
import random
import math

input_dir = r"D:\ZSHProject\DATA\ZSHCityWall_YOLO_COCO\images\严重光照"
output_dir_overexpose = r"D:\ZSHProject\DATA\ZSHCityWall_YOLO_COCO\images\val_extreme_overexpose"
output_dir_shadow = r"D:\ZSHProject\DATA\ZSHCityWall_YOLO_COCO\images\val_extreme_shadow_occlusion"

for folder in [output_dir_overexpose, output_dir_shadow]:
    if not os.path.exists(folder):
        os.makedirs(folder)

severity = 0.6


# ================================================

def cv_imread(file_path):
    return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)


def cv_imwrite(file_path, img):
    ext = os.path.splitext(file_path)[1]
    cv2.imencode(ext, img)[1].tofile(file_path)


def apply_adjustable_overexposure(img, severity):
    """可调节的过曝模拟"""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    center = (random.randint(-w // 4, int(w * 1.25)), random.randint(-h // 4, int(h * 1.25)))
    max_dist = math.sqrt(h ** 2 + w ** 2)

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    # 严酷度越高，光晕覆盖范围越大
    mask_spread = 1.5 - (0.5 * severity)
    mask = np.clip(1.0 - (dist_from_center / (max_dist * mask_spread)), 0, 1)
    mask = np.power(mask, random.uniform(1.0, 2.0))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

    # 动态参数：根据 severity 计算提亮和褪色程度
    v_boost_max = 50 + int(130 * severity)  # severity=0.5时约提亮 115
    s_drop_max = 20 + int(80 * severity)  # severity=0.5时约降低 60

    hsv[:, :, 2] = hsv[:, :, 2] + mask * random.uniform(v_boost_max * 0.8, v_boost_max)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] - mask * s_drop_max, 0, 255)

    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def apply_adjustable_shadow_occlusion(img, severity):
    """可调节的重度阴影与遮挡"""
    h, w = img.shape[:2]

    # 整体变暗程度 (severity越高越暗，但不至于全黑)
    global_darkness = 0.8 - (0.4 * severity)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = hsv[:, :, 2] * random.uniform(global_darkness, global_darkness + 0.2)

    mask = np.zeros((h, w), dtype=np.float32)

    # 遮挡块数量 (severity越高，杂草/遮挡物越多)
    shadow_count = int(2 + 6 * severity)
    for _ in range(shadow_count):
        pts = np.array(
            [[random.randint(-w // 2, int(w * 1.5)), random.randint(-h // 2, int(h * 1.5))] for _ in range(5)],
            np.int32)
        cv2.fillPoly(mask, [pts], 1.0)

    blur_size = random.choice([31, 51, 71])
    mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)

    # 局部极度变暗程度 (severity越高，阴影中心越接近死黑)
    # severity=0.5时，最黑的地方衰减为原来的 30%左右，还能隐约看到裂缝
    local_darkness_factor = 0.4 + (0.45 * severity)
    hsv[:, :, 2] = hsv[:, :, 2] * (1.0 - mask * local_darkness_factor)

    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# 执行生成
count = 0
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        filepath = os.path.join(input_dir, filename)
        img = cv_imread(filepath)
        if img is None: continue

        img_overexpose = apply_adjustable_overexposure(img, severity)
        cv_imwrite(os.path.join(output_dir_overexpose, filename), img_overexpose)

        img_shadow = apply_adjustable_shadow_occlusion(img, severity)
        cv_imwrite(os.path.join(output_dir_shadow, filename), img_shadow)
        count += 1

print(f"基于严酷度系数 Severity={severity} 生成了 {count} 张测试图。")