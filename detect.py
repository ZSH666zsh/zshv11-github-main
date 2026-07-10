from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    model = YOLO(r'D:\ZSHProject\weights\best.pt')

    model.predict(source=r'D:\ZSHProject\DATA\0006.jpg',

                  # save=True,
                  # show=True,
                  conf=0.5,
                  )

# 只显示特定类别：python detect.py --weights best.pt --source video.mp4 --classes 0  # 假设人的类别ID是0
# 实时摄像头检测：python detect.py --weights best.pt --source 0 --view-img # 0表示默认摄像头，view-img表示实时显示摄像头画面
# --save-txt	保存检测框的坐标到 .txt 文件
# --save-crop	裁剪检测到的目标
