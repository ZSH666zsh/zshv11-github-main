from ultralytics import YOLO
import warnings

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    model = YOLO(r'D:\ZSHProject\weights\best.pt')

    model.val(source=r'D:\ZSHProject\DATA\0006.jpg',
              imgsz=640,
              batch=4,
              # device='0',
              )
