from ultralytics import YOLO
import warnings

if __name__ == '__main__':
    model = YOLO('modelYaml')

    model.train(data=r'datasetsYaml',
                epochs=300,
                batch=16,
                project='runs/zsh',
                imgsz=640,
                device='0',
                )
