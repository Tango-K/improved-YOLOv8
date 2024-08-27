import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8s-ASF-P2.yaml')
    # model.load('./yolov8s.pt') # loading pretrain weights
    model.train(data='dataset/data-sar.yaml',
                cache=False,
                imgsz=640,
                epochs=100,
                batch=16,
                close_mosaic=10,
                workers=8,
                device='0',
                # optimizer='SGD', # using SGD
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                # project='runs/train',
                name='yolov8s-ASF-P2-Wise-inner-MPDIoU',
                )