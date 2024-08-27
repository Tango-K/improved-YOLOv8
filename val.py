import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/detect/yolov8s/weights/best.pt')
    model.val(data='dataset/data.yaml',
              split='val',
              imgsz=640,
              batch=1,
              # rect=False,
              save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='gaijinqian1',
              )
    # runs/prune/yolov8s-ASF-P2-Wise-inner-MPDIoU-lamp-exp1.5-finetune/weights/best.pt
    # runs/detect/yolov8s/weights/best.pt