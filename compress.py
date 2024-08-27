import warnings
warnings.filterwarnings('ignore')
import argparse, yaml, copy
from ultralytics.models.yolo.detect.compress import DetectionCompressor, DetectionFinetune
from ultralytics.models.yolo.segment.compress import SegmentationCompressor, SegmentationFinetune
from ultralytics.models.yolo.pose.compress import PoseCompressor, PoseFinetune
from ultralytics.models.yolo.obb.compress import OBBCompressor, OBBFinetune

def compress(param_dict):
    with open(param_dict['sl_hyp'], errors='ignore') as f:
        sl_hyp = yaml.safe_load(f)
    param_dict.update(sl_hyp)
    param_dict['name'] = f'{param_dict["name"]}-prune'
    param_dict['patience'] = 0
    compressor = DetectionCompressor(overrides=param_dict)
    # compressor = SegmentationCompressor(overrides=param_dict)
    # compressor = PoseCompressor(overrides=param_dict)
    # compressor = OBBCompressor(overrides=param_dict)
    prune_model_path = compressor.compress()
    return prune_model_path

def finetune(param_dict, prune_model_path):
    param_dict['model'] = prune_model_path
    param_dict['name'] = f'{param_dict["name"]}-finetune'
    trainer = DetectionFinetune(overrides=param_dict)
    # trainer = SegmentationFinetune(overrides=param_dict)
    # trainer = PoseFinetune(overrides=param_dict)
    # trainer = OBBFinetune(overrides=param_dict)
    trainer.train()

if __name__ == '__main__':
    param_dict = {
        # origin
        'model': 'runs/detect/yolov8s-ASF-P2-Wise-inner-MPDIoU-1.1/weights/best.pt',
        'data':r'C:\YOLO\ultralytics_20240505\dataset\data.yaml',
        'imgsz': 640,
        'epochs': 100,
        'batch': 8,
        'workers': 8,
        'cache': True,
        'optimizer': 'SGD',
        'device': '0',
        'close_mosaic': 0,
        'project':'runs/prune',
        'name':'yolov8s-ASF-P2-l1-1.5',
        
        # prune
        'prune_method':'l1',
        'global_pruning': True,
        'speed_up': 1.5,
        'reg': 0.0005,
        'sl_epochs': 500,
        'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
        'sl_model': None,
    }
    
    prune_model_path = compress(copy.deepcopy(param_dict))
    finetune(copy.deepcopy(param_dict), prune_model_path)