import os
import torch
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from torch.utils.data import DataLoader
from picosam2_model_distillation import PicoSAM2Dataset

IMAGE_SIZE = 96
NUM_SAMPLES = 1000
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(BASE_DIR, "..", "checkpoints")
COCO_IMG_ROOT = os.path.join(BASE_DIR, "..", "dataset", "val2017")
COCO_ANN_FILE = os.path.join(BASE_DIR, "..", "dataset", "annotations", "instances_val2017.json")
LVIS_IMG_ROOT = os.path.join(BASE_DIR, "..", "dataset", "val2017_lvis")
LVIS_ANN_FILE = os.path.join(BASE_DIR, "..", "dataset", "annotations", "lvis_v1_val.json")

# Stays the same as in benchmark.py
def unnormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    return (tensor * std + mean).clamp(0,1)

def calc_miou(preds, targets):
    preds = (tf.nn.sigmoid(preds) > 0.5).numpy()
    targets = (targets > 0.5).numpy()
    return np.mean([
        (np.logical_and(p,t).sum() / np.logical_or(p,t).sum())
        if np.logical_or(p,t).sum() else 1.0
        for p,t in zip(preds, targets)
    ])

def calc_map_iou_range(preds, targets, iou_thresholds=np.arange(0.5, 1.0, 0.05)):
    preds = tf.nn.sigmoid(preds).numpy()
    targets = targets.numpy()
    aps = []

    for iou_thresh in iou_thresholds:
        ap_per_thresh = []
        for p, t in zip(preds, targets):
            if t.sum() == 0:
                continue
            p_bin = p > 0.5
            t_bin = t > 0.5
            iou = np.logical_and(p_bin, t_bin).sum() / (np.logical_or(p_bin, t_bin).sum() + 1e-8)
            ap_per_thresh.append(1.0 if iou >= iou_thresh else 0.0)
        if ap_per_thresh:
            aps.append(np.mean(ap_per_thresh))

    return float(np.mean(aps)) if aps else 0.0


# TensorFlow SavedModel inference wrapper
class TFModelWrapper:
    def __init__(self, model_dir):
        self.model = tf.saved_model.load(model_dir)
        self.infer = self.model.signatures["serving_default"]

    def __call__(self, x):
        # x: numpy or tf.Tensor
        if not isinstance(x, tf.Tensor):
            x = tf.convert_to_tensor(x, dtype=tf.float32)
        outputs = self.infer(x)
        return list(outputs.values())[0]  # assume single output

# TFLite inference wrapper
class TFLiteModelWrapper:
    def __init__(self, tflite_path):
        self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(self, x):
        if not isinstance(x, np.ndarray):
            x = x.numpy()
        self.interpreter.set_tensor(self.input_details[0]["index"], x)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details[0]["index"])

# Adapt dataset to TensorFlow
# Dataset yields PyTorch tensors in NCHW
def torch_to_tf_input(x):
    x = x.cpu().numpy() # torch.Tensor NCHW = [1,3,H,W]
    x = np.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
    x.astype(np.float32)
    
    # Apply PyTorch normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    
    return x



# 1) TensorFlow evaluation loop
def evaluate_picosam_tf(model, loader, name):
    print(f"\nEvaluating: {name}")
    preds, gts, mious = [], [], []

    for i, (x, y, _, _) in enumerate(tqdm(loader)):
        if i >= NUM_SAMPLES:
            break

        x_tf = torch_to_tf_input(x)
        y_tf = tf.convert_to_tensor(y.numpy(), dtype=tf.float32)

        pred = model(x_tf)  # TF or TFLite
        pred = tf.image.resize(pred, y_tf.shape[-2:], method="bilinear")

        preds.append(pred)
        gts.append(y_tf)
        mious.append(calc_miou(pred, y_tf))

    preds = tf.concat(preds, axis=0)
    gts = tf.concat(gts, axis=0)

    print(
        f"{name} -> "
        f"mIoU: {np.mean(mious):.4f}, "
        f"mAP@[0.5:0.95]: {calc_map_iou_range(preds, gts):.4f}"
    )
    

# 2) Model size measurement
# 2.1) SavedModel size
def get_savedmodel_size(model_dir):
    total = 0
    for root, _, files in os.walk(model_dir):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total / (1024 * 1024)  # MB

# 2.2) TFLite size
def get_tflite_size(tflite_path):
    return os.path.getsize(tflite_path) / (1024 * 1024)



if __name__ == "__main__":
    coco_data = PicoSAM2Dataset(COCO_IMG_ROOT, COCO_ANN_FILE, image_size=IMAGE_SIZE)
    coco_loader = DataLoader(coco_data, batch_size=1, shuffle=False)

    # ===== Load model =====
    tf_model = TFModelWrapper("model_tf") # Pass the folder
    # OR
    tflite_model = TFLiteModelWrapper("PicoSAM2_student.tflite") # Pass the file

    # ===== Evaluate =====
    #evaluate_picosam_tf(tf_model, coco_loader, "PicoSAM2 TensorFlow (COCO)")
    evaluate_picosam_tf(tflite_model, coco_loader, "PicoSAM2 TensorFlowLite (COCO)")

    # ===== Model size =====
    #print(f"SavedModel size: {get_savedmodel_size('model_tf'):.2f} MB")
    #print(f"TFLite size: {get_tflite_size('models/PicoSAM2_student.tflite'):.2f} MB")
