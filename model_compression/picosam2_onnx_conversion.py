import os
import torch
import itertools
from torch.utils.data import DataLoader

from picosam2_model_distillation import PicoSAM2, PicoSAM2Dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
model_ckpt = os.path.join(BASE_DIR, "..", "checkpoints", "PicoSAM2_student_epoch1.pt")
onnx_output_path = os.path.join(BASE_DIR, "..", "checkpoints", "PicoSAM2_student.onnx")
val_images_dir = os.path.join(BASE_DIR, "..", "dataset", "val2017")
annotations_file = os.path.join(BASE_DIR, "..", "dataset", "annotations", "instances_val2017.json")

# Load model
model = PicoSAM2(in_channels=3)
model.load_state_dict(torch.load(model_ckpt, map_location="cpu"))
model.eval()  # Important for ONNX export

# Create dummy input for ONNX export
dummy_input = torch.randn(1, 3, 96, 96)  # batch_size=1, channels=3, height=96, width=96

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    onnx_output_path,
    export_params=True,        # store the trained parameter weights
    opset_version=17,          # ONNX opset version
    do_constant_folding=True,  # optimize constants
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    }
)

print(f"Exported model to ONNX at: {onnx_output_path}")