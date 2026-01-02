import tensorflow as tf
import itertools
from torch.utils.data import DataLoader
from picosam2_model_distillation import PicoSAM2Dataset

# Goal: Apply post training quantization (PTQ)
# Input: fp32 tensorflow model
# Output: int8 model (.tflite)

# 1) Load FP32 TFLite model
tflite_fp32_model_path = "./PicoSAM2_student.tflite"

# 2) Prepare representative dataset generator
# This should yield numpy arrays of shape [batch_size, H, W, C], dtype=float32
def representative_dataset_gen():
    dataset = PicoSAM2Dataset(
        image_root="../dataset/val2017",
        annotation_file="../dataset/annotations/instances_val2017.json",
        image_size=96
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    dataloader_iter = itertools.cycle(dataloader)
    
    for _ in range(10):  # 10 batches is enough for calibration
        batch = next(dataloader_iter)
        images = batch[0].numpy()  # convert PyTorch tensor to numpy
        images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC for TFLite
        yield [images.astype("float32")]

# 3) Create TFLite converter from FP32 TFLite
converter = tf.lite.TFLiteConverter.from_saved_model("./model_tf")  
# If your model is already a tflite file, you may need to convert via SavedModel first

# 4) Enable full integer quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT] # enables post-training optimization
converter.representative_dataset = representative_dataset_gen

converter.experimental_new_converter = True
converter._experimental_disable_per_channel = False

# Force INT8 input/output: Important for deployment
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8] # ensures only INT8 operations are used
converter.inference_input_type = tf.int8   # or tf.uint8
converter.inference_output_type = tf.int8  # or tf.uint8

# 5) Convert
tflite_quant_model = converter.convert()

# 6) Save
with open("PicoSAM2_student_int8.tflite", "wb") as f:
    f.write(tflite_quant_model)