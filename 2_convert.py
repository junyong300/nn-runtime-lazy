
"""LazyNet Model Converter \
    from Pytorch to ONNX, TFLite, and Pytorch Mobile
"""
import torch
import torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile
import model
from utils import *
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import numpy as  np
import yaml
from model.lazy import LazyNet
from argparse import ArgumentParser
from model.backbone import get_backbone

def parse_args():
    """Argument Parser function"""
    parser = ArgumentParser(
        description='convertion to onnx, tf, tflite')
    parser.add_argument('-c', type=str, default="./cfgs/c10/7.yml",
                        help='model name (default: bisenet_resnet34)')
    return parser.parse_args()

def main():
    """convert pytorch model to:
    pytorch mobile
    pytorch mobile optimized for mobile
    onnx
    tensorflow
    tflite
    """
    args = parse_args()
    cfg = read_yaml(args.c)

    device = 'cuda:0'

    if cfg['mode'] == 'lazy':
        model = LazyNet(cfg).to(device)
    
    else:
        model = get_backbone(cfg).to(device)
    SIZE=cfg['img_size']
    name = '{}_{}_{}'.format(cfg['dataset'], cfg['backbone'], cfg['mode'])
    folder = './outputs/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    print("loading model")
    model.load_state_dict(
            torch.load("./checkpoints/" + name + '.pth'))

    model.eval()
    model.train_mode = False
    example = torch.rand(1, 3, SIZE, SIZE).to(device)
    print("trace model...")
    traced_script_module = torch.jit.trace(model, example)
    print("optimizing for mobile...")
    traced_script_module_optimized = optimize_for_mobile(traced_script_module)
    print("transforming to ptl...")
    traced_script_module_optimized._save_for_lite_interpreter(
        folder + name + ".ptl")
    print("converting to onnx...")
    dummy_input = torch.randn(1, 3, SIZE, SIZE).to(device)
    input_names = ['input_image']
    output_names = ['preds',]
    torch.onnx.export(model, dummy_input, folder + name + ".onnx",
        input_names=input_names, output_names=output_names,
        verbose=False, opset_version=11)

    print("load onnx...")
    onnx_model = onnx.load(folder + name + ".onnx")
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(folder + name)

    print("convert to tf")
    model = tf.saved_model.load(folder + name)
    model.trainable = False
    input_tensor = tf.random.uniform([1, 3, SIZE, SIZE])
    out = model(**{'input_image': input_tensor})
    print("<<<convert to tflite>>>")
    converter = tf.lite.TFLiteConverter.from_saved_model(folder + name)
    tflite_model = converter.convert()

    # Save the model
    with open(folder + name + ".tflite", 'wb') as f:
        f.write(tflite_model)

    print("interpret tflite")
    interpreter = tf.lite.Interpreter(model_path= folder + name + ".tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data
    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # get_tensor() returns a copy of the tensor data
    # use tensor() in order to get a pointer to the tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data.shape)


if __name__ == '__main__':
    main()