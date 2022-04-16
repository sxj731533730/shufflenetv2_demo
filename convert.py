import torch
# import trochvision
import torch.utils.data
import argparse
import onnxruntime
from model import shufflenet_v2_x1_0
import os
import cv2
import numpy as np
from torch.autograd import Variable
from onnxruntime.datasets import get_example


def main(args):
    # print("the version of torch is {}".format(torch.__version__))
    dummy_input=getInput(args.img_size)#获得网络的输入
    # 加载模型
    model = shufflenet_v2_x1_0(num_classes=6)
    model_dict =  model.state_dict()
    if args.model_path:
        if os.path.isfile(args.model_path):
            print(("=> start loading checkpoint '{}'".format(args.model_path)))
            model = torch.load(args.model_path)
            print("load cls model successfully")
        else:
            print(("=> no checkpoint found at '{}'".format(args.model_path)))
            return
    model.to('cpu')
    model.eval()
    pre=model(dummy_input)
    print("the pre:{}".format(pre))
    #保存onnx模型
    torch2onnx(args,model,dummy_input)

def getInput(img_size):
    input = cv2.imread(r'G:\sxj731533730\sxj731533730\csdn\Test7_shufflenet\data_set\val\JI22CK\result1-0.jpg')
    input = cv2.resize(input, (img_size, img_size))  # hwc bgr
    input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)  # hwc rgb
    # [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    input = np.transpose(input, (2, 0, 1)).astype(np.float32)  # chw rgb
    # input=input/255.0
    print("befor the input[0,0,0]:{}".format(input[0, 0, 0]))
    print("the size of input[0,...] is {}".format(input[0, ...].shape))
    print("the size of input[1,...] is {}".format(input[1, ...].shape))
    print("the size of input[2,...] is {}".format(input[2, ...].shape))
    input[0, ...] = ((input[0, ...]/255.0)-0.485)/0.229
    input[1, ...] = ((input[1, ...]/255.0)-0.456)/0.224
    input[2, ...] = ((input[2, ...]/255.0)-0.406)/0.225
    print("after input[0,0,0]:{}".format(input[0, 0, 0]))

    now_image1 = Variable(torch.from_numpy(input))
    dummy_input = now_image1.unsqueeze(0)
    return dummy_input


def torch2onnx(args,model,dummy_input):
    input_names = ['input']#模型输入的name
    output_names = ['output']#模型输出的name
    # return
    torch_out = torch.onnx._export(model, dummy_input, os.path.join(args.save_model_path,r"G:\sxj731533730\sxj731533730\csdn\Test7_shufflenet\weights\model-149.onnx"),
                                   verbose=True, input_names=input_names, output_names=output_names)
    # test onnx model
    example_model = get_example(os.path.join(args.save_model_path,r"G:\sxj731533730\sxj731533730\csdn\Test7_shufflenet\weights\model-149.onnx"))
    session = onnxruntime.InferenceSession(example_model)
    # get the name of the first input of the model
    input_name = session.get_inputs()[0].name
    print('Input Name:', input_name)
    result = session.run([], {input_name: dummy_input.data.numpy()})
    # np.testing.assert_almost_equal(
    #     torch_out.data.cpu().numpy(), result[0], decimal=3)
    print("the result is {}".format(result[0]))
    #结果对比--有点精度上的损失
    # pytorch tensor([[ 5.8738, -5.4470]], device='cuda:0')
    # onnx [ 5.6525207 -5.2962923]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PyTorch model to onnx and ncnn")
    parser.add_argument('--model_path', type=str, default=r"G:\sxj731533730\sxj731533730\csdn\Test7_shufflenet\weights\model-149.pth",
                        help="For training from one model_file")
    parser.add_argument('--save_model_path', type=str, default=r"G:\sxj731533730\sxj731533730\csdn\Test7_shufflenet\weights\model-149.pth",
                        help="For training from one model_file")
    parser.add_argument('--onnx_model_path', type=str, default=r"G:\sxj731533730\sxj731533730\csdn\Test7_shufflenet\weights\model-149.pth",
                        help="For training from one model_file")
    parser.add_argument('--img_size', type=int, default=128,
                        help="the image size of model input")
    args = parser.parse_args()
    main(args)
