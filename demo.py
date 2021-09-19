import string
import argparse
import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import time
from utils import CTCLabelConverter, AttnLabelConverter
from dataset import RawDataset, AlignCollate
from model import Model
from collections import OrderedDict
import torch.quantization
import os

device = torch.device('cpu')# if torch.cuda.is_available() else 'cpu')

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

def model_equivalence(model_1, model_2, device, rtol=1e-05, atol=1e-08, num_tests=100, input_size=(1,3,32,32)):

    model_1.to(device)
    model_2.to(device)

    for _ in range(num_tests):
        x = torch.rand(size=input_size).to(device)
        y1 = model_1(x).detach().cpu().numpy()
        y2 = model_2(x).detach().cpu().numpy()
        if np.allclose(a=y1, b=y2, rtol=rtol, atol=atol, equal_nan=False) == False:
            print("Model equivalence test sample failed: ")
            print(y1)
            print(y2)
            return False

    return True

class QuantizedModel(torch.nn.Module):
    def __init__(self, model_fp32):
        super(QuantizedModel, self).__init__()
        # QuantStub converts tensors from floating point to quantized.
        # This will only be used for inputs.
        self.quant = torch.quantization.QuantStub()
        # DeQuantStub converts tensors from quantized to floating point.
        # This will only be used for outputs.
        self.dequant = torch.quantization.DeQuantStub()
        # FP32 model
        self.model_fp32 = model_fp32

    def forward(self, x, text, is_train=True):
        # manually specify where tensors will be converted from floating
        # point to quantized in the quantized model
        x = self.quant(x)
        x = self.model_fp32(x, text, is_train=True)
        # manually specify where tensors will be converted from quantized
        # to floating point in the quantized model
        x = self.dequant(x)
        return x

def demo(opt):
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)

    # model = torch.nn.DataParallel(model).cpu()

    # load model
    print('loading pretrained model from %s' % opt.saved_model)

    """ Load parameters for run on CPU"""
    new_state_dict = OrderedDict()
    state_dict = torch.load(opt.saved_model, map_location=device)

    # for k, v in state_dict.items():
    #     name = k[7:] # remove `module.`
    #     new_state_dict[name] = v.to(device)
    # # load params
    # model.load_state_dict(new_state_dict)
    model.load_state_dict(state_dict)
    # model.load_state_dict(torch.load(opt.saved_model, map_location=device))

    # # insert observers
    # torch.quantization.prepare(model, inplace=True)
    # # Calibrate the model and collect statistics

    # # convert to quantized version
    # torch.quantization.convert(model, inplace=True)

    # model.to(device)

    
    # print_size_of_model(quantized_model)

    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    demo_data = RawDataset(root=opt.image_folder, opt=opt)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)

    # predict
    # model.eval()


    # model_static_quantized = torch.quantization.quantize_dynamic(
    # model, qconfig_spec={
    # model.FeatureExtraction.ConvNet.conv0_1,
    # model.FeatureExtraction.ConvNet.bn0_1,
    # model.FeatureExtraction.ConvNet.conv0_2,
    # model.FeatureExtraction.ConvNet.bn0_2,
    # model.FeatureExtraction.ConvNet.relu,
    # model.FeatureExtraction.ConvNet.maxpool1,
    # model.FeatureExtraction.ConvNet.layer1,
    # model.FeatureExtraction.ConvNet.conv1,
    # model.FeatureExtraction.ConvNet.bn1,
    # model.FeatureExtraction.ConvNet.maxpool2,
    # model.FeatureExtraction.ConvNet.layer2,
    # model.FeatureExtraction.ConvNet.conv2,
    # model.FeatureExtraction.ConvNet.bn2,
    # model.FeatureExtraction.ConvNet.maxpool3,
    # model.FeatureExtraction.ConvNet.layer3,
    # model.FeatureExtraction.ConvNet.conv3,
    # model.FeatureExtraction.ConvNet.bn3,
    # model.FeatureExtraction.ConvNet.layer4,
    # model.FeatureExtraction.ConvNet.conv4_1,
    # model.FeatureExtraction.ConvNet.bn4_1,
    # model.FeatureExtraction.ConvNet.conv4_2,
    # model.FeatureExtraction.ConvNet.bn4_2,
    # }
    # , dtype=torch.qint8
    # )


    # import copy
    # fused_model = copy.deepcopy(model)
    # fused_model.eval()
    # # fused_model = torch.quantization.fuse_modules(fused_model, [["conv1", "bn1", "relu"]], inplace=True)
    # for module_name, module in fused_model.named_children():
    #     # print("##################", module_name)
    #     if "layer" in module_name:
    #         # print("######### here eeeeeeeeee #########")
    #         for basic_block_name, basic_block in module.named_children():
    #             torch.quantization.fuse_modules(basic_block, [["conv1", "bn1", "relu1"], ["conv2", "bn2"]], inplace=True)
    #             for sub_block_name, sub_block in basic_block.named_children():
    #                 if sub_block_name == "downsample":
    #                     torch.quantization.fuse_modules(sub_block, [["0", "1"]], inplace=True)

    # assert model_equivalence(model_1=model, model_2=fused_model, device=cpu_device, rtol=1e-03, atol=1e-06, num_tests=100, input_size=(1,3,32,32)), "Fused model is not equivalent to the original model!"
    quantized_model = QuantizedModel(model_fp32=fused_model)
    quantization_config = torch.quantization.get_default_qconfig("fbgemm")
    quantized_model.qconfig = quantization_config
    torch.quantization.prepare(quantized_model, inplace=True)
    model_static_quantized = torch.quantization.convert(quantized_model, inplace=True)
    model_static_quantized.eval()


    # backend = "qnnpack"
    # model.qconfig = torch.quantization.get_default_qconfig(backend)
    # torch.backends.quantized.engine = backend
    # model_static_quantized = torch.quantization.prepare(model, inplace=False)
    # model_static_quantized = torch.quantization.convert(model_static_quantized, inplace=False)
    # model_static_quantized.eval()

    

    # # print(quantized_model)
    print_size_of_model(model)
    print_size_of_model(model_static_quantized)
    with torch.no_grad():
        # totall = 0
        ts = time.time()
        print('#'*25)
        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            if 'CTC' in opt.Prediction:
                preds = model_static_quantized(image, text_for_pred)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                # preds_index = preds_index.view(-1)
                preds_str = converter.decode(preds_index, preds_size)

            else:
                # ts = time.time()

                preds = model_static_quantized(image, text_for_pred, is_train=False)

                # totall += time.time() - ts
                # print("### tt ###", time.time() - ts)

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)


            # log = open(f'./log_demo_result.txt', 'a')
            # dashed_line = '-' * 80
            # head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'
            
            # print(f'{dashed_line}\n{head}\n{dashed_line}')
            # log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')

            # preds_prob = F.softmax(preds, dim=2)
            # preds_max_prob, _ = preds_prob.max(dim=2)
            # for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
            #     if 'Attn' in opt.Prediction:
            #         pred_EOS = pred.find('[s]')
            #         pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
            #         pred_max_prob = pred_max_prob[:pred_EOS]

            #     # calculate confidence score (= multiply of pred_max_prob)
            #     # print('##########################')
            #     # print(pred_max_prob.cumprod(dim=0))
            #     confidence_score = pred_max_prob.cumprod(dim=0)[-1]

            #     print(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}')
            #     img = cv2.imread(img_name)
            #     img = cv2.putText(img, pred.upper() + '  conf:'+ f'{confidence_score:0.4f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 
            #        0.5, (0, 0, 255), 1, cv2.LINE_AA)
            #     cv2.imwrite('result/test_result/'+img_name.split('/')[-1],img)
            #     log.write(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}\n')

            # log.close()
        print(f'times: {time.time() - ts}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', required=True, help='path to image_folder which contains text images')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=100)
    parser.add_argument('--batch_size', type=int, default=512, help='input batch size')
    parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='1234abcdefghjkl', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True, help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    opt = parser.parse_args()

    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    # cudnn.benchmark = True
    # cudnn.deterministic = True
    # opt.num_gpu = torch.cuda.device_count()

    demo(opt)
