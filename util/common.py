import os
import torch
import random
import cv2
import numpy as np
import matplotlib
import torch.nn.functional as F
matplotlib.use('Agg')

def init_seed(seed=777):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    from torch.backends import cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True

def compute_metrics(tn_fp_fn_tps):
    p, r, f1, miou, iou_0, iou_1, oa, kappa = [], [], [], [], [], [], [], []
    count = 1
    for tn_fp_fn_tp in tn_fp_fn_tps:
        if count == 2:
            break
        count += 1

        tn, fp, fn, tp = tn_fp_fn_tp
        p_tmp = tp / (tp + fp)
        r_tmp = tp / (tp + fn)
        f1_tmp = 2 * p_tmp * r_tmp / (r_tmp + p_tmp)
        iou_0_tmp = tn / (tn + fp + fn)
        iou_1_tmp = tp / (tp + fp + fn)
        miou_tmp = 0.5 * tp / (tp + fp + fn) + 0.5 * tn / (tn + fp + fn)
        oa_tmp = (tp + tn) / (tp + tn + fp + fn)
        p0 = oa_tmp
        pe = ((tp+fp)*(tp+fn)+(fp+tn)*(fn+tn))/(tp+fp+tn+fn)**2
        kappa_tmp = (p0-pe) / (1-pe)
        
        p.append(p_tmp)
        r.append(r_tmp)
        f1.append(f1_tmp)
        miou.append(miou_tmp)
        iou_0.append(iou_0_tmp)
        iou_1.append(iou_1_tmp)
        oa.append(oa_tmp)
        kappa.append(kappa_tmp)
        
        print('Precision: {}\nRecall: {}\nF1-Score: {} \nmIOU:{} \nIOU_0:{} \nIOU_1:{}'.format(p,r,f1,miou,iou_0,iou_1))
        print('OA: {}\nKappa: {}'.format(oa,kappa))

def gpu_info():
    print("\n" + "-" * 30 + "GPU Info" + "-" * 30)
    gpu_count = torch.cuda.device_count()
    x = [torch.cuda.get_device_properties(i) for i in range(gpu_count)]
    s = 'Using CUDA '
    c = 1024 ** 2  # bytes to MB
    if gpu_count > 0:
        print("Using GPU count: {}".format(torch.cuda.device_count()))
        for i in range(0, gpu_count):
            if i == 1:
                s = ' ' * len(s)
            print("%sdevice%g name='%s', memory=%dMB" % (s, i, x[i].name, x[i].total_memory / c))
    else:
        print("Using CPU !!!")

class ScaleInOutput:
    def __init__(self, input_size=512):
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        self.input_size = input_size
        self.output_size = None

    def scale_input(self, imgs: tuple):
        assert isinstance(imgs, tuple), "Please check the input type. It should be a 'tuple'."
        imgs = list(imgs)
        self.output_size = imgs[0].shape[2:]
        for i, img in enumerate(imgs):
            imgs[i] = F.interpolate(img, self.input_size, mode='bilinear', align_corners=True)
        return tuple(imgs)

    def scale_output(self, outs: tuple):
        if type(outs) in [torch.Tensor]:
            outs = (outs,)
        assert isinstance(outs, tuple), "Please check the input type. It should be a 'tuple'."
        outs = list(outs)
        assert self.output_size is not None, \
            "Please call 'scale_input' function firstly, to make sure 'output_size' is not None"
        for i, out in enumerate(outs):
            outs[i] = F.interpolate(out, self.output_size, mode='bilinear', align_corners=True)
        return tuple(outs)

