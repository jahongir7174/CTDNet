import copy
import math
import random

import numpy
import torch


def setup_seed():
    """
    Setup random seed.
    """
    random.seed(0)
    numpy.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_multi_processes():
    """
    Setup multi-processing environment variables.
    """
    import cv2
    from os import environ
    from platform import system

    # set multiprocess start method as `fork` to speed up the training
    if system() != 'Windows':
        torch.multiprocessing.set_start_method('fork', force=True)

    # disable opencv multithreading to avoid system being overloaded
    cv2.setNumThreads(0)

    # setup OMP threads
    if 'OMP_NUM_THREADS' not in environ:
        environ['OMP_NUM_THREADS'] = '1'

    # setup MKL threads
    if 'MKL_NUM_THREADS' not in environ:
        environ['MKL_NUM_THREADS'] = '1'


def export_onnx(args):
    import onnx  # noqa

    inputs = ['images']
    outputs = ['outputs']
    dynamic = {'outputs': {0: 'batch', 1: 'anchors'}}

    m = torch.load('./weights/best.pt')['model'].float().fuse()
    x = torch.zeros((1, 3, args.input_size, args.input_size))

    torch.onnx.export(m.cpu(), x.cpu(),
                      './weights/best.onnx',
                      verbose=False,
                      opset_version=12,
                      # WARNING: DNN inference with torch>=1.12 may require do_constant_folding=False
                      do_constant_folding=True,
                      input_names=inputs,
                      output_names=outputs,
                      dynamic_axes=dynamic or None)

    # Checks
    model_onnx = onnx.load('./weights/best.onnx')  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model

    onnx.save(model_onnx, './weights/best.onnx')


def strip_optimizer(filename):
    x = torch.load(filename, map_location=torch.device('cpu'))
    x['model'].half()  # to FP16
    for p in x['model'].parameters():
        p.requires_grad = False
    torch.save(x, filename)


def load_checkpoint(model, ckpt, prefix='backbone.'):
    dst = model.state_dict()
    src = torch.load(ckpt)['model'].float().state_dict()
    ckpt = {}
    for k, v in src.items():
        if prefix + k in dst and v.shape == dst[prefix + k].shape:
            ckpt[prefix + k] = v
    model.load_state_dict(state_dict=ckpt, strict=False)
    return model


def weight_decay(model, lr):
    p1 = []
    p2 = []
    p3 = []
    p4 = []
    norm = tuple(v for k, v in torch.nn.__dict__.items() if "Norm" in k)
    for name, m in model.named_modules():
        if 'backbone' in name:
            for n, p in m.named_parameters(recurse=0):
                if n == "bias":  # bias (no decay)
                    p1.append(p)
                elif n == "weight" and isinstance(m, norm):  # weight (no decay)
                    p1.append(p)
                else:
                    p2.append(p)  # weight (with decay)
        else:
            for n, p in m.named_parameters(recurse=0):
                if n == "bias":  # bias (no decay)
                    p3.append(p)
                elif n == "weight" and isinstance(m, norm):  # weight (no decay)
                    p3.append(p)
                else:
                    p4.append(p)  # weight (with decay)
    return [{'params': p1, 'weight_decay': 0.00, 'lr': lr / 10},
            {'params': p3, 'weight_decay': 0.00, 'lr': lr},
            {'params': p2, 'weight_decay': 1E-4, 'lr': lr / 10},
            {'params': p4, 'weight_decay': 1E-4, 'lr': lr}]


def plot_lr(args, optimizer, scheduler, num_steps):
    from matplotlib import pyplot

    optimizer = copy.copy(optimizer)
    scheduler = copy.copy(scheduler)

    y = []
    for epoch in range(args.epochs):
        for i in range(num_steps):
            step = i + num_steps * epoch
            scheduler.step(step, optimizer)
            y.append(optimizer.param_groups[0]['lr'])
    pyplot.plot(y, '.-', label='LR')
    pyplot.xlabel('step')
    pyplot.ylabel('LR')
    pyplot.grid()
    pyplot.xlim(0, args.epochs * num_steps)
    pyplot.ylim(0)
    pyplot.savefig('./weights/lr.png', dpi=200)
    pyplot.close()


def compute_metric(outputs, targets, num):
    eps = numpy.finfo(float).eps

    shape = outputs.shape
    vector = torch.linspace(1. / (num + 1), 1. - 1. / (num + 1), num, device=targets.device)

    outputs = outputs.detach()
    targets = targets.detach()
    targets.requires_grad = False
    outputs.requires_grad = False
    outputs = outputs.view(shape[0], -1)
    targets = targets.view(shape[0], -1)
    length = outputs.shape[1]
    output = outputs.expand(num, shape[0], length)
    target = targets.expand(num, shape[0], length)
    vector = vector.expand(shape[0], length, num).permute(2, 0, 1)

    bi_res = (outputs > vector).float()
    intersect = (target * bi_res).sum(dim=2)
    recall = (intersect / (target.sum(dim=2) + eps)).sum(dim=1)
    precision = (intersect / (bi_res.sum(dim=2) + eps)).sum(dim=1)
    mae = (output[0] - target[0]).abs().sum() / length

    return precision, recall, mae


class CosineLR:
    def __init__(self, args, optimizer):
        self.min_lr = 1E-6
        self.epochs = args.epochs
        self.learning_rates = [x['lr'] for x in optimizer.param_groups]

    def step(self, epoch, optimizer):
        param_groups = optimizer.param_groups
        for param_group, lr in zip(param_groups, self.learning_rates):
            alpha = math.cos(math.pi * epoch / self.epochs)
            lr = 0.5 * (lr - self.min_lr) * (1 + alpha)
            param_group['lr'] = self.min_lr + lr


class EMA:
    """
    Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9995, tau=2000, updates=0):
        # Create EMA
        self.ema = copy.deepcopy(model).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        # decay exponential ramp (to help early epochs)
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        if hasattr(model, 'module'):
            model = model.module
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()


class AverageMeter:
    def __init__(self):
        self.num = 0
        self.sum = 0
        self.avg = 0

    def update(self, v, n):
        self.num = self.num + n
        self.sum = self.sum + v * n
        self.avg = self.sum / self.num


class EdgeLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.max_pool = torch.nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, outputs, targets):
        targets = self.mask_to_edge(targets)
        return self.bce_loss(outputs, targets)

    def mask_to_edge(self, targets):
        erosion = 1 - self.max_pool(1 - targets)  # erosion
        dilation = self.max_pool(targets)  # dilation
        return dilation - erosion


class DiceLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1E-5
        self.sigmoid = torch.nn.Sigmoid()
        self.bce_loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, outputs, targets):
        bce = self.bce_loss(outputs, targets)
        outputs = self.sigmoid(outputs)
        dice = (2 * (outputs * targets).sum() + self.eps) / (outputs.sum() + targets.sum() + self.eps)
        return bce + 1 - dice


class ComputeLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.edge_loss = EdgeLoss()
        self.dice_loss = DiceLoss()

    def forward(self, outputs, targets):
        loss1 = self.dice_loss(outputs[:, 0, :, :], targets)
        loss2 = self.dice_loss(outputs[:, 1, :, :], targets)
        loss3 = self.dice_loss(outputs[:, 2, :, :], targets)
        loss4 = self.dice_loss(outputs[:, 3, :, :], targets)
        loss5 = self.dice_loss(outputs[:, 4, :, :], targets)
        loss6 = self.edge_loss(outputs[:, 5, :, :], targets)
        return loss1 + loss2 + loss3 + loss4 + loss5 + loss6
