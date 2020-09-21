import torch
import torch.nn as nn
from fastai.vision.all import (
    L, Path, TensorCategory, Category,
    Transform, Pipeline, ToTensor, Normalize, Resize, IntToFloatTensor,
    PILImage, PILImageBW, Image, ResizeMethod, PadMode,
    Categorize,
    RandomSplitter,
    Datasets,
    Learner,
    get_image_files,
    show_image,
    Metric,
    TensorImage,
    load_learner,
)
import random
import numpy as np
import re


def seed_all(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def label_func(path: Path):
    """
    Read label from file name
    :param path: path of file
    :return: label
    """
    return list(path.name.split('.')[0])


class CategorizeList(Categorize):
    def __init__(self, vocab=None, add_na=False, blank='-'):
        super(CategorizeList, self).__init__(vocab=vocab, add_na=add_na, sort=False, )
        self.blank = blank

    @property
    def blank_idx(self):
        return self.vocab.o2i[self.blank]

    @property
    def n_classes(self):
        return len(self.vocab.items)

    def setups(self, dsets):
        dsets = sum(dsets, L(self.blank))
        super(CategorizeList, self).setups(dsets=dsets)

    def encodes(self, os):
        return TensorCategory([self.vocab.o2i[o] for o in os])

    def decodes(self, os, raw=False):
        s = ''.join([self.vocab[o] for o in os])
        if not raw:
            s = re.sub(self.blank, '', re.sub(r'(\w)\1+', r'\1', s))

        return Category(s)


class BeforeBatchTransform(Transform):
    """
    Resize image before create batch
    """

    def __init__(self, height=32, width=32 * 5, keep_ratio=False, min_ratio=5.):
        super(BeforeBatchTransform, self).__init__()
        self.height, self.width = height, width
        self.keep_ratio, self.min_ratio = keep_ratio, min_ratio

    def encodes(self, items):
        images, *labels = zip(*items)

        height, width = self.height, self.width

        if self.keep_ratio:
            max_ratio = self.min_ratio
            for image in images:
                w, h = image.size
                max_ratio = max(max_ratio, w / h)
            width = int(np.floor(height * max_ratio))

        rs_tfm = Resize(size=(height, width), method=ResizeMethod.Pad, pad_mode=PadMode.Border)
        images = [rs_tfm(image) for image in images]
        return zip(images, *labels)


class CreateBatchTransform(Transform):
    """
    Create batch
    """

    def __init__(self):
        super(CreateBatchTransform, self).__init__()
        self.pipeline = Pipeline(funcs=[ToTensor, ])

    def encodes(self, items):
        images, *labels = zip(*items)

        # process images
        images = self.pipeline(images)
        xs = TensorImage(torch.stack(images, dim=0))

        # process labels
        if len(labels) > 0:
            ys = labels[0]
            y_lengths = torch.LongTensor([y.size(0) for y in ys])
            ys = torch.cat(ys, dim=0)
            return xs, (ys, y_lengths)
        return xs,


def conv_block(in_c, out_c, ks, stride, p, bn=False, leaky_relu=False):
    layers = list()
    layers.append(nn.Conv2d(
        in_channels=in_c,
        out_channels=out_c,
        kernel_size=ks,
        stride=stride,
        padding=p,
    ))

    if bn:
        layers.append(nn.BatchNorm2d(num_features=out_c))

    if leaky_relu:
        layers.append(nn.LeakyReLU(0.2, True))
    else:
        layers.append(nn.ReLU())
    return layers


class CNN(nn.Module):
    def __init__(self, in_channels=3, leaky_relu=False, ):
        super(CNN, self).__init__()
        ks = [3, 3, 3, 3, 3, 3, 2]
        s = [1, 1, 1, 1, 1, 1, 1]
        p = [1, 1, 1, 1, 1, 1, 0]
        c = [64, 128, 256, 256, 512, 512, 512]
        mp = [(2, 2), (2, 2), None, ((1, 2), 2), None, ((1, 2), 2), None]
        bn = [False, False, False, False, True, True, False]

        layers = []
        for i in range(len(ks)):
            in_c = in_channels if i == 0 else c[i - 1]
            layers.extend(
                conv_block(in_c=in_c, out_c=c[i], ks=ks[i], stride=s[i], p=p[i], bn=bn[i], leaky_relu=leaky_relu))
            if mp[i] is not None:
                kernel_size, stride = mp[i]
                layers.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride))

        self.cnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.cnn(x)
        return out


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_first=False, bidirectional=True):
        super(RNN, self).__init__()
        self.input_size, self.hidden_size, self.output_size = input_size, hidden_size, output_size
        self.batch_first, self.bidirectional = batch_first, bidirectional

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            batch_first=batch_first,
        )

        self.h2o = nn.Linear(in_features=hidden_size * 2 if bidirectional else hidden_size, out_features=output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.h2o(out)


class CRNN(nn.Module):
    def __init__(self, in_channels, rnn_hidden_size, n_classes, leaky_relu=False):
        super(CRNN, self).__init__()
        self.cnn = CNN(in_channels=in_channels, leaky_relu=leaky_relu)
        self.rnn = nn.Sequential(
            RNN(
                input_size=512,
                hidden_size=rnn_hidden_size,
                output_size=rnn_hidden_size,
                batch_first=False,
                bidirectional=True
            ),
            RNN(
                input_size=rnn_hidden_size,
                hidden_size=rnn_hidden_size,
                output_size=n_classes,
                batch_first=False,
                bidirectional=True
            ),
        )

    def forward(self, x):
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.squeeze(2).permute(2, 0, 1)
        rnn_out = self.rnn(cnn_out)
        return rnn_out


class CTCLoss(nn.Module):
    def __init__(self, blank=0):
        super(CTCLoss, self).__init__()
        self.ctc_loss = nn.CTCLoss(blank=blank, zero_infinity=True)

    def forward(self, output, target):
        T, N, C = output.size()
        target, target_lengths = target
        output_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long, device=output.device)
        output = output.log_softmax(2)
        return self.ctc_loss(output, target, output_lengths, target_lengths)


class AccMetric(Metric):
    def __init__(self):
        self.y_true, self.y_pred = [], []

    def reset(self):
        self.y_true, self.y_pred = [], []

    def accumulate(self, learn):
        label_categorize = learn.dls.tfms[1][-1]
        yb_pred = learn.pred.permute(1, 0, 2).argmax(dim=2)
        (yb, y_lengths), = learn.yb

        yb = torch.split(yb, y_lengths.cpu().tolist())
        self.y_true.extend([label_categorize.decode(y, raw=True) for y in yb])

        self.y_pred.extend([label_categorize.decode(y) for y in yb_pred])

    @property
    def value(self):
        #         print(self.y_pred[:4], self.y_true[:4])
        n_correct = (np.array(self.y_pred) == np.array(self.y_true)).sum()
        return n_correct / float(len(self.y_true))

    @property
    def name(self):
        return 'accuracy'


