import random

import cv2
import numpy
import torch
from torch.utils import data


class Dataset(data.Dataset):
    def __init__(self, filenames, input_size, train):
        self.train = train
        self.input_size = input_size

        self.mean = numpy.array([0.406, 0.456, 0.485]).reshape((1, 1, 3)).astype('float32')
        self.std = numpy.array([0.225, 0.224, 0.229]).reshape((1, 1, 3)).astype('float32')

        self.filenames = filenames

    def __getitem__(self, index):
        filename = self.filenames[index]

        image = cv2.imread(filename)
        label = cv2.imread(filename[:-4] + '.png', flags=0)

        if self.train:
            image, label = random_crop(image, label)
            # HSV color-space
            augment_hsv(image)
            # Flip up-down
            if random.random() < 0.0:
                image = numpy.flipud(image)
                label = numpy.flipud(label)
            # Flip left-right
            if random.random() < 0.5:
                image = numpy.fliplr(image)
                label = numpy.fliplr(label)
            image, label = self.normalize(image, label)
        else:
            image = cv2.resize(image, dsize=(self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, dsize=(self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)
            image, label = self.normalize(image, label)
            image, label = to_tensor(image, label)

        return image, label

    def __len__(self):
        return len(self.filenames)

    def normalize(self, image, label):
        image = image.astype('float32') / 255.0
        label = (label > 128).astype('int64')
        image = image - self.mean
        image = image / self.std
        return image, label

    @staticmethod
    def collate_fn(batch):
        size = [224, 256, 288, 320, 352, 384]
        size = random.choice(size)
        images, labels = zip(*batch)
        images = list(images)
        labels = list(labels)
        for i in range(len(images)):
            images[i] = cv2.resize(images[i], dsize=(size, size), interpolation=resample())
            labels[i] = cv2.resize(labels[i], dsize=(size, size), interpolation=cv2.INTER_NEAREST)
            images[i], labels[i] = to_tensor(images[i], labels[i])
        images = torch.stack(images)
        labels = torch.stack(labels)
        return images, labels


def to_tensor(image, label):
    # Convert HWC to CHW, BGR to RGB
    image = image.transpose((2, 0, 1))[::-1]
    image = numpy.ascontiguousarray(image)
    image = torch.from_numpy(image)
    label = torch.from_numpy(label)
    return image, label


def random_crop(image, label):
    shape = image.shape

    h = numpy.random.randint(shape[0] / 8)
    w = numpy.random.randint(shape[1] / 8)

    offset_h = 0 if h == 0 else numpy.random.randint(h)
    offset_w = 0 if w == 0 else numpy.random.randint(w)

    y1 = offset_h
    x1 = offset_w
    x2 = shape[1] + offset_w - w
    y2 = shape[0] + offset_h - h
    return image[y1:y2, x1:x2, :], label[y1:y2, x1:x2]


def augment_hsv(image):
    # HSV color-space augmentation
    h = 0.015000
    s = 0.700000
    v = 0.400000

    r = numpy.random.uniform(-1, 1, 3) * [h, s, v] + 1
    h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))

    x = numpy.arange(0, 256, dtype=r.dtype)
    lut_h = ((x * r[0]) % 180).astype('uint8')
    lut_s = numpy.clip(x * r[1], 0, 255).astype('uint8')
    lut_v = numpy.clip(x * r[2], 0, 255).astype('uint8')

    im_hsv = cv2.merge((cv2.LUT(h, lut_h), cv2.LUT(s, lut_s), cv2.LUT(v, lut_v)))
    cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=image)  # no return needed


def resample():
    choices = (cv2.INTER_AREA,
               cv2.INTER_CUBIC,
               cv2.INTER_LINEAR,
               cv2.INTER_NEAREST,
               cv2.INTER_LANCZOS4)
    return random.choice(seq=choices)
