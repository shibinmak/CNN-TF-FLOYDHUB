import os
import glob2
import cv2
import numpy as np
from sklearn.utils import shuffle


def load_image(image_size):
    dir_path = os.path.dirname(os.path.realpath('__file__'))
    image_path = os.path.join(dir_path, 'image')
    images = []
    labels = []
    image_names = []
    category = []
    classes = os.listdir(image_path)
    for items in classes:
        index = classes.index(items)
        print('loading images of {}'.format(items))
        path = os.path.join(image_path, items, '*g')
        files = glob2.glob(path)

        for image_file in files:
            image = cv2.imread(image_file)
            image = cv2.resize(image, (image_size, image_size), fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            image = np.multiply(image, 1.0/255)
            images.append(image)
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels.append(label)
            filename = os.path.basename(image_file)
            image_names.append(filename)
            category.append(items)

    images = np.array(images)
    labels = np.array(labels)
    image_names = np.array(image_names)
    category = np.array(category)

    return images, labels, image_names, category


class ImageData(object):
    def __init__(self, images, labels, image_names, category):
        self._total_images = images.shape[0]
        self._images = images
        self._labels = labels
        self._image_names = image_names
        self._category = category
        self._epochs = 0
        self._processed = 0

    @property
    def total_images(self):
        return self._total_images

    @property
    def images(self):
        return self._images

    @property
    def lables(self):
        return self._labels

    @property
    def image_names(self):
        return self._image_names

    @property
    def category(self):
        return self._category

    @property
    def epochs(self):
        return self._epochs

    @property
    def processed(self):
        return self._processed

    def batch(self, batch_size):
        start = self._processed
        self._processed += batch_size

        if self._processed > self._total_images:
            start = 0
            self._epochs += 1
            assert batch_size <= self._total_images
        end = self._processed

        return self._images[start:end], self._labels[start:end], self._image_names[start:end], self._category[start:end]


def train_test_split(image_size,test_size):

    class Image:
        pass
    data = Image()

    images, labels, image_names, category = load_image(image_size=128)
    images, labels, image_names, category = shuffle(images, labels, image_names, category)

    if isinstance(test_size,float):
        test_size = int(test_size*images.shape[0])

    train_images = images[test_size:]
    train_labels = labels[test_size:]
    train_image_names = image_names[test_size:]
    train_category = category[test_size:]

    test_images = images[:test_size]
    test_labels = labels[:test_size]
    test_image_names = image_names[:test_size]
    test_category = category[:test_size]

    data.train = ImageData(train_images, train_labels, train_image_names, train_category)
    data.test = ImageData(test_images, test_labels, test_image_names, test_category)

    return data









