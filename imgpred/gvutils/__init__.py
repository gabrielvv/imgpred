# @see https://matplotlib.org/users/image_tutorial.html

import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import random
import imutils
from imutils import paths
import cv2
import os
from functools import reduce

from .clean_img import clean_img
from .create_sets import create_sets
from .plot_confusion_matrix import plot_confusion_matrix
from .print_save_plot import print_save_plot
from .list_to_file import list_to_file
from .print_matrix import print_matrix, read_matrix


def print_progress(i, slow_factor=0.1):
    sys.stdout.write('\r')
    p = "|/\\"
    p_len = len(p)
    sys.stdout.write("%s\b" % (p[int(i*slow_factor)%p_len]))
    sys.stdout.flush()

def pred_label(cat, not_cat):
    # build the label
    #label = "Cat" if cat > not_cat else "Not Cat"
    #proba = cat if cat > not_cat else not_cat
    return "{0}: {1:.2f}%{2}{3}: {4:.2f}%".format("Cat", cat * 100, os.linesep, "Not Cat", not_cat * 100)

def imgs_stats(path, style="pie"):
    train_val = [[], []]
    test = [[], []]
    for img_path in imutils.paths.list_images(path):
        if img_path.find("train") >= 0:
            if img_path.find("not_cat") >= 0:
                train_val[1].append(img_path)
            elif img_path.find("cat") >= 0:
                train_val[0].append(img_path)
        if img_path.find("test") >= 0:
            if img_path.find("not_cat") >= 0:
                test[1].append(img_path)
            elif img_path.find("cat") >= 0:
                test[0].append(img_path)
    fig,ax=plt.subplots(nrows=1, ncols=2,figsize=(20, 5))

    train_val = [len(train_val[0]), len(train_val[1])]
    test = [len(test[0]), len(test[1])]

    ax[0].set_title('Train+Test1 Set ({})'.format(reduce(lambda a,b:a+b, train_val)))
    ax[1].set_title('Test2 Set ({})'.format(reduce(lambda a,b:a+b, test)))

    if style == "bar":
        ax[0].annotate(str(train_val[0]), (0, train_val[0]))
        ax[0].annotate(str(train_val[1]), (1, train_val[1]))
        ax[1].annotate(str(test[0]), (0, test[0]))
        ax[1].annotate(str(test[1]), (1, test[1]))

        ax[0].bar(range(len(train_val)), train_val, align='center')
        ax[1].bar(range(len(train_val)), test, align='center')

        plt.sca(ax[0])
        plt.xticks(range(2), ["cat","!cat"])
        plt.sca(ax[1])
        plt.xticks(range(2), ["cat","!cat"]);

    if style == "pie":
        #https://matplotlib.org/gallery/pie_and_polar_charts/pie_features.html#sphx-glr-gallery-pie-and-polar-charts-pie-features-py
        ax[0].pie(train_val,
            labels=("cat ({})".format(train_val[0]), "!cat ({})".format(train_val[1])),
            autopct='%1.1f%%',startangle=90)
        ax[1].pie(test,
            labels=("cat ({})".format(test[0]), "!cat ({})".format(test[1])),
            autopct='%1.1f%%',startangle=90)
        # Equal aspect ratio ensures that pie is drawn as a circle.
        ax[0].axis('equal')
        ax[1].axis('equal')

# https://gist.github.com/soply/f3eec2e79c165e39c9d540e916142ae1
def show_images_plt(images, titles = None, cols = 2):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)

# http://answers.opencv.org/question/175912/how-to-display-multiple-images-in-one-window/
def show_images_cv(img_paths, titles = None, cols=2, rows=2):
    WIDTH = 256
    HEIGHT = 168
    OFFSET = 20
    LINE_HEIGHT = 30
    imgs = []
    for i,p in enumerate(img_paths):
        #print(p)
        img = cv2.imread(p)
        resized = imutils.resize(img, width=WIDTH)
        resized = resized[:HEIGHT]
        #print(np.shape(resized))
        imgs.append(resized)

    numpy_horizontal_1 = np.hstack((imgs[0], imgs[1]))
    numpy_horizontal_2 = np.hstack((imgs[2], imgs[3]))
    numpy_vertical = np.vstack((numpy_horizontal_1, numpy_horizontal_2))

    # https://docs.opencv.org/3.3.0/d6/d6e/group__imgproc__draw.html#ga5126f47f883d730f633d74f07456c576
    (h, w, c) = np.shape(numpy_vertical)

    for i in range(rows):
        for j in range(cols):
            for k, line in enumerate(titles[j+i*rows].split(os.linesep)):
                x = WIDTH*j + OFFSET
                y = HEIGHT*i + OFFSET + LINE_HEIGHT*k
                cv2.putText(numpy_vertical, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # cv2.putText(numpy_vertical, titles[0], (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # cv2.putText(numpy_vertical, titles[1], (20+WIDTH,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # cv2.putText(numpy_vertical, titles[2], (20,h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # cv2.putText(numpy_vertical, titles[3], (20+WIDTH,h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # cv2.imshow('Confusion Matrix', numpy_vertical)
    # cv2.waitKey()
    return numpy_vertical

# TODO resize images
# TODO handle case where len(value) == 0
def save_confusion_matrix_samples(imgs, plot_path, service="cv"):
    random.seed(48)
    d = {}
    for key, value in imgs.items():
        (img_path, pred_label) = value[random.randint(0, len(value)-1)]
        k="{0}{1}{2}".format(key.upper(), os.linesep, pred_label)
        v= img_path if service == "cv" else mpimg.imread( img_path )
        d[k] = v
    if service == "cv":
        img = show_images_cv(list(d.values()), list(d.keys()))
        #https://docs.opencv.org/3.3.0/d4/da8/group__imgcodecs.html#gabbc7ef1aa2edfaa87772f1202d67e0ce
        cv2.imwrite(plot_path, img)
    else:
        show_images_plt(list(d.values()), list(d.keys()))
        plt.savefig(plot_path)

# test
"""
imgs = {
    "tp": [("tmp_img/cat/google00000.jpg","95.2")],
    "tn": [("tmp_img/not_cat/ukbench00000.jpg","95.2")],
    "fp": [("tmp_img/not_cat/ukbench00001.jpg","95.2")],
    "fn": [("tmp_img/cat/google00001.jpg","95.2")]
}
save_conf_samples(imgs, "reporting/conf_trial.png")
"""
