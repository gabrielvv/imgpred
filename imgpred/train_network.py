# set the matplotlib backend so figures can be saved in the background
import matplotlib
# @see https://matplotlib.org/glossary/index.html#term-agg
matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from pyimagesearch.lenet import LeNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-w", "--width", default=28, type=int,
	help="width of the resized images")
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# @see https://keras.io/getting-started/faq/#what-does-sample-batch-epoch-mean
# initialize the number of epochs to train for, initial learning rate,
# and batch size
EPOCHS = 25 # training epochs
INIT_LR = 1e-3 # initial learning rate
BS = 32 # batch size
WIDTH = args["width"]

# init the data and labels
print("[INFO] loading images...")
data=[]
labels=[]

image_paths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(image_paths)

for image_path in image_paths:
    img = cv2.imread(image_path)
    img = cv2.resize(img, (WIDTH, WIDTH))
    img = img_to_array(img)
    data.append(img)

    # os.path.sep => '\\' on Windows
    label = image_path.split(os.path.sep)[-2]
    label = 1 if label == "cat" else 0
    labels.append(label)

# scale the raw pixel intensities to the range [0,1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# convert the labels from integers to vectors
# [0,1,0] => [[1,0], [0,1], [1,0]]
trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)

# construct the image generator for data augmentation
# @see https://keras.io/preprocessing/image/
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
horizontal_flip=True, fill_mode="nearest")

# init the model
print("[INFO] compiling model...")
model = LeNet.build(width=WIDTH, height=WIDTH, depth=3, classes=2)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS), validation_data=(testX, testY),
steps_per_epoch=len(trainX) // BS, epochs=EPOCHS, verbose=1)

# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"])

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Cat/Not Cat")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
