# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import os
import argparse
from imutils import paths
import cv2
import sys

def print_progress(i, slow_factor=0.1):
	sys.stdout.write('\r')
	p = "|/\\"
	p_len = len(p)
	sys.stdout.write("%s\b" % (p[int(i*slow_factor)%p_len]))
	sys.stdout.flush()

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-d", "--dataset", required=True,
	help="path to test images")
ap.add_argument("-w", "--width", default=28, type=int,
	help="width of the resized images")
args = vars(ap.parse_args())

WIDTH = args["width"]

# load the trained convolutional neural network
print("[INFO] loading network...")
model = load_model(args["model"])

print("[INFO] testing images...")
(tp, tn, fp, fn, counter) = (0, 0, 0, 0, 0)
for image_path in paths.list_images(args["dataset"]):
	counter+=1
	# pre-process the image for classification
	image = cv2.imread(image_path)
	image = cv2.resize(image, (WIDTH, WIDTH))
	image = image.astype("float") / 255.0
	image = img_to_array(image)
	# https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.expand_dims.html
	image = np.expand_dims(image, axis=0)
	# classify the input image
	(not_cat, cat) = model.predict(image)[0]
	label = image_path.split(os.path.sep)[-2]
	label = 1 if label == "cat" else 0
	tp += 1 if (cat > not_cat and label == 1) else 0
	tn += 1 if (cat < not_cat and label == 0) else 0
	fp += 1 if (cat > not_cat and label == 0) else 0
	fn += 1 if (cat < not_cat and label == 1) else 0
	print_progress(counter, 0.02)

print("\r[INFO] tp={0} tn={1} fp={2} fn={3}".format(tp, tn, fp, fn))
print("[INFO] total images={}".format(counter))
print("[INFO] accuracy={:.2f}".format((tp+tn)/counter))
