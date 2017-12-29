# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-w", "--width", default=28, type=int,
	help="width of the resized images")
args = vars(ap.parse_args())

WIDTH = args["width"]

# load the image
image = cv2.imread(args["image"])
orig = image.copy()

# pre-process the image for classification
image = cv2.resize(image, (WIDTH, WIDTH))
image = image.astype("float") / 255.0
image = img_to_array(image)
# https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.expand_dims.html
image = np.expand_dims(image, axis=0)

# load the trained convolutional neural network
print("[INFO] loading network...")
model = load_model(args["model"])

# classify the input image
(not_cat, cat) = model.predict(image)[0]

# build the label
label = "Cat" if cat > not_cat else "Not Cat"
proba = cat if cat > not_cat else not_cat
label = "{}: {:.2f}%".format(label, proba * 100)

# draw the label on the image
output = imutils.resize(orig, width=400)

# https://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html#puttext
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 255, 0), 2)

# show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)
