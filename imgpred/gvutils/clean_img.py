import os
import cv2
from imutils import paths

def clean_img(p):
	for image_path in paths.list_images(p):
		# initialize if the image should be deleted or not
		delete = False

		# try to load the image
		try:
			image = cv2.imread(image_path)

			# if the image is `None` then we could not properly load it
			# from disk, so delete it
			if image is None:
				delete = True

		# if OpenCV cannot load the image then the image is likely
		# corrupt so we should delete it
		except:
			print("Unexpected error: {}".format(sys.exc_info()[0]))
			delete = True

		# check to see if the image should be deleted
		if delete:
			print("[INFO] deleting {}".format(image_path))
			os.remove(image_path)
