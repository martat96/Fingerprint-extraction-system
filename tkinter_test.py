#!/usr/bin/env python
# -*- coding: utf-8 -*-

from Tkinter import *
from PIL import Image
from PIL import ImageTk
import tkFileDialog
import cv2
import pylab
import numpy as np
from numpy.core._multiarray_umath import ndarray


def select_image():

	global panelA, panelB, sgm

	path = tkFileDialog.askopenfilename()
	try:
		if len(path) > 0:
			# Reading the image file
			image = cv2.imread(path)
			img = cv2.imread(path, 0)
			img = cv2.resize(img, (200, 350))
			img1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3)

			# Skeletonization
			size = np.size(img1)
			print (size)
			skel = np.zeros(img1.shape, np.uint8)
			element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
			done = False
			while (not done):
				erosion = cv2.erode(img1, element)
				temp = cv2.dilate(erosion, element)
				temp = cv2.subtract(img1, temp)
				skel = cv2.bitwise_or(skel, temp)
				img1 = erosion.copy()
				zeros = size - cv2.countNonZero(img1)
				if zeros == size:
					done = True

			# Saving type of minutiae
			inv = np.invert(skel)
			pts = []
			for (x, y), value in np.ndenumerate(inv):
				if value == 0:
					if x == inv.shape[0] - 1 or y == inv.shape[1] - 1:
						continue
					if (inv[x - 1][y - 1] / 255) + (inv[x + 1][y - 1] / 255) + (inv[x][y - 2] / 255) + (
							inv[x][y] / 255) + (inv[x - 1][y - 2] / 255) + (inv[x + 1][y - 2] / 255) + (
							inv[x + 1][y] / 255) + (inv[x - 1][y] / 255) == 7:
						print 'RIDGE ENDING OF MINUTIAE - COORDINATES : ', (x, y)
						pts.append(("Ridge ending", (x, y)))
					elif (inv[x - 1][y - 1] / 255) + (inv[x + 1][y - 1] / 255) + (inv[x][y - 2] / 255) + (
							inv[x][y] / 255) + (inv[x - 1][y - 2] / 255) + (inv[x + 1][y - 2] / 255) + (
							inv[x + 1][y] / 255) + (inv[x - 1][y] / 255) == 5:
						print 'RIDGE BIFURICATION OF MINUTIAE - COORDINATES : ', (x, y)
						pts.append(("Ridge bifurcation", (x, y)))

			# Extraction of minutiae

			# Harris corner
			harris_corners = cv2.cornerHarris(skel, 2, 3, 0.1)
			dst = cv2.dilate(harris_corners, None)
			harris_normalized = cv2.normalize(dst, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)

			# Extraction of keypoints
			threshold_harris = 125
			keypoints = []

			for x in range(0, harris_normalized.shape[0]):
				for y in range(0, harris_normalized.shape[1]):
					if harris_normalized[x][y] > threshold_harris:
						keypoints.append(cv2.KeyPoint(y, x, 1))
			orb = cv2.ORB_create()

			# Descriptors computation
			kp, des = orb.compute(img, keypoints)
			key = cv2.drawKeypoints(skel, kp, outImage=None)
			sgm = cv2.resize(key, (200, 350))
			pts = np.array([kp[idx].pt for idx in range(len(kp))], dtype=np.int8).reshape(-1, 1, 2)

			image = cv2.resize(image, (200, 350))
			image = Image.fromarray(image)
			sgm = Image.fromarray(sgm)

			image = ImageTk.PhotoImage(image)
			sgm = ImageTk.PhotoImage(sgm)

			if panelA is None or panelB is None:
				panelA = Label(image=image)
				panelA.image = image
				panelA.pack(side="left", padx=10, pady=10)

				panelB = Label(image=sgm)
				panelB.image = sgm
				panelB.pack(side="right", padx=10, pady=10)

				downloadImageBtn.configure(state=NORMAL)

			else:
				panelA.configure(image=image)
				panelB.configure(image=sgm)
				panelA.image = image
				panelB.image = sgm

				downloadImageBtn.configure(state=NORMAL)

	except IOError:
		pass

def save_image():
	savePath = tkFileDialog.asksaveasfilename() + '.jpg'
	sgm._PhotoImage__photo.write(savePath)


## ADDITIONAL FUNCTIONS ##

# def bin():
# 	cv2.imshow("BINARIZATION", img1)
# def skeletization():
# 	cv2.imshow("SKELETIZATION", skel)
# def harris():
# 	cv2.imshow("HARRIS CORNER DETECTOR", harris_normalized)
# def wsp():
# 	print pts


root = Tk()
panelA = None
panelB = None

topLabel = Label(root,
				 text="Fingerprint extraction system",
				 bg="gray")
topLabel.pack(side="top", fill="both", expand="yes", padx="10", pady="10")

opis1 = Label(root, text="Load the image from the file by pressing the 'Load image' button, then you will see the result of extracting characteristic features ")
opis1.pack(side="top", fill="both", expand="yes", padx="10", pady="10")

funkcje = Label(root, text="Available functions:")
funkcje.pack(side="top", fill="both", expand="yes", padx="10", pady="10")

wczytajBtn = Button(root, text="Load an image from file", command=select_image)
wczytajBtn.pack(side="top", fill="both", expand="yes", padx="10", pady="10")


downloadImageBtn = Button(root, text="Save result on computer", state=DISABLED, command=save_image)
downloadImageBtn.pack(side="top", fill="both", expand="yes", padx="10", pady="10")

## ADDITIONAL BUTTONS ##

# pobierzWspBtn = Button(root, text="Pobierz współrzędne minucji do pliku tekstowego", state=DISABLED, command=wsp)
# pobierzWspBtn.pack(side="top", fill="both", expand="yes", padx="10", pady="10")
#
# binaryzacjaBtn = Button(root, text="Zobacz wynik binaryzacji", state=DISABLED, command=bin)
# binaryzacjaBtn.pack(side="top", fill="both", expand="yes", padx="10", pady="10")
#
# szkieletyzacjaBtn = Button(root, text="Zobacz wynik szkieletyzacji", state=DISABLED, command=skeletization)
# szkieletyzacjaBtn.pack(side="top", fill="both", expand="yes", padx="10", pady="10")
#
# harrisBtn = Button(root, text="Zobacz wynik detektora krawędzi Harrisa", state=DISABLED, command=harris)
# harrisBtn.pack(side="top", fill="both", expand="yes", padx="10", pady="10")

root.mainloop()