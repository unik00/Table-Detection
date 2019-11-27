import cv2 as cv
import numpy as np
import copy


def find_table_features(image):
	scale = 50
	im = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
	im = cv.adaptiveThreshold(im,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
	            cv.THRESH_BINARY,11,2)
	im = ~im
	verticalsize = im.shape[1] // scale
	horizontalsize = im.shape[0] // scale

	kernel = \
		[cv.getStructuringElement(cv.MORPH_RECT,(1, verticalsize)),
		cv.getStructuringElement(cv.MORPH_RECT,(horizontalsize,1))
		]

	pos_ims = [im, im]

	for i in [0,1]:
		pos_ims[i] = cv.erode(pos_ims[i], kernel[i], iterations=1)
		pos_ims[i] = cv.dilate(pos_ims[i], kernel[i], iterations=1)
		refine = max(2, im.shape[1] // 300)
		print(refine)
		refine = (refine, refine)
		pos_ims[i] = cv.dilate(pos_ims[i], np.ones(refine,np.uint8), iterations=1)

	return pos_ims[0] + pos_ims[1]

if __name__ == "__main__":
	imname = 'test'
	im = cv.imread('test_images/'+imname+'.jpg')

	im_bak = copy.deepcopy(im)

	# im = cv.resize(im, (im.shape[1]//4,im.shape[0]//4))
	# cv.imshow('before', im)

	im = find_table_features(im)
	cv.imshow('after', im)
	cv.imwrite(imname+'_feature.jpg', im)
	
	contours, hierarchy = cv.findContours(im, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
	im = cv.cvtColor(im, cv.COLOR_GRAY2BGR)
	# im = np.zeros(im.shape)

	table = contours[0]
	for i, contour in enumerate(contours):
		if cv.contourArea(table) < cv.contourArea(contour):
			table = contour

	rect = cv.minAreaRect(table)
	box = cv.boxPoints(rect)
	box = np.int0(box)

	im = im_bak
	cv.drawContours(im, [box], 0, (0,0,255), 2)
	cv.imshow('after', im)
	cv.imwrite(imname+'_result.jpg', im)

	cv.waitKey()