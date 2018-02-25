
import cv2
import os
import numpy as np
import sys 
import argparse
from sklearn.svm import SVC
import pickle

sys.path.append('/Users/developer/guru/utility')

from image_processing import basics
from fileOp.conf import Conf
from fileOp.imgReader import ImageReader
from fileOp.h5_dataset import h5_dump_dataset, h5_load_dataset
from annotation.pascal_voc import pacasl_voc_reader
from frameDetect.frameDetect import FrameDetectByOneImage, FrameDetectByDiffImages
from search.search import searchImageByHOGFeature, HOGParam
from classifier.classifier import Classifier
from feature.HOG import HOG

path = os.path.dirname(os.path.abspath(__file__))

def imgDiffRatio(img0, img1):
	diff = cv2.absdiff(img0, img1)
	diff = basics.threshold_img(diff, '50', False)
	nonZeroCnt = cv2.countNonZero(diff)
	return (diff, float(nonZeroCnt)/float(diff.shape[0]*diff.shape[1]))


def main():

	ap = argparse.ArgumentParser()
	ap.add_argument("-v", "--video", required=True, help="Path to the video file")
	ap.add_argument("-m", "--main", required=True, help="json file main menu configuration")
	ap.add_argument("-i", "--item", required=True, help="json file menu item configuration")
	args = vars(ap.parse_args())

	mainMenuConf = Conf(args['main'])
	menuItemConf = Conf(args['item'])
	(mainMenufeatureList, mainMenulabels) = h5_load_dataset(mainMenuConf['feature_file'], mainMenuConf['dataset_feature_name'])

	# read main menu class
	classInfo = []
	mainMenuClassName = None
	if (mainMenuConf['class'] != None):
		for name in open(mainMenuConf['class']).read().split("\n"):
			classInfo.append(name)
	if len(classInfo) != 0:
		mainMenuClassName = classInfo[0]
	else:
		mainMenuClassName = 'mainMenu'

	voc = pacasl_voc_reader(mainMenuConf['dataset_xml'])
	objectList = voc.getObjectList()
	for (className, mainMenuBox) in objectList:
		if (className == mainMenuClassName):
			break

	# read menu item class
	itemClassInfo = []
	if (menuItemConf['class'] != None):
		for name in open(menuItemConf['class']).read().split("\n"):
			itemClassInfo.append(name)

	voc = pacasl_voc_reader(menuItemConf['dataset_xml'])
	objectList = voc.getObjectList()
	for (className, itemBox) in objectList:
		if (className == itemClassInfo[0]):
			break


	itemClassifier = Classifier(menuItemConf['classifier_path'], "SVC")
	itemHOG = HOG(	menuItemConf['orientations'], 
					menuItemConf['pixels_per_cell'], 
					menuItemConf['cells_per_block'], 
					True if menuItemConf['transform_sqrt']==1 else False, 
					menuItemConf['normalize'])

	imgReader = ImageReader(args['video'], True)
	hogParam = HOGParam(orientations=mainMenuConf['orientations'], 
						pixels_per_cell=mainMenuConf['pixels_per_cell'], 
						cells_per_block=mainMenuConf['cells_per_block'], 
						transform_sqrt=True if mainMenuConf['transform_sqrt']==1 else False, 
						block_norm=mainMenuConf['normalize'])
	mainMenuLoc = None
	mainMenuImg = None
	bFound = False
	frameCnt = 0
	searchRegion = None
	while True: 
		(ret, frame, fname) = imgReader.read()
		if ret == False:
			break
		templateShape = [mainMenuBox[3]-mainMenuBox[1]+1, mainMenuBox[2]-mainMenuBox[0]+1]
		frameOrigin = frame.copy()
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		key = ' '
		if (bFound == True):

			testImg = frame[mainMenuLoc[1]:mainMenuLoc[3], mainMenuLoc[0]:mainMenuLoc[2]]
			e1 = cv2.getTickCount()
			(diff, ratio) = imgDiffRatio(testImg, mainMenuImg)
			e2 = cv2.getTickCount()
			time = (e2 - e1)/ cv2.getTickFrequency()
			print('[{}] ratio {}'.format(frameCnt, ratio))
			if (ratio < 0.1):
				bFound = True
				(x, y, w, h) = (mainMenuLoc[0], mainMenuLoc[1], mainMenuLoc[2]-mainMenuLoc[0], mainMenuLoc[3]-mainMenuLoc[1])
			else:
				bFound = False
		else:
			if searchRegion == None:
				searchRegion = tuple(mainMenuConf['mainMenuSearchRegion'])
			e1 = cv2.getTickCount()
			(bFound, val, (x, y, w, h)) = searchImageByHOGFeature(mainMenufeatureList[0], 
									templateShape,
									frame, 
									searchRegion, 
									mainMenuConf['mainMenuHOGDistanceThreshold'], 
									hogParam, 
									(10, 10), 
									bVisualize=False)
			e2 = cv2.getTickCount()
			time = (e2 - e1)/ cv2.getTickFrequency()
			if bFound == True:
				frameDetectImg = frame[y:y+h, x:x+w]

		if bFound == True:
			print('[{}] search result time {}, val = {}, loc = {}'.format(frameCnt, time, val, (x, y, w, h)))
			searchRegion = (x, y, x+w, y+h)
			mainMenuLoc = (x, y, x+w, y+h)
			mainMenuImg = frame[y:y+h, x:x+w]
			frameDetectImg = mainMenuImg
			cv2.rectangle(frameOrigin, (x, y), (x+w-1, y+h-1), (0, 255, 0), 2)

			e1 = cv2.getTickCount()
			(rtn, (fx, fy, fw, fh)) = FrameDetectByOneImage(frameDetectImg, 
															frameDetectImg, 
															minW=200, minH=60, 
															frameRatio=mainMenuConf['mainMenuFrameRectRatio'])
			e2 = cv2.getTickCount()
			time = (e2 - e1)/ cv2.getTickFrequency()
			if rtn==True:
				fx = fx + x
				fy = fy + y
				print('frame detected {}, takes {}'.format((fx, fy, fw, fh), time))
				cv2.rectangle(frameOrigin, (fx, fy), (fx+fw, fy+fh), (0, 0, 255), 2)
				bh = itemBox[3]-itemBox[1]+1
				bw = itemBox[2]-itemBox[0]+1
				roi = frame[fy:fy+bh, fx:fx+bw]
				e1 = cv2.getTickCount()
				(feature, _) = itemHOG.describe(roi)
				predictIdx = itemClassifier.predict(feature)
				e2 = cv2.getTickCount()
				time = (e2 - e1)/ cv2.getTickFrequency()
				print('    predict {} takes {}'.format(predictIdx, time))
				cv2.putText(frameOrigin, str(predictIdx), (fx+fw+10, fy+fh), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,0,0),3,cv2.LINE_AA)

			key = basics.showResizeImg(frameOrigin, 'result', 1)
		else:
			print('[{}] Not found, takes  {}'.format(frameCnt, time))
			key = basics.showResizeImg(frameOrigin, 'result', 1)
		
		if key == ord('q'):
			break
		frameCnt = frameCnt + 1

main()