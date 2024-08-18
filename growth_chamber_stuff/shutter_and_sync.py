#!/usr/bin/env python3

############################################################
# Usage: python camerastand.py <NAME>
# Takes a picture with attached camera and saves to Raspberry
# Pi and a remote host
############################################################

import sys
from datetime import datetime 	
import serial	
import subprocess as sp


def takePic(name):
	# get the filepath of the picture
	picName = name
	picDir = "/home/sparkslab/Desktop/local-folder/"
	# modify picPath to a existing directory to store acquired images
	picPath = picDir + picName
	
	#call the camera to save the image
	sp.call(["libcamera-still","-o",picPath])
	print ("Image saved: " + picPath)
	
	#sync the directory, functionally uploading the photo to google drive
	sp.call(["rclone","sync",picDir, "gdrive:rpi-folder"])

#------------------------------------------------
# function that contains the main part of the code that will be run.
# usage: main() 

def main():
	
	print(" ")
	print("taking picture")
	
	picname = sys.argv[1]
	date = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
	name = picname +"_"+ str(date) +".jpg"
	
	# use the takePic function to take a picture at the starting position
	takePic(name)


# Only run this part of the script if it is called directly
# and not used as a module
if __name__ == "__main__":


	# run the main part of the script using the serial
	# port specified above
	main()

