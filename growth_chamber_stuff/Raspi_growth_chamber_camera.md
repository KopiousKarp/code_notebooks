## A guide on how to setup a raspberry pi for growth chamber monitoring
This will go from setting up a clean raspberry pi with a camera module to take images, sync with google drive, and then use plantCV on google Colab to analyze those images. 
Everything should be self contained in a folder with this README. Here is a table of contents:
- [Step 1: Make sure the camera works](#step-1-make-sure-the-camera-works)
- [Step 2: Install & configure Rclone](#step-2-install--configure-rclone)
- [Step 3: Set up the automation](#step-3-set-up-the-automation)

The software we will need to use on the RPI:
- Rclone (to sync with google drive)
- libcamera-still (built-in camera operating software to take still images)
- python to name and place the images in the proper place and format

### Step 1: Make sure the camera works
with the RPI up and running and the camera module plugged in. You should be able to run:
```bash
libcamera-still -o test.jpg
```
`-o` specifies the output file.    
What should happen is that a preview window opens up, there will be a small delay, and then an image should be saved called `test.jpg` in your present working directory

### Step 2: Install & configure Rclone
run this command in a terminal to install Rclone     
This software will be used to sync up the Pi to google drive
```bash
sudo -v ; curl https://rclone.org/install.sh | sudo bash
```
After that install is finished.
Configure Rclone to use your google drive using `rclone config` and follow the steps in the configuration for a google drive remote. Leave all of the options blank, we can get away with default settings.     
At the end of the configuration process. Rclone will ask you if you have a device that can run a browser. The RPI doesn't quite have the resources to use a browser. So answer "no" and then follow the steps to complete the authorization on another computer.    
Test that it is working by running
`rclone sync localfolder gdrive:rpi-folder`
a folder should show up in your google drive that is an exact copy of the local folder on the RPI. 


NOTE: The `sync` command does not "mirror". The destination will always be made to match the source. So don't put files in there on google drive, they will be deleted the next time the Pi syncs up.  

### Step 3: Set up the automation
There is a crontab file in this folder. Cron is a system automation service that will reliably run when the Pi is operational.
##### photo-schedule.crontab
```bash
# Standard Raspberry Pi 18-hour day photography schedule 
# For the imaging hours 6am to midnight take images every 15 minutes 
0,15,30,45 6-23 * * * /usr/bin/python /home/sparkslab/Desktop/local-folder/shutter_and_sync.py topview
```
Translation: Upon the 0th, 15th, 30th, and 45th minute of every hour from 6:00 through 23:00, the service will run a python script at the designated location. 
NOTE: This will not capture an image at midnight, when the lights are set to turn off. 

This crontab can be installed by running the command `crontab photo-schedule.crontab`

editing this cron job:     
If you want to change the frequency, add more points to the minute section of this crontab before installing it
`0,5,10,15,20,25,30,35,40,45,50,55 6-23 * * * /usr/bin/python /home/sparkslab/Desktop/local-folder/shutter_and_sync.py topview`

If your python install isn't the standard one at `/usr/bin/python` then change that part of the code. you must use absolute paths because cron runs as the root user. So no `~` here. You can find out the absolute path for python by typing `which python` into the terminal. 

Make sure your python script location is also in the correct place. Keep in mind that the username `sparkslab` is not the standard on that most RPi's use. So your RPi might have a different username like `pi`, which would change the `/home/sparkslab` to `/home/pi`

Lastly, The files will be named with a prefix passed in through this cronjob so `topview` can be changed to whatever you want your prefix to be. Example, say I want my next Pi camera to have the sideview label. 

`0,15,30,45 6-23 * * * /usr/bin/python /home/sparkslab/Desktop/local-folder/shutter_and_sync.py sideview`     

Now the images would be named "sideview_2024-08-06_13:15:01.jpg"

##### shutter_and_sync.py
```python
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

```

This is a very simple python script that will name your photo with the format specified by:      
- The argument passed to the script, in this case "topview" is passed in like this: `picname = sys.argv[1]`
- Date and time following the format specified this way: `date = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')`
- The directory that the Pi saving to and syncing google drive with is specified like this: `picDir = "/home/sparkslab/Desktop/local-folder/"`
- And the Rclone command is called like this: `sp.call(["rclone","sync",picDir, "gdrive:rpi-folder"])`
    - depending on what you want the folder to be named on google drive and what you named the remote connection to google drive when configuring Rclone, you will need to change this command to match. specifically the `"gdrive:rpi-folder"`. 



### Final Checklist
After you have:     
[] Confirmed that the camera is working with `libcamera-still -o test.jpg`     
[] Confirmed that Rclone is working with `rclone sync /home/sparkslab/Desktop/local-folder gdrive:rpi-folder`   
[] Confirmed that the python script works with `python shutter_and_sync.py test`   
[] Confirmed that you crontab is correct and run `crontab photo-schedule.crontab`      

This Pi will be opperation as a growth chamber monitor. So long as:      
[] It has a connection to the internet to sync up, photos will keep being taken if this drops out     
[] It has power, I can't build in falure tolerance to this. Sorry.     
[] The camera is operational      

If part of this is failing. I haven't confirmed this, but you should be able to run `sudo grep 'CRON' /var/log/syslog | tail -n 10` which should show you the last ten log entries for cron and show you which part of the python script is failing. 





