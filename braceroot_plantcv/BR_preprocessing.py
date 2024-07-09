# Isolate brace roots python script
# A preprossessing script for braceroot images

#Import required libraries
import numpy as np
import cv2
import glob
import os

#Custom sobel edge detection algorithm
def edgeDetectSobel(img):
    #better version of Sobel Filter from Chapter 2 slide 64
    kernel = np.array([[-3,0,3],
                       [-10,0,10],
                       [-3,0,3]])/32
    
    edgeX = np.zeros_like(img) #create arrays for edge data
    edgeY = np.zeros_like(img)
    kernelShape = kernel.shape #tuple of kernel dimensions
    imageShape = img.shape     #tuple of image dimensions
    
    #Zero padding below to add 2 units around the border of the image
    paddedDimensions = (imageShape[0]+kernelShape[0]-1,imageShape[1]+kernelShape[1]-1)
    paddedImage = np.zeros(paddedDimensions)
    for i in range(imageShape[0]):
        for j in range(imageShape[1]):
          paddedImage[i+int((kernelShape[0]-1)/2), j+int((kernelShape[1]-1)/2)] = img[i,j]
    
    #running the Filter
    for i in range(imageShape[0]):   #Create moving window
        for j in range(imageShape[1]):
            window = paddedImage[i:i+kernelShape[0],j:j+kernelShape[1]] #window matrix gathers values from image                            
            edgeX[i,j] = np.sum(window*kernel)#window gets multiplied against the kernel 
            edgeY[i,j] = np.sum(window*np.flip(kernel.T,axis=0))
            
    gradient = np.sqrt(np.square(edgeX) + np.square(edgeY))
    gradient *= 255.0 / gradient.max() #using Gradient equation from Chapter 2 Slide 60
    
    return edgeX, edgeY, gradient


#Primary algorithm 
def isolate_brace_roots(img):
    #remove gaussian noise
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gaussian_gray = cv2.GaussianBlur(gray_img,(3,3),0)
    sharp_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    im_denoize = cv2.filter2D(gaussian_gray, -1, sharp_kernel)
    #Detect edges 
    _, _, edges = edgeDetectSobel(im_denoize)
    _, mask = cv2.threshold(edges.astype(np.uint8), 200, 255, cv2.THRESH_BINARY)
    #fill in the plant mask
    fill_kernel = np.ones((15,15)) #removes pepper
    for i in range(3):
        mask = cv2.filter2D(mask, -1, fill_kernel)
        mask = cv2.medianBlur(mask,99)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    #apply plant mask to image
    ret = np.zeros_like(img)
    for i in range(3):
        ret[:,:,i] = img[:,:,i] * (255*mask)
    return ret



#USAGE: For all images in the current directory, Isolate brace roots and save to a new folder


# Get all image files in the current directory
image_files = glob.glob("*.jpg") + glob.glob("*.jpeg") + glob.glob("*.png") + glob.glob("*.gif")

# Make output directory if it doesn't exist
if not os.path.exists("output"):
    os.makedirs("output")

# Process each image
for image_file in image_files:
    # Load the image
    img = cv2.imread(image_file)

    out = isolate_brace_roots(img)

    # Save the resulting images
    cv2.imwrite("output/" + image_file, out)
