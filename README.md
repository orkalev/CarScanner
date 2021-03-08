# CarScanner
CarScanner is an Automatic Number Plate Recognition (ANPR) application for obtaining car details that are listed in the government databases. 
It is a desktop application combining Image processing, Optical Character Recognition (OCR), and REST API, for providing car details of a vehicle out of an image or video.

## Authors 
CarScanner was created in 2021 by Ittai Corem and Or Kalev, 3rd Year Computer Science undergraduate students, as a final project for the course “Introduction to Computational and Biological Vision” – ICBV211.


## Installation
- Clone the repository:

###Mac
```bash
git clone https://github.com/orkalev/CarScanner.git
```




- Select python 3.9 as the interpreter.
- Use the package manager [pip](https://pip.pypa.io/en/stable/) to install all the dependecies of CarScanner.


```bash
pip install <dependency>
```

###Windows
Clonse and install dependencies.
After installing tesseract, add the following line under the import statements
```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
```
## Usage
1) Run the application using PyCharm or your favorite IDE.
2) Import an image or a video to detect a license-plate.
3) Impress your friends with you knowledge of cars.

## Approach and Method
CarScanner uses classic image processing techniques such as filtering, edge detection, color detection, segmentation, contour finding, and more subjects learned throughout the course ICBV211.
To accomplish displaying correct vehicle information from a given input picture/video, CarScanner operates in 3 main stages:
1) License plate detection and recognition (Image Processing).
2) License number detection (OCR).
3) Vehicle database query (HTTP).
Private and commercial vehicles in Israel have a yellow license plate containing only numbers. These insights were used for the first two stages in the process.

### 1. License Plate Detection
At first, a license plate must be detected in the input image.
The License plate detection stage contains 7 sub-stages:
#### 1) RGB -> HSV conversion:
converting BGR image to HSV color-space, is useful to extract a colored object. In HSV, it is easier to represent a color than in RGB color-space. In our application, we extracted yellow-colored objects.
#### 2) Yellow color detection:
Finding yellow areas in the image, one of which should be the license plate.
#### 3) Closing Morphology transformation:
Performing dilation followed by erosion for closing small holes (numbers and other objects) inside the foreground object (license plate).
#### 4) boundaries detection:
Finding connected contours – curves joining all the continuous points (along a boundary). All the redundant points are removed to compresses the contour for saving memory.
#### 5) Rectangle shape search:
Making sure the object detected is a rectangle - the shape of a license plate.
#### 6) Yellow color density threshold:
Verifying the rectangle is yellow enough, as the license plate should be.
#### 7) Cropping:
Cropping the license plate only, preparing for the number detection.


### 2. Number Detection
After detecting the license plate, the plate number is detected using OCR with Tesseract.
The number detection stage contains 2 sub-stages:
#### 1) Pre-processing adjustments using:
###### a. Adaptive thresholding:
Giving better results for images with varying illumination.
b. Contours detection:
Finding the numbers in the license plate and verifying the number of found contours is reasonable.
###### c. Geometric transformation:
Using perspective transformation to increase chances of recognizing the numbers correctly.
###### d. Contrast enhancement:
Increasing contrast for each pixel gray value for achieving better results using Tesseract.
#### 2) Tesseract:
Using Google OCR engine for recognizing the license plate number.
### 3. Vehicle database query for details
The last stage is sending an http request to Data Gov for obtaining the vehicle details according to the vehicle number detected in the last stage, and displaying it to the user.

