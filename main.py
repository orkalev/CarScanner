import cv2
import pytesseract
import tkinter
from tkinter import *
import requests
import tkinter.filedialog as tkFileDialog
import numpy as np
from tkinter import font
from random import choice


def filter_contrast(image):
    contrastPrsent = 10
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayscaleHistogram = cv2.calcHist([grayImage], [0], None, [256], [0, 256]) # Get the grayscale histogram
    acc = []
    acc.append(float(grayscaleHistogram[0]))
    for index in range(1, len(grayscaleHistogram)):
        acc.append(acc[index - 1] + float(grayscaleHistogram[index]))

    contrastPrsent *= (acc[-1] / 100.0)
    contrastPrsent /= 2.0

    minGray = 0
    while acc[minGray] < contrastPrsent:
        minGray += 1

    maxGray = len(grayscaleHistogram) - 1
    while acc[maxGray] >= (acc[-1] - contrastPrsent):
        maxGray -= 1

    return cv2.convertScaleAbs(image, alpha=255 / (maxGray - minGray), beta=-minGray * 255 / (maxGray - minGray))


def detect_plate(originalImage):
    licenseNum = 'none'
    imageHeight, imageWidth, c = originalImage.shape
    copyImage = originalImage.copy()       # Copy Image
    hsvColorImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2HSV) # RGB -> HSV (for yellow sepration )
    yellowImage = cv2.inRange(hsvColorImage, np.array([17, 90, 90]), np.array([30, 255, 255]))      # get all range from low to high
    yellowGrayImage = cv2.bitwise_and(yellowImage, yellowImage, mask=yellowImage)   # bit wise and to transporm to gray image
    k = np.ones((5, 5), np.uint8)      #Creat structer element

    # Double closing to the image
    closingMorpho = cv2.morphologyEx(yellowGrayImage, cv2.MORPH_CLOSE, k)   # Fill litel holes using morphology close opration
    closingMorpho = cv2.morphologyEx(closingMorpho, cv2.MORPH_CLOSE, k)

    # Detected yellow area
    contours, her = cv2.findContours(closingMorpho, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)    # contours aka claster
    # print(contours)


    # Loop over contours and find license plates
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)     # find the rect of the shape

        # Conditions on crops dimensions and area
        if h * 6 > w > 2 * h and h > 0.1 * w and w * h > imageHeight * imageWidth * 0.0001:        #check the size of the rect
            cropImage = originalImage[y:y + h, x - round(w / 10):x]    # crop the plant
            cropImage = cropImage.astype('uint8')

            # Compute yellow color density in the crop
            # Make a crop from the RGB image
            imgray = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
            cropImageYellow = originalImage[y:y + h, x:x + w]
            cropImageYellow = cropImageYellow.astype('uint8')
            # Detect yellow color
            hsvColorImage = cv2.cvtColor(cropImageYellow, cv2.COLOR_BGR2HSV)
            yellowImage = cv2.inRange(hsvColorImage, np.array([20, 100, 100]), np.array([30, 255, 255]))

            # Condition on yellow color density in the crop
            if yellowImage.sum() > 255 * cropImage.shape[0] * cropImage.shape[0] * 0.4:

                # Make a crop from the gray image
                corpImageGray = imgray[y:y + h, x:x + w]
                corpImageGray = corpImageGray.astype('uint8')

                # At this point we know that the crop image is the yellow plant


                # Detect chars inside yellow crop with specefic dimension and area
                th = cv2.adaptiveThreshold(corpImageGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                           11, 2)   # make a mask(black and white) img
                                                    # from the croped yellow plate
                contours2, her = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                #find contoures like before
                #then run for each contour in contours and try to match bounding box for each letter

                # Init number of chars
                chars = 0
                for c in contours2:
                    area2 = cv2.contourArea(c)
                    x2, y2, w2, h2 = cv2.boundingRect(c)
                    if w2 * h2 > h * w * 0.01 and h2 > w2 and area2 < h * w * 0.9:
                        chars += 1

                # Condition on the number of chars
                if 20 > chars > 4:
                    box = np.int0(cv2.boxPoints(cv2.minAreaRect(contour)))
                    pts = np.array(box)
                    # Order the rect corners
                    rect = np.zeros((4, 2), dtype="float32")
                    s = pts.sum(axis=1)
                    rect[0] = pts[np.argmin(s)]
                    rect[2] = pts[np.argmax(s)]
                    diff = np.diff(pts, axis=1)
                    rect[1] = pts[np.argmin(diff)]
                    rect[3] = pts[np.argmax(diff)]

                    # Transform the points
                    (tl, tr, br, bl) = rect
                    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
                    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
                    maxWidth = max(int(widthA), int(widthB))
                    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
                    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
                    maxHeight = max(int(heightA), int(heightB))
                    dst = np.array([
                        [0, 0],
                        [maxWidth - 1, 0],
                        [maxWidth - 1, maxHeight - 1],
                        [0, maxHeight - 1]], dtype="float32")
                    M = cv2.getPerspectiveTransform(rect, dst)
                    adjusted = cv2.warpPerspective(copyImage, M, (maxWidth, maxHeight))

                    adjusted = filter_contrast(adjusted)
                    plate = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
                    licenseNum = pytesseract.image_to_string(plate, config='--psm 13 -c tessedit_char_whitelist=0123456789')
                    licenseNum = licenseNum[:-2]
                    # Put the license number on the photo
                    originalImage = cv2.putText(originalImage, licenseNum , (x - 100, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                          (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.drawContours(originalImage, [box], 0, (0, 0, 255), 2)


    return originalImage, licenseNum

def importVideo():
    path = tkFileDialog.askopenfilename()
    video = cv2.VideoCapture(path)

    if video.isOpened() == False:
        print("Error opening video")

    while (video.isOpened()):
        ret, frame = video.read()
        if ret == True:
            originalImage, licenseNum = detect_plate(frame)
            print(licenseNum)
            cv2.putText(originalImage, 'Press \'Q\' to exit !',(50, 50),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 255), 2)
            cv2.imshow('Frame', originalImage)
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        # Break the loop
        else:
            break
    # When everything done, release the video capture object
    video.release()
    cv2.destroyAllWindows()


def importImage():
    # open a file chooser dialog and allow the user to select an input image
    canvas1.delete("CarInfo")
    path = tkFileDialog.askopenfilename()
    if len(path) > 0:
        # Read the image file
        image = cv2.imread(path)
        cv2.imshow("The car before detection", image)
        cv2.waitKey(0)
        cv2.destroyWindow("The car before detection")

        detection, licenseNum = detect_plate(image)
        print(licenseNum)

        cv2.imshow("The car after detection", detection)
        cv2.waitKey(0)
        cv2.destroyWindow("The car after detection")

        # Get the data from API source
        payload = {'resource_id': '053cea08-09bc-40ec-8f7a-156f0677aff3', 'q': licenseNum}
        r = requests.get('https://data.gov.il/api/3/action/datastore_search', params=payload)
        res = r.json()
        record1 = res['result']['records']
        if len(record1) == 0:
            canvas1.create_text(350, 230, text='The car is not at the data base' ,fill="white",font=('Andale Mono', 20), anchor="w", tag="CarInfo")
        else:
            record = record1[0]
            mispar_rechev = record["mispar_rechev"]
            tozeret_cd = record["tozeret_cd"]
            tozeret_nm = record["tozeret_nm"]
            degem_nm = record["degem_nm"]
            ramat_gimur = record["ramat_gimur"]
            ramat_eivzur_betihuty = record["ramat_eivzur_betihuty"]
            kvutzat_zihum = record["kvutzat_zihum"]
            shnat_yitzur = record["shnat_yitzur"]
            degem_manoa = record["degem_manoa"]
            mivchan_acharon_dt = record["mivchan_acharon_dt"]
            tokef_dt = record["tokef_dt"]
            baalut = record["baalut"]
            misgeret = record["misgeret"]
            tzeva_rechev = record["tzeva_rechev"]
            zmig_kidmi = record["zmig_kidmi"]
            zmig_ahori = record["zmig_ahori"]
            sug_delek_nm = record["sug_delek_nm"]
            horaat_rishum = record["horaat_rishum"]
            kinuy_mishari = record["kinuy_mishari"]
            canvas1.create_text(350, 200, text=str(mispar_rechev),fill="white",font=('Andale Mono', 15), anchor="w", tag="CarInfo")
            canvas1.create_text(350, 230, text=tozeret_nm,fill="white",font=('Andale Mono', 15), anchor="w", tag="CarInfo")
            canvas1.create_text(350, 260, text=ramat_gimur,fill="white",font=('Andale Mono', 15), anchor="w", tag="CarInfo")
            canvas1.create_text(350, 290, text=ramat_eivzur_betihuty,fill="white",font=('Andale Mono', 15), anchor="w", tag="CarInfo")
            canvas1.create_text(350, 320, text=shnat_yitzur,fill="white",font=('Andale Mono', 15), anchor="w", tag="CarInfo")
            canvas1.create_text(350, 350, text=mivchan_acharon_dt,fill="white",font=('Andale Mono', 15), anchor="w", tag="CarInfo")
            canvas1.create_text(350, 380, text=tokef_dt, anchor="w",fill="white",font=('Andale Mono', 15), tag="CarInfo")
            canvas1.create_text(350, 410, text=baalut, anchor="w",fill="white",font=('Andale Mono', 15), tag="CarInfo")
            canvas1.create_text(350, 440, text=misgeret, anchor="w",fill="white",font=('Andale Mono', 15), tag="CarInfo")
            canvas1.create_text(350, 470, text=tzeva_rechev, anchor="w",fill="white",font=('Andale Mono', 15), tag="CarInfo")
            canvas1.create_text(350, 500, text=sug_delek_nm, anchor="w",fill="white",font=('Andale Mono', 15), tag="CarInfo")
            canvas1.create_text(350, 530, text=kinuy_mishari, anchor="w",fill="white",font=('Andale Mono', 15), tag="CarInfo")

            canvas1.update()
        return




def exitUI():
    exit(0)

# Create object
root = Tk()

# Adjust size
root.geometry("1280x740")

root.title('Car Scanner')

# Add image file
# bg = PhotoImage(file="test2.png")
bg = PhotoImage(file="unnamed.png")
# Create Canvas
canvas1 = Canvas(root, width=510,
                 height=400)

canvas1.pack(fill="both", expand=True)

# Display image
canvas1.create_image(0, 0, image=bg,
                     anchor="nw")

# Add Text
# canvas1.create_text(320, 30, text="Welcome to the car scanner",fill="white", font=('c', 18, 'bold'))
font.families()
# Add the field text
canvas1.create_text(50, 200, text="Car Number:",fill="white",font=('Andale Mono', 15), anchor="w")
canvas1.create_text(50, 230, text="Manufacturer country:",fill="white",font=('Andale Mono', 15), anchor="w")
canvas1.create_text(50, 260, text="Level:",fill="white",font=('Andale Mono', 15), anchor="w")
canvas1.create_text(50, 290, text="Fitting safety level:",fill="white",font=('Andale Mono', 15), anchor="w")
canvas1.create_text(50, 320, text="Production year:",fill="white",font=('Andale Mono', 15), anchor="w")
canvas1.create_text(50, 350, text="Last vehicle licensing test:",fill="white",font=('Andale Mono', 15), anchor="w")
canvas1.create_text(50, 380, text="Next vehicle licensing test:",fill="white",font=('Andale Mono', 15), anchor="w")
canvas1.create_text(50, 410, text="Current ownership:",fill="white",font=('Andale Mono', 15), anchor="w")
canvas1.create_text(50, 440, text="Car build number:",fill="white",font=('Andale Mono', 15), anchor="w")
canvas1.create_text(50, 470, text="Color:",fill="white",font=('Andale Mono', 15), anchor="w")
canvas1.create_text(50, 500, text="Fuel type:",fill="white",font=('Andale Mono', 15), anchor="w")
canvas1.create_text(50, 530, text="Trade alias:",fill="white",font=('Andale Mono', 15), anchor="w")

# Create Buttonsv
importImagePhoto = PhotoImage(file = "importImage.png")
importVideoPhoto = PhotoImage(file = "ImportVideo.png")
exitPhoto = PhotoImage(file = "Exit.png")

importImageButton = tkinter.Button(root, image = importImagePhoto ,command=importImage,height=40, width=157)
importVideoButton = tkinter.Button(root, image = importVideoPhoto, command=importVideo, height=35, width=149)
exitButton = tkinter.Button(root, image = exitPhoto, command=exitUI, height=41, width=79)

# Display Buttons
importImageButton_canvas = canvas1.create_window(48, 105,
                                                 anchor="nw",
                                                 window=importImageButton)
importImageButton_canvas
importVideoButton_canvas = canvas1.create_window(215, 105,
                                                 anchor="nw",
                                                 window=importVideoButton)
importImageButton
exitButton_canvas = canvas1.create_window(375, 106, anchor="nw",
                                          window=exitButton)
# Execute tkinter
root.mainloop()
