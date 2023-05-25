import cv2
import pytesseract
import numpy as np
from matplotlib import pyplot as plt
from imutils.object_detection import non_max_suppression
import glob
import time
import concurrent.futures
import argparse
import statistics



def measure_staffgauge_len(image_path):
    
    img = cv2.imread(image_path)
    cv2.imshow("Staffgauge Original Image", img)
    cv2.waitKey(0)
    
    # Crop for only area of staffgauge, ROI should be from the YOLOv4 result
    h = img.shape[0]
    w = img.shape[1]
    print(w, h)
    if image_path == "../images/staffgauge-1.png":
        img_crop = img[int(h*0.05):int(h*1), int(w*0.25):int(w*0.65)]
    else:
        img_crop = img
    cv2.imshow("Cropped Image", img_crop)
    cv2.waitKey(0)






    # convert to hsv colorspace
    hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)

    # lower bound and upper bound for Yellow color (Daylight)
    #lower_color = np.array([20, 90, 90])
    #upper_color = np.array([30, 255, 255])
    
    # lower bound and upper bound for Yellow color (Night with LED lighting, Kinda white)
    lower_color = np.array([0, 0, 180])
    upper_color = np.array([30, 255, 255])

    # find the colors within the boundaries
    mask = cv2.inRange(hsv, lower_color, upper_color)
    
    # bitwise with mask
    img_segmented = cv2.bitwise_and(img_crop, img_crop, mask=mask)
    cv2.imshow("Seg Image", img_segmented)
    cv2.waitKey(0)

    # convert the input image to grayscale
    gray = cv2.cvtColor(img_segmented, cv2.COLOR_BGR2GRAY)
    # apply thresholding to convert grayscale to binary image
    ret, thresh = cv2.threshold(gray, 70, 255, 0)
    cv2.imshow("Gray Image", gray)
    cv2.imshow("Binary Image", thresh)
    cv2.waitKey(0)

    # apply hole filling to binary image
    kernel_size = (7, 7)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("Filling Image", closing)
    cv2.waitKey(0)

    # Find the water line
    # Use sum of grey-value in row of gray-image
    h_gray = closing.shape[0]
    w_gray = closing.shape[1]
    sum_gray = []
    for row in range(h_gray):
        tmp = 0
        for col in range(w_gray):
            tmp = tmp + closing[row, col]
        sum_gray.append(tmp)
    mean = statistics.mean(sum_gray)
    std  = statistics.stdev(sum_gray)
    print("mean, std = {}, {}".format(mean, std))
    # Filter the sum_gray waveform
    sum_gray_filter = []
    yd = 0
    y  = 0
    x  = 0
    a  = 0.12
    waterline_row = []
    threshold = (mean - std)*0.4
    edge_state=0
    for row in range(h_gray):
        x = sum_gray[row]
        y = a*x + (1 - a)*yd
        yd = y
        sum_gray_filter.append(y)
        # Thresholding to find the staffgauge line
        if edge_state == 0:
            if y > threshold:
                waterline_row.append(row)
                edge_state=1
        elif edge_state == 1:
            if y < threshold:
                waterline_row.append(row)
                edge_state=2

    print("waterline row=")
    print(waterline_row)
    plt.title("Line graph")
    plt.plot(sum_gray, color="red")
    plt.plot(sum_gray_filter, color="blue")
    plt.show()

    # Draw the water-line into staffgauge image
    # Using cv2.line() method
    # Draw a diagonal green line with thickness of 9 px
    top = 0
    bottom = h_gray
    if len(waterline_row) > 0 and len(waterline_row) <= 2:
        # Top line
        top = waterline_row[0]
        start_point = (0, waterline_row[0])
        end_point = (w_gray-1, waterline_row[0])
        color = (0, 255, 0)
        thickness=3
        img_waterline = cv2.line(img_crop, start_point, end_point, color, thickness)
        
        if len(waterline_row) == 2:
            # Bottom line (Waterline)
            bottom = waterline_row[1]
            start_point = (0, waterline_row[1])
            end_point = (w_gray-1, waterline_row[1])
            color = (0, 255, 0)
            thickness=3
            img_waterline = cv2.line(img_waterline, start_point, end_point, color, thickness)
    staffgauge_len = bottom - top

    # Displaying the image
    print("staff len = {} from total {}".format(staffgauge_len, h_gray))
    cv2.imshow("waterline result", img_waterline)
    cv2.waitKey(0)



if __name__ == "__main__":

    # Initialize parser
    parser = argparse.ArgumentParser()
    # Adding optional argument
    parser.add_argument("-i", "--image", help="image path")

    # Read arguments from command line
    args = parser.parse_args()

    # test measurement algorithm
    measure_staffgauge_len(args.image)
