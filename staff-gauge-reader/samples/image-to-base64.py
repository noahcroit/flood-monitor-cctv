import os
import glob
import base64
import pandas as pd



def extract_coor_from_yolomark(txt_path):
    f = open(txt_path, "r")
    data = f.read().strip("\n").split(" ")
    x_center = float(data[1])
    y_center = float(data[2])
    x_width  = float(data[3])
    y_width  = float(data[4])
    
    x1 = x_center - x_width/2
    y1 = y_center - y_width/2 
    x2 = x_center + x_width/2
    y2 = y_center + y_width/2

    return x1, y1, x2, y2

if __name__ == "__main__":
    yolomark_dir = "../yolov4-staffgauge/img"

    l_x1 = []
    l_y1 = []
    l_x2 = []
    l_y2 = []
    l_base64 = []
    l_name = []

    # iterate all files in image folder (.txt from Yolomark and .png image)
    txt_files = []
    txt_files = glob.glob(yolomark_dir + "/*.txt")
    for file in txt_files:
        # extract ROI square coordinate (X1, Y1, X2, Y2)
        x1, y1, x2, y2 = extract_coor_from_yolomark(file)
        l_x1.append(x1)
        l_y1.append(y1)
        l_x2.append(x2)
        l_y2.append(y2)
        
        # convert image to base64
        image_filename = file.split("../yolov4-staffgauge/img/")[1].split(".txt")[0] + ".png" # find the current image's filename
        img_path="../yolov4-staffgauge/img/" + image_filename
        with open(img_path, "rb") as image2string:
            img_base64 = base64.b64encode(image2string.read())
            img_decode = base64.b64decode(img_base64) 
            #image_result = open(image_filename, 'wb')
            #image_result.write(img_decode)
            #image_txtfile = open(image_filename + ".txt", 'wb')
            #image_txtfile.write(img_base64)
        l_name.append(image_filename)
        l_base64.append(img_base64)
        
        

    # convert to pandas's dataframe
    
    pdata = pd.DataFrame({'img_name': l_name,
                          'base64': l_base64,
                              'x1': l_x1,
                              'y1': l_y1,
                              'x2': l_x2,
                              'y2': l_y2
                              })
    """ 
    pdata = pd.DataFrame({'img_name': l_name,
                              'x1': l_x1,
                              'y1': l_y1,
                              'x2': l_x2,
                              'y2': l_y2
                              })
                              """
    # writing to files
    filename_excel = "staffgauge.xlsx"
    filename_csv = "staffgauge.csv"
    pdata.to_excel(filename_excel)
    pdata.to_csv(filename_csv, sep=',', encoding='utf-8')

