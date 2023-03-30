import os
import glob
import base64
import pandas as pd


# read image
img_path="sample-staffgauge.JPG"
with open(img_path, "rb") as image2string:
    img_base64 = base64.b64encode(image2string.read())
    #img_decode = base64.b64decode(img_base64) 
    #image_result = open(image_filename, 'wb')
    #image_result.write(img_decode)
    #image_txtfile = open(image_filename + ".txt", 'wb')
l = []
l.append(img_base64)
print("len=", len(img_base64), "\n\n")
#print(img_base64)
# save to excel
df = pd.DataFrame({'base': l})
filename = "portfolio.xlsx"
df.to_excel(filename)



# read excel test
df2 = pd.read_excel(filename, index_col=None, header=None)

# iterating over rows using iterrows() function
for i, row in df2.iterrows():
    if i == 1:
        b64 = row[1]
        print("\n*************************************************")
        print("len=", len(b64), "\n\n")
        #print(b64)
        img_decode = base64.b64decode(b64) 
        image_result = open("after-decode.png", 'wb')
        image_result.write(img_decode)
