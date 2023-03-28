import pandas as pd
import cv2
import base64

def base64_to_image(img_b64):
    #img = base64.b64decode(img_b64)
    pass


if __name__ == "__main__":
    
    # read excel file
    df = pd.read_excel('staffgauge.xlsx', index_col=None, header=None)

    # iterating over rows using iterrows() function
    for i, row in df.iterrows():
        if i == 1:
            b64 = row[1]
            x1 = row[2]
            y1 = row[3]
            x2 = row[4]
            y2 = row[5]
            base64_to_image(b64)
    
    """
    # Window name in which image is displayed
    window_name = 'Image'
      
    w = frame.shape[1]
    h = frame.shape[0]

    # Start coordinate, here (5, 5)
    # represents the top left corner of rectangle
    start_point = (5, 5)
      
    # Ending coordinate, here (220, 220)
    # represents the bottom right corner of rectangle
    end_point = (220, 220)
          
    # Blue color in BGR
    color = (255, 0, 0)
      
    # Line thickness of 2 px
    thickness = 2
      
    # Using cv2.rectangle() method
    # Draw a rectangle with blue line borders of thickness of 2 px
    image = cv2.rectangle(image, start_point, end_point, color, thickness)
      
    # Displaying the image 
    cv2.imshow(window_name, image)
    """    
