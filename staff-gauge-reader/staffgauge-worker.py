import threading
import queue
import time
import cv2
import numpy as np
import argparse



# Global Variables for Threads 
snapshot_isrun = True



def task_snapshot(queue_snapshot, source):
    global snapshot_isrun

    stream = cv2.VideoCapture(source)
    try:
        # looping
        while stream.isOpened() and snapshot_isrun:
            ret, frame = stream.read()
            queue_snapshot.put(frame) # put frame into queue
            time.sleep(1)    # simulate blocking delay
        stream.release()
        snapshot_isrun = False
    except:
        print("something wrong in task snapshot!")
        stream.release()
        snapshot_isrun = False
        
def task_overlay(queue_roi):
    global snapshot_isrun

    # looping
    while snapshot_isrun:
        try:
            if not queue_roi.empty():
                dict_roi = queue_roi.get()
                frame = dict_roi['frame']
                obj_class = dict_roi['class']
                x1 = dict_roi['x1']
                y1 = dict_roi['y1']
                x2 = dict_roi['x2']
                y2 = dict_roi['y2']
                color = dict_roi['color']
                print(obj_class)
                if not obj_class is None:
                    for i in range(len(obj_class)):
                        # draw a bounding box rectangle and label on the frame
                        cv2.rectangle(frame, (x1[i], y1[i]), (x2[i], y2[i]), [102, 220, 225], 2)
                        #text = "{}: {:.4f}".format(obj_class[i], confidences[i])
                        text = "{}".format(obj_class[i])
                        cv2.putText(frame, text, (x1[i], y1[i] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, [102, 220, 225], 5)

                    # resize image frame for 1/2
                    w_resize = int(frame.shape[1] * 0.5)
                    h_resize = int(frame.shape[0] * 0.5)
                    dim = (w_resize, h_resize)
                    frame_resize = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
                    
                    # Using cv2.putText() method
                    # font
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    org = (50, 50)
                    fontScale = 1
                    color = (255, 255, 255)
                    thickness = 2
                    level = dict_roi['level']
                    cv2.putText(frame_resize, 'waterlevel={}'.format(level), org, font, fontScale, color, thickness, cv2.LINE_AA)

                    # draw overlay
                    cv2.imshow("output frame", frame_resize)
                    cv2.waitKey(1000)
            time.sleep(0.1)
        except Exception as e:
            print("something is wrong in task overlay")
            print(e)

    cv2.destroyAllWindows()

def task_find_roi(queue_in, queue_out):
    global snapshot_isrun

    # Label File Configuration
    # load the class output labels of input YOLO model
    label_file = "yolov4-staffgauge/obj.names"
    LABELS = open(label_file).read().strip().split("\n")
    print(LABELS)

    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    CLASS_COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

    # YOLO Configuration
    # load YOLO object detector (.weight file and .cfg file)
    cfg_file = "yolov4-staffgauge/yolov4-staffgauge.cfg"
    weight_file = "yolov4-staffgauge/backup/yolov4-staffgauge_1000.weights"
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(cfg_file, weight_file)

    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    print("getUnconectedOutLayer", net.getUnconnectedOutLayers())
    #ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    
    # Confidence & Threshold for making the decision in YOLO
    user_confidence=0.5
    user_threshold=0.3

    # looping for YOLO
    while snapshot_isrun:
        try:
            # read image frame from queue
            frame_ready = False
            if not queue_in.empty():
                # read input buffer
                frame_in = queue_in.get()
                if not frame_in is None:
                    frame_ready = True

            # processing
            # YOLO object detection
            if frame_ready:  
                # "frame" needs to be opencv's image-dtype
                # Input Image Configuration
                print("[INFO] accessing image ...")
                (H, W) = frame_in.shape[:2]
            
                # construct a blob from the input frame and then perform a forward
                # pass of the YOLO object detector, giving us our bounding boxes
                # and associated probabilities
                blob = cv2.dnn.blobFromImage(frame_in, 1 / 255.0, (416, 416), swapRB=True, crop=False)
                net.setInput(blob)

                layerOutputs = net.forward(ln)

                # initialize our lists of detected bounding boxes, confidences,
                # and class IDs, respectively
                boxes = []
                confidences = []
                classIDs = []

                # loop over each of the layer outputs
                for output in layerOutputs:
                    # loop over each of the detections
                    for detection in output:
                        # extract the class ID and confidence (i.e., probability)
                        # of the current object detection
                        scores = detection[5:]
                        classID = np.argmax(scores)
                        confidence = scores[classID]

                        # filter out weak predictions by ensuring the detected
                        # probability is greater than the minimum probability
                        if confidence > user_confidence:
                            # scale the bounding box coordinates back relative to
                            # the size of the image, keeping in mind that YOLO
                            # actually returns the center (x, y)-coordinates of
                            # the bounding box followed by the boxes' width and
                            # height
                            box = detection[0:4] * np.array([W, H, W, H])
                            (centerX, centerY, width, height) = box.astype("int")

                            # use the center (x, y)-coordinates to derive the top
                            # and and left corner of the bounding box
                            x = int(centerX - (width / 2))
                            y = int(centerY - (height / 2))

                            # update our list of bounding box coordinates,
                            # confidences, and class IDs
                            boxes.append([x, y, int(width), int(height)])
                            confidences.append(float(confidence))
                            classIDs.append(classID)

                # apply non-maxima suppression to suppress weak, overlapping
                # bounding boxes
                idxs = cv2.dnn.NMSBoxes(boxes, confidences, user_confidence, user_threshold)

                output_class = None
                pos_x = None
                pos_y = None
                pos_x1 = None
                pos_y1 = None
                pos_x2 = None
                pos_y2 = None
                output_class = None
                classes_color = None

                # ensure at least one detection exists
                if len(idxs) > 0:
                    pos_x1 = []
                    pos_y1 = []
                    pos_x2 = []
                    pos_y2 = []
                    output_class = []
                    classes_color = []

                    # loop over the indexes we are keeping
                    for i in idxs.flatten():
                        # extract the bounding box coordinates
                        (x1, y1) = (boxes[i][0], boxes[i][1])
                        (w, h) = (boxes[i][2], boxes[i][3])
                        x2 = x1 + w
                        y2 = y1 + h
                        # output the result with this format
                        # output_class = [output_class[0], output_class[1], output_class[2], ...]
                        # [x1[0], x1[1], x1[2], ...]
                        # [y1[0], y1[1], y1[2], ...]
                        # [x2[0], x2[1], x2[2], ...]
                        # [y2[0], y2[1], y2[2], ...]
                        color = [int(c) for c in CLASS_COLORS[classIDs[i]]]
                        output_class.append(LABELS[classIDs[i]])
                        pos_x1.append(x1)
                        pos_y1.append(y1)
                        pos_x2.append(x2)
                        pos_y2.append(y2)
                        classes_color.append(color)
                
                # write to output buffer after finished the process
                print("Yolo finished")

                # waterlevel calculation
                level=None
                if not output_class is None:
                    level = measure_waterlevel(frame_in, pos_x1[0], pos_x2[0], pos_y1[0], pos_y2[0])
                    print("level={}".format(level))

                dict_output = {"frame":frame_in, 
                               "class":output_class, 
                               "x1":pos_x1, 
                               "y1":pos_y1, 
                               "x2":pos_x2, 
                               "y2":pos_y2,
                               "color":classes_color,
                               "level":level
                               }
                queue_out.put(dict_output)
            else:
                print("frame=None")
            time.sleep(0.1)

        except Exception as e:
            print("something wrong in task YOLO")
            print(e)

def measure_waterlevel(img, x1, x2, y1, y2):
    
    # Crop for only area of staffgauge, ROI should be from the YOLOv4 result
    img_crop = img[y1:y2, x1:x2]

    # convert to hsv colorspace
    hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)

    # lower bound and upper bound for Yellow color
    lower_bound = np.array([20, 130, 130])
    upper_bound = np.array([50, 255, 255])

    # find the colors within the boundaries
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # bitwise with mask
    img_segmented = cv2.bitwise_and(img_crop, img_crop, mask=mask)

    # convert the input image to grayscale
    gray = cv2.cvtColor(img_segmented, cv2.COLOR_BGR2GRAY)
    # apply thresholding to convert grayscale to binary image
    ret, thresh = cv2.threshold(gray, 70, 255, 0)

    # Find the water line
    # Use sum of grey-value in row of gray-image
    h_gray = gray.shape[0]
    w_gray = gray.shape[1]
    sum_gray = []
    for row in range(h_gray):
        tmp = 0
        for col in range(w_gray):
            tmp = tmp + gray[row, col]
        sum_gray.append(tmp)
    
    count = sum(sum_gray)
    
    # linear regression
    a = -0.8859
    b = 102.8
    count = count / 100000
    level = a*count + b
    return round(level, 2)



if __name__ == "__main__":
    
    # Initialize parser
    parser = argparse.ArgumentParser()
    # Adding optional argument
    parser.add_argument("-s", "--source", help="source-type (rtsp, video)")

    # Read arguments from command line
    args = parser.parse_args()

    # URL for video or camera source
    if args.source == "rtsp":
        #source="rtsp://admin:scadatest1234@192.168.4.197:554/Streaming/Channels/101/"
        source="rtsp://root:axiscamera@192.168.4.196/axis-media/media.amp"
    if args.source == "video":
        #source="videos/site-sample-2.mp4"
        source="videos/00008.mp4"



    # Task: Snapshot
    # @Description: Take snapshot a frame from RTSP camera with opencv, Then queue frame to image queue
    # @Wait for task:

    # Task: ROI Extraction
    # @Description: ROI extraction with Yolov4 StaffGauge Detection
    # @Wait for task: 
    # - Snapshot

    # Task: Image Overlay & Waterlevel Extraction
    # @Description: Create new image with ROI overlay on the original image,
    #                Apply waterlevel measurement algorithm
    #                (These 2 tasks should run simultaneously with multiprocess module) 
    # @Wait for task:
    # - ROI Extraction
    # - Snapshot



    # shared queue
    queue_snapshot = queue.Queue()
    queue_roi = queue.Queue()

    # config tasks
    t1 = threading.Thread(target=task_snapshot, args=(queue_snapshot, source))
    t2 = threading.Thread(target=task_find_roi, args=(queue_snapshot, queue_roi))
    t3 = threading.Thread(target=task_overlay, args=(queue_roi,))
    
    # start tasks
    t1.start()
    t2.start()
    t3.start()

    # wait for all threads to finish
    while snapshot_isrun:
        try:
            if not t1.is_alive():
                print("snapshot task is dead. stop worker...")
                snapshot_isrun = False
            if not t2.is_alive():
                print("restart task find roi")
                t2.start()
            if not t3.is_alive():
                print("restart task overlay")
                t3.start()

            time.sleep(5)

        except KeyboardInterrupt:
            print("main thread interrupted. Stop snaptshot")
            snapshot_isrun = False

