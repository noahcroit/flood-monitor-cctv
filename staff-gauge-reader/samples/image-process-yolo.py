import threading
import queue
import time
import cv2
import numpy as np



def task_readframe(buffer_in):
    #source="rtsp://admin:scadatest1234@192.168.4.197:554/Streaming/Channels/101/"
    source="../site-sample.mp4"
    #source="../00008.mp4"
    stream = cv2.VideoCapture(source)
    try:
        while stream.isOpened():
            ret, frame = stream.read()
            queue_in.put(frame) # put frame into queue
            time.sleep(0.6)  # simulate blocking delay
        stream.release()
    except KeyboardInterrupt:
        pass

def task_process(queue_in, queue_out):
    
    user_confidence=0.5
    user_threshold=0.3

    # Label File Configuration
    # load the class output labels of input YOLO model
    label_file = "../yolov4-staffgauge/obj.names"
    LABELS = open(label_file).read().strip().split("\n")
    print(LABELS)

    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    CLASS_COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

    # YOLO Configuration
    # load YOLO object detector (.weight file and .cfg file)
    cfg_file = "../yolov4-staffgauge/yolov4-staffgauge.cfg"
    weight_file = "../yolov4-staffgauge/backup/yolov4-staffgauge_1000.weights"
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(cfg_file, weight_file)

    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    print("getUnconectedOutLayer", net.getUnconnectedOutLayers())
    #ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]




    try:
        while True:
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
                # ensure at least one detection exists
                if len(idxs) > 0:
                    output_class = []
                    pos_x = []
                    pos_y = []
                    # loop over the indexes we are keeping
                    for i in idxs.flatten():
                        # extract the bounding box coordinates
                        (x, y) = (boxes[i][0], boxes[i][1])
                        (w, h) = (boxes[i][2], boxes[i][3])

                        # draw a bounding box rectangle and label on the frame
                        color = [int(c) for c in CLASS_COLORS[classIDs[i]]]
                        cv2.rectangle(frame_in, (x, y), (x + w, y + h), color, 2)
                        text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                        cv2.putText(frame_in, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 5)
                    
                        # output the result with this format
                        # output_class = [output_class[0], output_class[1], output_class[2], ...]
                        # [x[0], x[1], x[2], ...]
                        # [y[0], y[1], y[2], ...]
                        output_class.append(LABELS[classIDs[i]])
                        pos_x.append(x)
                        pos_y.append(y)
                
                # resize image frame for 1/2
                w_resize = int(frame_in.shape[1] * 0.5)
                h_resize = int(frame_in.shape[0] * 0.5)
                dim = (w_resize, h_resize)
                frame_resize = cv2.resize(frame_in, dim, interpolation=cv2.INTER_AREA)
                print("Yolo finished, ", output_class)



                # write to output buffer after finished the process
                queue_out.put(frame_resize)
    
    except KeyboardInterrupt:
        pass

def task_display(queue_out):
    try:
        while True:
            if not queue_out.empty():
                frame = queue_out.get()
                if not frame is None:
                    cv2.imshow("output frame", frame)
            # Press Q on keyboard to  exit
            cv2.waitKey(100)

    except KeyboardInterrupt:
        cv2.destroyAllWindows()

def task_qsize(queue):
    try:
        while True:
            print("q size = ")
            print(queue.qsize())
            time.sleep(1)

    except KeyboardInterrupt:
        cv2.destroyAllWindows()





if __name__ == "__main__":
    # shared queue
    queue_in   = queue.Queue()
    queue_out  = queue.Queue()

    # config tasks
    t1 = threading.Thread(target=task_readframe, args=(queue_in,))
    t2 = threading.Thread(target=task_process, args=(queue_in, queue_out))
    t3 = threading.Thread(target=task_display, args=(queue_out,))
    t4 = threading.Thread(target=task_qsize, args=(queue_in,))
    
    # start tasks
    t1.start()
    t2.start()
    t3.start()
    t4.start()
    # wait for all threads to finish
    t1.join()
    t2.join()
    t3.join()
    t4.join()
