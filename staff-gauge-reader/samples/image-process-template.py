import threading
import queue
import time
import cv2
import numpy as np



def task_readframe(buffer_in):
    source="rtsp://admin:scadatest1234@192.168.4.197:554/Streaming/Channels/101/"
    stream = cv2.VideoCapture(source)
    try:
        while stream.isOpened():
            ret, frame = stream.read()
            queue_in.put(frame) # put frame into queue
            time.sleep(0.1)  # simulate blocking delay
        stream.release()
    except KeyboardInterrupt:
        pass

def task_process(queue_in, queue_out):
    try:
        while True:
            # read input buffer
            frame_in = queue_in.get()

            # processing
            # still do nothing
            time.sleep(0.01)  # simulate blocking delay

            # write to output buffer after finished the process
            queue_out.put(frame_in)
    
    except KeyboardInterrupt:
        pass

def task_display(queue_out):
    try:
        while True:
            if not queue_out.empty():
                frame = queue_out.get()
                if not frame is None:
                    frame = cv2.resize(frame, (960, 540))
                    cv2.imshow("output frame", frame)
            # Press Q on keyboard to  exit
            cv2.waitKey(10)

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
    t4 = threading.Thread(target=task_qsize, args=(queue_out,))
    
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
