import threading
import queue
import time
import cv2
import argparse
import json


def task_readframe(queue, source):
    stream = cv2.VideoCapture(source)
    print("start capture")
    try:
        while stream.isOpened():
            ret, frame = stream.read()
            queue.put(frame) # put frame into queue
            time.sleep(0.1)  # simulate blocking delay
        stream.release()
        print("capture is over")
    except KeyboardInterrupt:
        pass

def task_display(queue, displayflag):
    try:
        while True:
            if not queue.empty():
                frame = queue.get()
                if not frame is None:
                    if displayflag == 'true':
                        frame = cv2.resize(frame, (960, 540))
                        cv2.imshow("output frame", frame)
                time.sleep(0.1)  # simulate blocking delay
            # Press Q on keyboard to  exit
            cv2.waitKey(10)

    except KeyboardInterrupt:
        cv2.destroyAllWindows()

def task_isqueuefull(queue):
    try:
        while True:
            print("q size = ")
            print(queue.qsize())
            time.sleep(1)

    except KeyboardInterrupt:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    
    # Initialize parser
    parser = argparse.ArgumentParser()
    # Adding optional argument
    parser.add_argument("-s", "--source", help="source-type (rtsp, video)")
    parser.add_argument("-j", "--json", help="JSON file for source path")
    parser.add_argument("-d", "--displayflag", help="display image with cv or not (true, false)", default='false')

    # Read arguments from command line
    args = parser.parse_args()

    # URL for video or camera source
    f = open(args.json)
    data = json.load(f)
    if args.source == "rtsp":
        source = data['rtsp']
    if args.source == "video":
        source = data['video']
    f.close()

    # Queue for frame buffering in image processing
    queue = queue.Queue()

    # config tasks
    t1 = threading.Thread(target=task_readframe, args=(queue, source))
    t2 = threading.Thread(target=task_display, args=(queue, args.displayflag))
    t3 = threading.Thread(target=task_isqueuefull, args=(queue,))
    
    # start tasks
    t1.start()
    t2.start()
    t3.start()

    # wait for all threads to finish
    t1.join()
    t2.join()
    t3.join()
