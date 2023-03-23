import threading
import queue
import time
import cv2


def task_readframe(queue):
    source="rtsp://admin:scadatest1234@192.168.4.197:554/Streaming/Channels/101/"
    stream = cv2.VideoCapture(source)
    try:
        while stream.isOpened():
            ret, frame = stream.read()
            queue.put(frame) # put frame into queue
            time.sleep(0.1)  # simulate blocking delay
        stream.release()
    except KeyboardInterrupt:
        pass

def task_display(queue):
    try:
        while True:
            if not queue.empty():
                frame = queue.get()
                frame = cv2.resize(frame, (960, 540))
                time.sleep(0.1)  # simulate blocking delay
                cv2.imshow("frame", frame)
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
    # shared queue
    queue = queue.Queue()

    # config tasks
    t1 = threading.Thread(target=task_readframe, args=(queue,))
    t2 = threading.Thread(target=task_display, args=(queue,))
    t3 = threading.Thread(target=task_isqueuefull, args=(queue,))
    
    # start tasks
    t1.start()
    t2.start()
    t3.start()
    # wait for all threads to finish
    t1.join()
    t2.join()
    t3.join()
