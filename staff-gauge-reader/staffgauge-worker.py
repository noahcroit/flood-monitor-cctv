import threading
import cv2











if __name__ == "__main__":
    
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

    while is_run:


