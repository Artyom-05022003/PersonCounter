from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *


def main():
    # cap = cv2.VideoCapture(0)
    # cap.set(3, 720)
    # cap.set(4, 480)

    cap = cv2.VideoCapture("../Person-counter/people-video.mp4")  # For Video
    model = YOLO("../Yolo-Weights/yolov8m.pt")
    model.to('cuda')
    classNames = model.names
    print(model.device)
    #mask = cv2.resize(cv2.imread('mask.png'), (960, 540))

    # Tracking
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

    limits = [500, 20, 500, 500]
    limitsUp = [103, 161, 296, 161]
    limitsDown = [527, 489, 735, 489]
    totalCount = []

    while True:
        success, img = cap.read()
        #imgRegion = cv2.bitwise_and(img, mask)

        results = model(img, stream=True)

        detections = np.empty((0, 5))

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                w, h = x2 - x1, y2 - y1

                # Confidence
                conf = math.ceil(box.conf[0] * 100) / 100
                # Class Name
                cls = int(box.cls[0])
                currentClass = classNames[cls]
                if currentClass == "person" and conf > 0.2:
                    # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                    #                    scale=0.6, thickness=1, offset=3)
                    # cvzone.cornerRect(img, (x1, y1, w, h), l=10, t=2)
                    currentArray = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, currentArray))

        resultsTracker = tracker.update(detections)

        # cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), color=(0, 0, 255), thickness=5)
        cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), color=(0, 0, 255), thickness=5)
        cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), color=(0, 0, 255), thickness=5)

        for result in resultsTracker:
            x1, y1, x2, y2, id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h), l=10, t=2)
            cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(35, y1)),
                               scale=1, thickness=1, offset=5)

            cx, cy = x1+w//2, y1+h//2
            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            if limits[0]-100 < cx < limits[2]+100 and limits[1] < cy < limits[3]:
                if totalCount.count(id) == 0:
                    totalCount.append(id)
                    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), color=(0, 255, 0), thickness=5)
            # if limitsUp[0] < cx < limitsUp[2] and limitsUp[1] - 10 < cy < limitsUp[3] + 10:
            #     if totalCount.count(id) == 0:
            #         totalCount.append(id)
            #         cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]),
            #                  color=(0, 255, 0), thickness=5)
            #
            # elif limitsDown[0] < cx < limitsDown[2] and limitsDown[1] - 10 < cy < limitsDown[3] + 10:
            #     if totalCount.count(id) == 0:
            #         totalCount.append(id)
            #         cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]),
            #                  color=(0, 255, 0), thickness=5)

        cvzone.putTextRect(img, f'Count: {len(totalCount)}', (50, 50))

        cv2.imshow("Image", img)
        # cv2.imshow("ImageRegion", imgRegion)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
