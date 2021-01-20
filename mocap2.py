import cv2
import numpy as np
import time

smin = 8000
smax = 100000
flag = ((580, 130), (850, 600))
cars = [(0, 0, 0)] * 4
best_laps = [0] * 4
nums_laps = [0] * 4
last_logs = [0] * 4

MAX_DIFFERENCE = 80
TIME_THRESHOLD = 0.5


def hsv_to_rgb(h, s, v):  # convert float HSV to byte RGB
    if s == 0.0:
        v *= 255
        return v, v, v
    i = int(h * 6.)
    f = (h * 6.) - i
    p, q, t = int(255 * (v * (1. - s))), int(255 * (v * (1. - s * f))), int(255 * (v * (1. - s * (1. - f))))
    v *= 255
    i %= 6
    if i == 0:
        return v, t, p
    if i == 1:
        return q, v, p
    if i == 2:
        return p, v, t
    if i == 3:
        return p, q, v
    if i == 4:
        return t, p, v
    if i == 5:
        return v, p, q


def difference(clr1, clr2):
    dif = 0
    for i in range(3):
        dif += abs(clr1[i] - clr2[i]) * 100
    return dif


def findcar(colour):  # returns the car index with the lowest color difference, or -1 if it is greater than MAX_DIFFERENCE
    minimum_value, minimum_index = -1, -1
    for car_index in range(len(cars)):
        current_difference = difference(cars[car_index], colour)
        if current_difference < minimum_value or minimum_value < 0:
            if current_difference < MAX_DIFFERENCE:
                minimum_value = current_difference
                minimum_index = car_index
    return minimum_index


def timerecorded(clr, timestamp):
    most_similar = findcar(clr)
    pass


def main():
    # Initialize the video capture
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
    cap.set(cv2.CAP_PROP_EXPOSURE, -5.0)  # fixed exposure at
    back_sub = cv2.createBackgroundSubtractorMOG2(history=150, varThreshold=25, detectShadows=True)

    kernel = np.ones((30, 30), np.uint8)
    phist = [((0, 0), 0)]
    clr = (0, 0, 0)
    while (True):

        ret, frame = cap.read()

        fg_mask = back_sub.apply(frame)

        fg_mask = cv2.medianBlur(fg_mask, 5)

        cv2.imshow("bs", fg_mask)
        _, fg_mask = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)

        fg_mask_bb = fg_mask
        contours, hierarchy = cv2.findContours(fg_mask_bb, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        areas = [cv2.contourArea(c) for c in contours]
        for i in range(len(areas)):
            if areas[i] < smin or areas[i] > smax:
                contours[i] = None

        for c in contours:
            if c is None:
                continue
            cnt = c
            x, y, w, h = cv2.boundingRect(cnt)

            x2 = x + int(w / 2)
            y2 = y + int(h / 2)

            if (x2, y2) > flag[0]:
                if (x2, y2) < flag[1]:
                    currtime = time.time()
                    laptime = (int((currlap - prevlap) * 1000)) / 1000
                    col = back_sub.apply(frame)
                    col, col2 = col[y:y + h, x:x + w], frame[y:y + h, x:x + w]
                    cv2.imshow("clip", col2)
                    countpx = 0
                    ra, ga, ba = (0, 0, 0)
                    for y in range(h // 10, h // 2, 4):
                        for x in range(w // 4, 3 * w // 4, 4):
                            if col[y, x] == 255:
                                ra += col2[y, x][0]
                                ga += col2[y, x][1]
                                ba += col2[y, x][2]
                                countpx += 1
                    ra = int(ra / (countpx + 1))
                    ga = int(ga / (countpx + 1))
                    ba = int(ba / (countpx + 1))
                    clr = (ra, ga, ba)
                    timerecorded(clr, currtime)
                cv2.circle(frame, (x2, y2), 8, clr, -1)

        # frame = cv2.rectangle(frame, flag[0], flag[1], clr)
        # frame = cv2.rectangle(frame, (0, 620), (320, 720), cars[1], -1)
        # frame = cv2.rectangle(frame, (320, 620), (640, 720), cars[2], -1)
        # frame = cv2.rectangle(frame, (640, 620), (720, 960), cars[3], -1)
        # frame = cv2.rectangle(frame, (640, 620), (960, 1280), cars[4], -1)
        # frame = cv2.putText(frame, str(laptime), (8, 705), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print(__doc__)
    main()
