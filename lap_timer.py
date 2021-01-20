import cv2
import numpy as np
import time

# set camera parameters
VIDEO_DEVICE = 1  # change this if the wrong camera is detected
RESOLUTION = (1280, 720)  # the resolution must be supported by the camera, it uses defaults otherwise
EXPOSURE = -5  # set to zero for default, probably powers of 2 (-5 = 1/16)

# set detection parameters
MINIMUM_CONTOUR_SIZE = 8000  # minimum pixel area for valid contours
MAXIMUM_CONTOUR_SIZE = 100000  # maximum pixel area for valid contours
DETECTION_BOX = ((580, 130), (850, 600))  # coordinates for the finish line bounding box

# set identification parameters
MAX_DIFFERENCE = 80  # maximum RGB difference to identify a car (0 - 768)
TIME_THRESHOLD = 0.5  # minimum time difference before detecting a new lap





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


def average_colour(image, mask):  # averages some pixels in image where the corresponding pixel in mask is 255
    width = image.shape[1]
    height = image.shape[0]
    r_total, g_total, b_total = (0, 0, 0)
    pixel_count = 0
    # the loop only uses every 4th row and column for speed
    # it ignores the left and right 25% of the image
    # it also ignores the top 10% and bottom 50% of the image
    # this can provide better recognition if a car has different colours on the left and right side
    # the x axis is expected to be perpendicular to the cars' direction of travel
    for y in range(height // 10, height // 2, 4):
        for x in range(width // 4, 3 * width // 4, 4):
            if mask[y, x] == 255:
                r_total += image[y, x][0]
                g_total += image[y, x][1]
                b_total += image[y, x][2]
                pixel_count += 1
    r_total = int(r_total / (pixel_count + 1))
    g_total = int(g_total / (pixel_count + 1))
    b_total = int(b_total / (pixel_count + 1))
    clr = (r_total, g_total, b_total)


def difference(clr1, clr2):  # find the sum of absolute differences between two RGB values (0 - 768)
    dif = 0
    for i in range(3):
        dif += abs(clr1[i] - clr2[i])
    return dif


def select_similar_colour(colour, colour_list, maximum_difference):  # returns the car index with the lowest color difference, or -1 if it is greater than MAX_DIFFERENCE
    minimum_value, minimum_index = -1, -1
    for car_index in range(len(colour_list)):
        current_difference = difference(colour_list[car_index], colour)
        if current_difference < minimum_value or minimum_value < 0:
            if current_difference < maximum_difference:
                minimum_value = current_difference
                minimum_index = car_index
    return minimum_index


def timerecorded(clr, timestamp):
    most_similar = select_similar_colour(clr)
    pass


def main():
    # initialize parallel arrays to store car data
    cars_colour = []  # RGB byte tuples for each car's colour
    cars_best_lap = []  # int for each car's best lap in milliseconds
    cars_num_laps = []  # int for each car's lap number
    cars_last_time = []  # int for each car's last timestamp in milliseconds
    cars_last_lap = []  # int for each car's last lap time in milliseconds

    # Initialize the video capture
    cap = cv2.VideoCapture(VIDEO_DEVICE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
    if EXPOSURE < 0:  # otherwise use defaults
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)  # turn automatic exposure off
        cap.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE)
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
            if areas[i] < MINIMUM_CONTOUR_SIZE or areas[i] > MAXIMUM_CONTOUR_SIZE:
                contours[i] = None

        for c in contours:
            if c is None:
                continue
            cnt = c
            x, y, w, h = cv2.boundingRect(cnt)
            x2 = x + int(w / 2)
            y2 = y + int(h / 2)

            if (x2, y2) > DETECTION_BOX[0]:
                if (x2, y2) < DETECTION_BOX[1]:
                    currtime = time.time()
                    laptime = (int((currlap - prevlap) * 1000)) / 1000
                    col = back_sub.apply(frame)
                    col, col2 = col[y:y + h, x:x + w], frame[y:y + h, x:x + w]
                    cv2.imshow("clip", col2)
                    countpx = 0

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
