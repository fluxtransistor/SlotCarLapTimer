import cv2
import time
import sys

# set camera parameters
VIDEO_DEVICE = 1  # change this if the wrong camera is detected
RESOLUTION = (1280, 720)  # the resolution must be supported by the camera, it uses defaults otherwise
EXPOSURE = -5  # set to zero for default, probably powers of 2 (-5 = 1/16)
MINIMUM_FRAME_TIME = 1000 / 30  # in milliseconds, to prevent duplicate frame processing

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
    return r_total, g_total, b_total


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


def record_time(colour, timestamp):
    pass


def current_time_ms():
    return int(time.time() * 1000)


def main():
    # initialize parallel arrays to store car data
    cars_colour = []  # RGB byte tuples for each car's colour
    cars_best_lap = []  # int for each car's best lap in milliseconds
    cars_num_laps = []  # int for each car's lap number
    cars_last_time = []  # int for each car's last timestamp in milliseconds
    cars_last_lap = []  # int for each car's last lap time in milliseconds

    # Initialize the video capture
    capture = cv2.VideoCapture(VIDEO_DEVICE)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
    if EXPOSURE < 0:  # otherwise use defaults
        capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)  # turn automatic exposure off
        capture.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE)
    background_subtractor = cv2.createBackgroundSubtractorMOG2(history=150, varThreshold=25, detectShadows=True)

    # initialize the last frame time
    last_frame_time = current_time_ms()

    while True:
        # code below is executed for every frame

        if last_frame_time + 20 > current_time_ms():  # checks whether the minimum frame time elapsed
            continue
        last_frame_time = current_time_ms()

        # capture the video frame
        return_value, frame = capture.read()
        if not return_value:
            print("The frame could not be captured.", file=sys.stderr)
            break
        cv2.imshow('frame', frame)
        # make a motion_mask and process it
        motion_mask = background_subtractor.apply(frame)
        motion_mask = cv2.medianBlur(motion_mask, 5)  # blur the mask to reduce noise
        motion_mask = cv2.threshold(motion_mask, 127, 255, cv2.THRESH_BINARY)[1]  # clamp everything to 0 or 255
        cv2.imshow("Slot Car Lap Timer: Motion Mask", motion_mask)  # optionally show motion_mask in a separate window

        # find contours

        contours = cv2.findContours(motion_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:][1]  # checks a maximum of 6 contours
        contour_areas = [cv2.contourArea(c) for c in contours]
        for i in range(len(contour_areas)):
            if contour_areas[i] < MINIMUM_CONTOUR_SIZE or contour_areas[i] > MAXIMUM_CONTOUR_SIZE:
                contours[i] = None  # invalidate the contour if it is too large or small

        for contour in contours:
            if contour is None:
                continue

            # executes for each valid contour
            x, y, w, h = cv2.boundingRect(contour)
            x_center = x + int(w / 2)
            y_center = y + int(h / 2)

            if (x_center, y_center) > DETECTION_BOX[0]:
                if (x_center, y_center) < DETECTION_BOX[1]:
                    time_detected = current_time_ms()
                    mask_clip = background_subtractor.apply(frame)
                    # create cropped clips of the captured motion
                    mask_clip, frame_clip = mask_clip[y:y + h, x:x + w], frame[y:y + h, x:x + w]
                    # cv2.imshow("clip", frame_clip) #  display the captured motion clip for debugging
                    countpx = 0

                    record_time((0, 0, 0), time_detected)
                cv2.circle(frame, (x_center, y_center), 8, (255, 255, 255), -1)

        # frame = cv2.rectangle(frame, flag[0], flag[1], clr)
        # frame = cv2.rectangle(frame, (0, 620), (320, 720), cars[1], -1)
        # frame = cv2.rectangle(frame, (320, 620), (640, 720), cars[2], -1)
        # frame = cv2.rectangle(frame, (640, 620), (720, 960), cars[3], -1)
        # frame = cv2.rectangle(frame, (640, 620), (960, 1280), cars[4], -1)
        # frame = cv2.putText(frame, str(laptime), (8, 705), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
