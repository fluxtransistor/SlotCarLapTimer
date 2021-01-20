import cv2
import time
import sys
import operator
import coldiff

# set camera parameters
VIDEO_DEVICE = 1  # change this if the wrong camera is used
RESOLUTION = (1280, 720)  # defaults if unsupported, but may crash because of invalid coordinates
EXPOSURE = -5  # set to zero for default, probably powers of 2 (-5 = 1/16)
MINIMUM_FRAME_TIME = 1000 / 30  # in milliseconds, to prevent duplicate frame processing

# set detection parameters
MINIMUM_CONTOUR_SIZE = 10000  # minimum pixel area for valid contours
MAXIMUM_CONTOUR_SIZE = 100000  # maximum pixel area for valid contours
DETECTION_BOX = ((750, 130), (850, 600))  # coordinates for the finish line bounding box

# set identification parameters
MAX_DIFFERENCE = 14  # the sensitivity for detecting different cars
TIME_THRESHOLD = 500  # minimum time difference before detecting a new lap (ms)

# set colour averaging parameters
# these specify how much of the cropped image is taken into account
STEP_X = 4  # row step, higher is faster
STEP_Y = 4  # column step, higher is faster
MIN_Y = 20  # lower bound for y in percent
MAX_Y = 50  # upper bound for y in percent
MIN_X = 25  # lower bound for x in percent
MAX_X = 75  # upper bound for x in percent

# set graphics parameters (these settings are best for 1280x720)
FONT_SCALE = 2  # arbitrary unit
GRAPHIC_POSITION = "bottom"  # bottom or top
GRAPHIC_HEIGHT = 100  # pixels
MAX_COLUMNS = 4  # change if text does not fit
TEXT_PADDING_X = 8  # pixels
TEXT_PADDING_Y = 40  # pixels
LITTLE_TEXT_PADDING = 10  # pixels


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
    for y in range(MIN_Y * height // 100, MAX_Y * height // 100, STEP_Y):
        for x in range(MIN_X * width // 100, MAX_X * width // 100, STEP_X):
            if mask[y, x] == 255:
                r_total += image[y, x][0]
                g_total += image[y, x][1]
                b_total += image[y, x][2]
                pixel_count += 1
    r_total = int(r_total / (pixel_count + 1))
    g_total = int(g_total / (pixel_count + 1))
    b_total = int(b_total / (pixel_count + 1))
    return r_total, g_total, b_total


def current_time_ms():
    return int(time.time() * 1000)


class Car():

    def __init__(self):
        self.colour = (0, 0, 0)
        self.best_lap = 0
        self.num_laps = 0  # the number of times a car has crossed the finish line, not complete laps
        self.last_timestamp = 0  # the last time the car crossed the finish line (ms)
        self.lap_times = []  # a list of the lap times in ms

    def record_lap(self, colour, time_recorded):
        if time_recorded > self.last_timestamp + TIME_THRESHOLD:
            r_new, g_new, b_new = colour
            r_old, g_old, b_old = self.colour
            r = (r_old * self.num_laps + r_new) // (self.num_laps + 1)  # average the new colour with the old colours
            g = (g_old * self.num_laps + g_new) // (self.num_laps + 1)  # average the new colour with the old colours
            b = (b_old * self.num_laps + b_new) // (self.num_laps + 1)  # average the new colour with the old colours
            self.colour = (r, g, b)
            self.num_laps += 1
            if self.last_timestamp != 0:
                new_lap = time_recorded - self.last_timestamp
                for old_lap in self.lap_times:
                    if old_lap < new_lap:
                        break
                else:
                    self.best_lap = new_lap
                self.lap_times.append(new_lap)
            self.last_timestamp = time_recorded

    def difference(self, colour):  # difference between black and white is about 58
        colour_lab = coldiff.rgb2lab(colour)
        own_colour_lab = coldiff.rgb2lab(self.colour)
        return coldiff.cie94(colour_lab, own_colour_lab)


def show_graphic(image, car_list):
    car_list.sort(key=operator.attrgetter("num_laps"))
    car_list.reverse()
    valid_cars = sum([1 for i in car_list if i.lap_times])
    number_of_columns = min(valid_cars, MAX_COLUMNS)
    if number_of_columns == 0:
        return image
    for i in range(number_of_columns):
        x1 = (i * RESOLUTION[0]) // number_of_columns
        x2 = ((i + 1) * RESOLUTION[0]) // number_of_columns
        if GRAPHIC_POSITION == 'bottom':
            y2 = RESOLUTION[1]
        else:
            y2 = GRAPHIC_HEIGHT
        y1 = y2 - GRAPHIC_HEIGHT
        colour = car_list[i].colour
        if (colour[0] * 0.299 + colour[1] * 0.587 + colour[2] * 0.114) > 186:
            text_colour = (0, 0, 0)
        else:
            text_colour = (255, 255, 255)
        text = str((car_list[i].lap_times[-1] // 10) / 100)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), colour, -1)
        image = cv2.putText(image, text, (x1 + TEXT_PADDING_X, y2 - TEXT_PADDING_Y), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, text_colour, 3)
    return image


def select_car(colour, car_list):
    minimum_difference = 0
    minimum_index = -1
    for i in range(len(car_list)):
        difference = car_list[i].difference(colour)
        if (difference < minimum_difference or minimum_index == -1) and difference < MAX_DIFFERENCE:
            minimum_index = i
            minimum_difference = difference
    print(car_list, minimum_index)
    return minimum_index


def main():
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

    car_list = []
    while True:
        # code below is executed for every frame

        if last_frame_time + MINIMUM_FRAME_TIME > current_time_ms():  # checks whether the minimum frame time elapsed
            continue
        last_frame_time = current_time_ms()

        # capture the video frame
        return_value, frame = capture.read()
        if not return_value:
            print("The frame could not be captured.", file=sys.stderr)
            break

        # make a motion_mask and process it
        motion_mask = background_subtractor.apply(frame)
        motion_mask = cv2.medianBlur(motion_mask, 5)  # blur the mask to reduce noise
        motion_mask = cv2.threshold(motion_mask, 127, 255, cv2.THRESH_BINARY)[1]  # clamp everything to 0 or 255
        cv2.imshow("Slot Car Lap Timer: Motion Mask", motion_mask)  # optionally show motion_mask in a separate window

        # find contours
        contours = cv2.findContours(motion_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-8:][0]  # checks a maximum of 8 contours
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
                    # create cropped clips of the captured motion
                    mask_clip, frame_clip = motion_mask[y:y + h, x:x + w], frame[y:y + h, x:x + w]
                    cv2.imshow("clip", frame_clip)  # display the captured motion clip for debugging
                    colour = average_colour(frame_clip, mask_clip)
                    car_index = select_car(colour, car_list)
                    if car_index == -1:
                        car_list.append(Car())
                    car_list[car_index].record_lap(colour, time_detected)
                    frame = cv2.circle(frame, (x_center, y_center), 8, (255, 255, 255), -1)

        frame = cv2.rectangle(frame, (0, 0), (100, 100), colour, -1)
        # draw the detection box on the image
        frame = cv2.rectangle(frame, DETECTION_BOX[0], DETECTION_BOX[1], (255, 255, 255))
        frame = show_graphic(frame, car_list)
        # frame = cv2.rectangle(frame, (0, 620), (320, 720), cars[1], -1)
        # frame = cv2.rectangle(frame, (320, 620), (640, 720), cars[2], -1)
        # frame = cv2.rectangle(frame, (640, 620), (960, 720), cars[3], -1)
        # frame = cv2.rectangle(frame, (960, 620), (1280, 720), cars[4], -1)
        # frame = cv2.putText(frame, str(laptime), (8, 705), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
