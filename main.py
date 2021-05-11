import numpy as np
import cv2


def find_marker(img):
    # should find the location of the marker from a given mask input
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
    imgCanny = cv2.Canny(imgBlur, 50, 50)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(area)
    pass


cap = cv2.VideoCapture(0)

# img = np.zeros((480, 640, 3), np.uint8)
# img = cv2.line(img, (0, 0), (480, 640), (255, 0, 0), 5)

fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    frame = np.flip(frame, 1)
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # lower mask (0-10)
    lower_red = np.array([0, 100, 60])
    upper_red = np.array([10, 255, 255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([170, 100, 60])
    upper_red = np.array([180, 255, 255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

    # join my masks
    mask = mask0 + mask1
    # find_marker(mask)
    # red_bound = (np.array([50, 0, 0]), np.array([255, 50, 50]))
    # mask = cv2.inRange(frame, red_bound[0], red_bound[1])
    # fgmask = fgbg.apply(frame)
    # Insert processing operations here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    output_img = frame.copy()
    output_img[np.where(mask == 0)] = 0

    # print(frame.shape)

    cv2.imshow('frame', mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()