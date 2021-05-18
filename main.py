import numpy as np
import cv2
import math


class Color:
    def __init__(self, bgr_form, hsv_lower, hsv_upper):
        self.bgr_form = bgr_form
        self.hsv_lower = hsv_lower
        self.hsv_upper = hsv_upper
        self.last_pos = 0
        self.pos = 0
        self.scale = 0


def get_contours(img):

    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area > 300:

            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            obj_cor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)

            if obj_cor > 3:
                object_type = "Marker"
            else:
                object_type = "None"

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, object_type, (x + (w // 2) - 10, y + (h // 2) - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 2)

            return area, tuple([x, y])
    return 0, 0


def check_color(img, output_canvas, color_id):

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img_hsv, color_id.hsv_lower, color_id.hsv_upper)

    filtered_img = img.copy()
    filtered_img[np.where(mask == 0)] = 0
    filtered_img[np.where(mask != 0)] = 255

    color_id.scale, color_id.pos = get_contours(cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(cv2.cvtColor(filtered_img, cv2.COLOR_HSV2BGR),
                                                                      cv2.COLOR_BGR2GRAY), (7, 7), 1), 50, 50))

    # print(color_id.scale, color_id.pos, type(color_id.last_pos))

    if type(color_id.pos) == tuple:

        if type(color_id.last_pos) == int:
            color_id.last_pos = color_id.pos
        else:
            if math.hypot(color_id.pos[0]-color_id.last_pos[0], color_id.pos[1]-color_id.last_pos[1]) < 20:
                output_canvas = cv2.line(output_canvas, color_id.pos, color_id.last_pos, color_id.bgr_form, int(color_id.scale/100))

        color_id.last_pos = color_id.pos

    return output_canvas, color_id, filtered_img


cap = cv2.VideoCapture(1)

canvas = np.zeros(cap.read()[1].shape[0:2] + tuple([3]))

render_board = np.zeros(tuple([cap.read()[1].shape[0] + 120, cap.read()[1].shape[1] + 80, 3]))
render_board[:] = (130, 130, 0)
cv2.putText(render_board, "Drawing Simulator", (200, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 4)
cv2.putText(render_board, "Supports BLUE, GREEN, and RED. Press 'r' to reset and 'q' to quit", (80, 80), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

upper_red = Color(tuple([0, 0, 255]), tuple([0, 120, 100]), tuple([10, 255, 255]))
lower_red = Color(tuple([0, 0, 255]), tuple([170, 120, 100]), tuple([180, 255, 255]))
green = Color(tuple([0, 255, 0]), tuple([40, 60, 50]), tuple([80, 255, 255]))
blue = Color(tuple([255, 0, 0]), tuple([100, 120, 100]), tuple([125, 255, 255]))

while True:

    ret, frame = cap.read()
    frame = np.flip(frame, 1)
    input_img = frame.copy()
    imgContour = frame.copy()

    canvas, upper_red, a = check_color(input_img, canvas, upper_red)
    canvas, lower_red, b = check_color(input_img, canvas, lower_red)
    canvas, green, c = check_color(input_img, canvas, green)
    canvas, blue, blue_filt = check_color(input_img, canvas, blue)

    x_off = (render_board.shape[1]-canvas.shape[1])//2
    y_off = render_board.shape[0]-canvas.shape[0]-20
    render_board[y_off:y_off+canvas.shape[0], x_off:x_off+canvas.shape[1]] = canvas

    # cv2.imshow('mask', mask)

    # cv2.imshow('frame', imgContour)
    # cv2.imshow('canvas', canvas)

    cv2.imshow('DrawSimulator', render_board)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.waitKey(1) & 0xFF == ord('r'):
        canvas = np.zeros(cap.read()[1].shape[0:2] + tuple([3]))


cap.release()
cv2.destroyAllWindows()

# img_hsv = cv2.cvtColor(imgContour, cv2.COLOR_BGR2HSV)
    # mask = cv2.inRange(img_hsv, green.hsv_lower, green.hsv_upper)
    #
    # filtered_img = imgContour.copy()
    # filtered_img[np.where(mask == 0)] = 0
    #
    # cv2.imshow("filtered green", filtered_img)



    # img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #
    # # lower mask (0-10)
    # upper_red = Color(tuple([0, 0, 255]), tuple([0, 120, 100]), tuple([10, 255, 255]))
    # lower_red = Color(tuple([0, 0, 255]), tuple([170, 120, 100]), tuple([180, 255, 255]))
    # green = Color(tuple([0, 255, 0]), tuple([50, 120, 100]), tuple([70, 255, 255]))
    #
    # check_color()
    #
    # lower_red = np.array([0, 120, 100])
    # upper_red = np.array([10, 255, 255])
    # mask0 = cv2.inRange(img_hsv, lower_red, upper_red)
    #
    # # upper mask (170-180)
    # lower_red = np.array([170, 120, 100])
    # upper_red = np.array([180, 255, 255])
    # mask1 = cv2.inRange(img_hsv, lower_red, upper_red)
    #
    # # join masks
    # mask = mask0 + mask1
    #
    # output_img = frame.copy()
    # output_img[np.where(mask == 0)] = 0
    # scale, pos = get_contours(cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(cv2.cvtColor(output_img, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY), (7, 7), 1), 50, 50))
    #
    # if type(pos) == tuple:
    #     if type(last_pos) == int:
    #         last_pos = pos
    #     else:
    #         if math.hypot(pos[0]-last_pos[0], pos[1]-last_pos[1]) < 20:
    #             for k in range(pos[0] - 2, pos[0] + 3):
    #                 for i in range(pos[1] - 2, pos[1] + 3):
    #                     # canvas[i, k] = (0, 0, 255)
    #                     canvas = cv2.line(canvas, pos, last_pos, (0, 0, 255), int(scale/60))
    #     last_pos = pos

    # print(pos[0], pos[1], canvas.shape)