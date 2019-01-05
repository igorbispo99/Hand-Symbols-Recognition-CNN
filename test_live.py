import os
#os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import cv2 as cv
import numpy as np

from keras.models import load_model
from scipy.misc import imresize


CURRENT_FRAME = np.full((450, 450), 255, dtype=np.uint8)
LABELS_CHR = np.load("labels_name.npy")
DRAW_MODE = False

def mouse_event(event, x, y, flags, params):
    global CURRENT_FRAME, DRAW_MODE

    if event == cv.EVENT_LBUTTONDOWN or event == cv.EVENT_LBUTTONUP:
        DRAW_MODE = not(DRAW_MODE)

    elif event == cv.EVENT_MOUSEMOVE:
        if DRAW_MODE:
            cv.circle(CURRENT_FRAME, (x, y), 10, 0, -1)

def process_image():
    return np.array(1/255 * imresize(CURRENT_FRAME, (45, 45))).reshape((1, 45, 45, 1))

def main():
    global CURRENT_FRAME

    model = load_model("cnn_mathsym_v2.h5")

    window_name = "Frame"
    cv.namedWindow(window_name)
    cv.setMouseCallback(window_name, mouse_event) 

    while(1):
        cv.imshow(window_name, CURRENT_FRAME)
        key = cv.waitKey(1)

        if key == ord('q'):
            break
        elif key == ord('p'):
            pred = model.predict_on_batch(process_image())
            pred_chr = LABELS_CHR[pred.argmax()
                    ]
            print("Predicted => {}".format(pred_chr))
            print("Confidence => {0:.2f}%".format(100 * pred.max()))
        elif key == ord('c'):
            CURRENT_FRAME = np.full((450, 450), 255, dtype=np.uint8)


    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
