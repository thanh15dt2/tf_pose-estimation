import sys
import cv2
import numpy as np
from PIL import Image
from infer import EventScoring
scoring = EventScoring()
path = '/home/dsoft/Music/me/project/tf-pose-estimation/image_test_copy'

def border_resize(image, output_size=360):

    frame_size = tuple(image.shape[:2])
    ratio = output_size / max(frame_size)
    new_size = tuple([int(x * ratio) for x in frame_size])
    delta_w = output_size - new_size[1]
    delta_h = output_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    resized = cv2.resize(image, (new_size[1], new_size[0]))

    b_img = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=[0.406 * 255, 0.456 * 255, 0.485 * 255])  # ImageNet means (BGR)

    b_img_rgb = cv2.cvtColor(b_img, cv2.COLOR_BGR2RGB)
    cv2.imshow('image', b_img)
    cv2.waitKey(0)
    return b_img

if __name__ == "__main__":

    # path_a = '/home/dsoft/Music/me/project/image_true/7.png'
    # img = cv2.imread(path_a)
    # size = tuple(img.shape[:2])
    # image = border_resize(img)
    # cv2.imwrite(path_a, image)

    scoring.score_infer(path)




