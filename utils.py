import numpy as np
import cv2

# from models.utils import resize_image


def crop(frame, roi):
    p1 = roi.position.astype(int)
    p1 = np.clip(p1, [0, 0], [frame.shape[1], frame.shape[0]])
    p2 = (roi.position + roi.size).astype(int)
    p2 = np.clip(p2, [0, 0], [frame.shape[1], frame.shape[0]])
    return frame[p1[1]:p2[1], p1[0]:p2[0]]


def cut_rois(frame, rois):
    return [crop(frame, roi) for roi in rois]


def resize_input(image, target_shape):
    _, _, h, w = target_shape
    # resized_image = resize_image(image, (w, h))
    resized_image = cv2.resize(image, dsize=(w,h))
    resized_image = resized_image.transpose((2, 0, 1)) # HWC->CHW
    resized_image = resized_image.reshape(target_shape)
    return resized_image

