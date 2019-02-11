import logging

import cv2
import timeit
import numpy as np
import holo_detector
import utils
import scipy.stats
from matplotlib import pyplot as plt

logger = logging.getLogger()


def main():
    infile = 'data/video1.mp4'
    outfile = 'out.mp4'
    w, h = 1080, 640
    rotate = False
    cap = cv2.VideoCapture(infile)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    cap.set(cv2.CAP_PROP_FOCUS, 0)

    fps = cap.get(cv2.CAP_PROP_FPS)
    res = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info('Video filename: {0}'.format(infile))
    logger.info('Video FPS: {0}'.format(fps))
    logger.info('File length: {0} frames'.format(length))

    detector = holo_detector.HoloDetector(debug=True)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(outfile, fourcc, round(fps), (w, h) if not rotate else (h, w))
    frame_number = 0
    cur_time = timeit.default_timer()
    while cap.isOpened():
        (ret, frame) = cap.read()
        if frame is None:
            break
        frame_number += 1
        if rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        holo_mask, img_holo = detector.detect_holos(frame, False)
        if img_holo is not None:
            out_img = utils.show(img_holo, ref_imgs=list(detector.dbg_images.values()), cvshow=True)
        cv2.waitKey(10)
        out.write(img_holo)

    out.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
