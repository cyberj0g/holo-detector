import itertools

import cv2
import numpy as np
import numpy.linalg as la
import scipy.stats
from utils import closest, angle


class HoloDetector:
    TRAIN_FRAMES = 10
    NUM_CANDIDATES = 5
    MIN_HOLOS = 50
    HOLO_THRESHOLD = 80
    SCALE_FACTOR = 0.3
    UNIFORMITY_THRESHOLD = 100
    lk_params = dict(winSize=(19, 19),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def __init__(self, debug=False):
        self.debug = debug
        self.dbg_images = {}
        self.stack = []
        self.cur_img = None
        self.rect_areas = []
        self.last_features = None
        self.train_features = None
        self.is_training = True
        self.last_rect = None
        self.last_b_rect = None
        self.last_gray = None
        self.holo_stack = None
        self.cur_stack_idx = 0
        self.holo_mask = None
        self.holo_res = None
        self.stack_filled = False
        self.holo_detected = False

    @staticmethod
    def checkedTrace(img0, img1, p0, back_threshold=1.0):
        p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **HoloDetector.lk_params)
        p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **HoloDetector.lk_params)
        d = abs(p0 - p0r).reshape(-1, 2).max(-1)
        status = d < back_threshold
        return p1, status

    def calc_holo_points(self):
        # quantile range for holo pixels on H component is expected to be much wider
        qr = np.quantile(self.holo_stack[:, :, 0, :], q=0.95, axis=2) - np.quantile(self.holo_stack[:, :, 0, :], q=0.05, axis=2)
        # Saturation and Value thresholds because on lower values H component may be unstable
        ms = np.mean(self.holo_stack[:, :, 1, :], axis=2)
        mv = np.mean(self.holo_stack[:, :, 2, :], axis=2)
        filtered_points = []
        holo_points = np.where((ms > 50) & (mv > 50) & (qr > HoloDetector.HOLO_THRESHOLD))
        # filter detected pixels by uniformity of their distribution, holo points are taking multiple colors,
        # while misaligned edge pixels will have only few different values
        for y, x in zip(*holo_points):
            freq = np.histogram(self.holo_stack[y, x, 0, :], bins=20, range=(0, 255))[0]
            # checks for uniformity without expected frequencies parameter
            chi, _ = scipy.stats.chisquare(freq)
            if chi < HoloDetector.UNIFORMITY_THRESHOLD:
                filtered_points.append((y, x))
        # highlight pixels on mask
        self.holo_mask[tuple(zip(*filtered_points))] = (0, 255, 0)

    def detect_rect(self, gray):
        """
        Detects large rectangular shape on the image
        :param gray:
        :return:
        """
        # get corners
        features = cv2.goodFeaturesToTrack(gray, 500, 0.01, 10)
        corners = features.squeeze()
        # get some number of corners closest to corresponding frame corners
        corner_candidates = list(map(lambda p: closest(corners, p[0], p[1], HoloDetector.NUM_CANDIDATES),
                                     ((0, 0), (0, gray.shape[0]), (gray.shape[1], gray.shape[0]), (gray.shape[1], 0))))
        # check for rectangularity and get a maximum area rectangle
        combs = itertools.product(*corner_candidates)
        max_rect = None
        max_area = 0
        for c1, c2, c3, c4 in combs:
            angles = [angle(c1 - c2, c3 - c2),
                      angle(c2 - c3, c4 - c3),
                      angle(c1 - c4, c3 - c4)]
            if np.allclose(angles, np.pi / 2, rtol=0.05):
                area = la.norm(c2 - c1) * la.norm(c3 - c2)
                if area > max_area:
                    max_rect = [c1, c2, c3, c4]
                    max_area = area
        if self.debug:
            self.dbg_images['corners'] = self.cur_img.copy()
            for c in range(4):
                # draw candidates
                if corner_candidates:
                    list(map(lambda p: cv2.circle(self.dbg_images['corners'], tuple(p), 4, (0, 0, 255), 4), corner_candidates[c][:HoloDetector.NUM_CANDIDATES]))
                # draw selected rect
                if max_rect:
                    cv2.circle(self.dbg_images['corners'], tuple(max_rect[c]), 7, (0, 255, 0), 4)
        return max_rect, max_area

    def lock_rect(self, rect, gray):
        self.is_training = False
        # get start keypoints inside rectangle
        features = cv2.goodFeaturesToTrack(gray, 1000, 0.01, 19)
        rect_contour = np.array(rect).astype(np.int32)
        # take only points inside rectangle area
        self.last_features = np.array(list(filter(lambda p: cv2.pointPolygonTest(rect_contour, tuple(p.squeeze()), False), features)))
        self.train_features = self.last_features.copy()
        self.last_gray = gray
        self.last_rect = rect
        self.last_b_rect = cv2.boundingRect(np.array(self.last_rect).astype(np.int32))

    def detect_holos(self, img, search_rect=False):
        """
        Returns the mask with hologram pixel values > 0, and input image with overlayed holo mask
        :param img:
        :param search_rect: whether to wait for a stable rectangle before starting holo detection
        :return:
        """
        self.dbg_images.clear()
        holo_mask_t = None
        self.cur_img = img
        img_holo = img
        # convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.is_training:
            if search_rect:
                # search for a rectangular area
                rect, area = self.detect_rect(gray)
                if rect is not None:
                    self.rect_areas.append(area)
                    self.last_rect = rect
                if len(self.rect_areas) > HoloDetector.TRAIN_FRAMES:
                    # check if rectangle is stable
                    if np.std(self.rect_areas) < 0.005 * np.mean(self.rect_areas):
                        self.lock_rect(rect, gray)
                    else:
                        self.reset()
            else:
                # just use full image if rectangle tracking is disabled
                rect = [(1, 1), (img.shape[1] - 1, 1), (img.shape[1] - 1, img.shape[0] - 1),
                        (1, img.shape[0] - 1)]
                self.lock_rect(rect, gray)
        else:
            # calculate optical flow with cross check
            features, status = HoloDetector.checkedTrace(self.last_gray, gray, self.last_features)
            # filter only cross-checked features
            self.last_features = self.last_features[status]
            self.train_features = self.train_features[status]
            features = features[status]
            # not enough features - reset
            if len(features) < 4:
                self.reset()
                return None, None
            # estimate transformation matrix
            m, mask = cv2.findHomography(features, self.train_features, cv2.RANSAC, 10.0)
            if m is None:
                self.reset()
                return None, None
            # unwarp image into original image coordinates
            unwarped = img.copy()
            unwarped = cv2.warpPerspective(unwarped, m, img.shape[:2][::-1], flags=cv2.INTER_LINEAR)
            unwarped = unwarped[self.last_b_rect[1]:(self.last_b_rect[1] + self.last_b_rect[3]), self.last_b_rect[0]:(self.last_b_rect[0] + self.last_b_rect[2])]
            # convert to HSV colors for analysis
            unwarped_hsv = cv2.cvtColor(unwarped, cv2.COLOR_BGR2HSV)
            if self.holo_stack is None:
                # pre-initialize an array to store image stack
                self.holo_res = int(unwarped.shape[1] * HoloDetector.SCALE_FACTOR), int(unwarped.shape[0] * HoloDetector.SCALE_FACTOR)
                self.holo_stack = np.zeros(shape=self.holo_res[::-1] + (3, HoloDetector.MIN_HOLOS,))
                self.holo_mask = np.zeros(shape=self.holo_stack.shape[:3], dtype=np.uint8)
            # image is resized to speed up processing
            self.holo_stack[..., self.cur_stack_idx] = cv2.resize(unwarped_hsv, self.holo_res, cv2.INTER_CUBIC)
            self.cur_stack_idx += 1
            if self.cur_stack_idx == HoloDetector.MIN_HOLOS:
                self.stack_filled = True
                self.cur_stack_idx = 0
            # run detection only once image stack is filled
            if self.stack_filled:
                self.calc_holo_points()
                # clear holo stack, detected hologram pixels are preserved
                self.holo_stack = np.zeros(shape=self.holo_res[::-1] + (3, HoloDetector.MIN_HOLOS,))
                self.stack_filled = False
            fs_mask = cv2.resize(self.holo_mask, unwarped.shape[:2][::-1], cv2.INTER_LINEAR_EXACT)
            # save unwarped image for demonstration
            self.dbg_images['unwarp'] = cv2.addWeighted(unwarped, 1, fs_mask, 10, 0.0)
            # warp holo mask back to overlay on original image
            fs_mask_warped = cv2.warpPerspective(fs_mask, m, img.shape[:2][::-1], flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
            img_holo = cv2.addWeighted(img, 1, fs_mask_warped, 10, 0.0)
            self.last_gray = gray
            self.last_features = features
        return holo_mask_t, img_holo

    def reset(self):
        self.rect_areas.clear()
        self.holo_stack = None
        self.holo_res = None
        self.stack_filled = False
        self.is_training = True
        self.cur_stack_idx = 0
