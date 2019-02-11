from matplotlib import pyplot as plt
import numpy as np
import numpy.linalg as la
import cv2

color_pallete = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255], [255, 255, 0], [255, 0, 255]]


def angle(v1, v2):
    """
    Calculates angle between vectors in radians
    :param v1:
    :param v2:
    :return:
    """
    c = np.dot(v1, v2)
    s = la.norm(np.cross(v1, v2))
    return np.arctan2(s, c)


def closest(points, x, y, n=10):
    """
    Returns N closest points
    :param points:
    :param x:
    :param y:
    :param n:
    :return:
    """
    dist = (points[:, 0] - x) ** 2 + (points[:, 1] - y) ** 2
    return points[dist.argsort()[:n]]


def show(img=None, highlight=False, title='', axes=False, ref_imgs=None, cvshow=False):
    """
    Displays a main image along with a list of reference images converting colors and size as required
    :param img:
    :param highlight:
    :param title:
    :param axes:
    :param ref_imgs:
    :param cvshow:
    :return:
    """
    if img is None and len(ref_imgs) > 0:
        img = ref_imgs.pop()
    src_img = img
    if highlight:
        img = np.zeros(shape=(img.shape[0], img.shape[1], 3), dtype=np.uint8)
        colors = np.unique(img)
        for i, c in enumerate([clr for clr in colors if clr != 0]):
            img[np.where((src_img == c))] = color_pallete[i]
    if ref_imgs:
        refs = []
        for im in ref_imgs:
            if len(im.shape) == 2 or im.shape[2] == 1:
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            im = cv2.resize(im, (src_img.shape[1], src_img.shape[0]))
            refs.append(im)
        if img is not None:
            if len(img.shape) == 2 or img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            refs.append(img)
        img = np.vstack(refs)
    if len(img.shape) == 3 and img.shape[2] == 3 and not cvshow:
        # BGR to RGB
        img = img[..., ::-1]
    # resize to fit screen
    img = cv2.resize(img, (int(img.shape[1]*(900/img.shape[0])), 900))

    if not cvshow:
        plt.imshow(img, cmap=('gist_gray' if len(img.shape) == 2 or img.shape[2] == 1 else None))
        plt.title(title)
        if not axes:
            plt.axes().get_xaxis().set_visible(False)
            plt.axes().get_yaxis().set_visible(False)
        plt.show()
    else:
        cv2.imshow('win1', img)
        cv2.moveWindow('win1', 100, 20)
    return img
