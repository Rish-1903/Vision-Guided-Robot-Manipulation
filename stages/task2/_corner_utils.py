"""Shared checkerboard corner detector for Task 2."""
import cv2
import numpy as np
from make_checkerboard import INNER_CORNERS


def detect_corners(bgr: np.ndarray):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
    ok, corners = cv2.findChessboardCornersSB(gray, INNER_CORNERS, flags=flags)
    if ok:
        return corners.astype(np.float32)
    ok, corners = cv2.findChessboardCorners(
        gray, INNER_CORNERS,
        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if ok:
        corners = cv2.cornerSubPix(
            gray, corners, (7, 7), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, .001),
        )
        return corners.astype(np.float32)
    return None
