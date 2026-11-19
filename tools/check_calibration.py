"""
Find which clicked keypoint is hurting your court calibration.

Usage (from the repository folder):
    uv run python tools/check_calibration.py videos/match_calibration.json

For every point you clicked it shows:
  fit error   - how far the point lands from where it should, using the
                saved calibration (in metres on the court)
  left-out    - the same, but with a calibration computed WITHOUT that point.
                This is the better test: a bad click pulls the fit towards
                itself, which hides its own error. A bad point stands out here.
"""

import json
import sys

import cv2
import numpy as np

from courtshot.court import COURT_MODEL


def project(H: np.ndarray, px: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(px.reshape(-1, 1, 2), H).reshape(-1, 2)


def main(path: str) -> None:
    with open(path) as f:
        cal = json.load(f)
    names = list(cal["labeled_keypoints"])
    px = np.array([cal["labeled_keypoints"][n] for n in names], dtype=np.float64)
    m = np.array([COURT_MODEL.KEYPOINTS[n] for n in names], dtype=np.float64)
    H = np.array(cal["homography_matrix"], dtype=np.float64)

    fit_err = np.linalg.norm(project(H, px) - m, axis=1)

    loo_err = np.full(len(names), np.nan)
    if len(names) >= 5:
        for i in range(len(names)):
            keep = np.arange(len(names)) != i
            H_i, _ = cv2.findHomography(px[keep], m[keep], 0)  # plain least squares
            if H_i is not None:
                loo_err[i] = np.linalg.norm(project(H_i, px[i : i + 1])[0] - m[i])

    print(f"{len(names)} points, saved reprojection error {cal.get('reprojection_error', 0):.3f} m\n")
    print(f"{'point':<22}{'fit error':>11}{'left-out':>11}")
    order = np.argsort(-np.nan_to_num(loo_err, nan=-1) if len(names) >= 5 else -fit_err)
    for i in order:
        loo = "    n/a" if np.isnan(loo_err[i]) else f"{loo_err[i]:9.3f} m"
        print(f"{names[i]:<22}{fit_err[i]:9.3f} m{loo}")

    if len(names) < 5:
        print("\nOnly 4 points: the fit is exact, so these numbers say nothing. Click more points.")
        return
    worst = int(np.nanargmax(loo_err))
    others = np.delete(loo_err, worst)
    if loo_err[worst] > 2 * np.nanmedian(others) and loo_err[worst] > 0.2:
        print(f"\n'{names[worst]}' stands out: re-click it, or skip it with N.")
    elif np.nanmedian(loo_err) > 0.2:
        print("\nAll points are off by a similar amount. That points to lens distortion")
        print("or blurry lines rather than one bad click (see notes).")
    else:
        print("\nNo single bad point. This calibration is about as good as clicking gets.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "videos/match_calibration.json")
