from typing import Optional

import numpy as np


def select_roi(frame: np.ndarray) -> np.ndarray:
    pass


def calculate_value(video, mask: np.ndarray) -> np.ndarray:
    pass


def calculate_licking(t: np.ndarray, value: np.ndarray) -> np.ndarray:
    pass


def load_roi_mask(file: str) -> np.ndarray:
    ret: np.ndarray = np.load(file)
    # shape: (N, (time, x, y, width, height))
    if not (ret.ndim == 2 and ret.shape[1] == 5):
        raise RuntimeError('not a roi mask')
    return ret


def save_roi_mask(file: str, mask: np.ndarray):
    np.save(file, mask)


def save_licking_result(file: str, data: np.ndarray):
    pass


def main(video_file: str,
         output_file: Optional[str] = None,
         roi_use_file: Optional[str] = None,
         roi_output_file: Optional[str] = None):
    pass


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument('--roi',
                    metavar='FILE',
                    default=None,
                    help='use ROI file',
                    dest='use_roi')
    ap.add_argument('--save-mask',
                    metavar='FILE',
                    default=None,
                    help='save roi mask usage',
                    dest='save_roi')
    ap.add_argument('-o', '--output', '--output-data-path',
                    metavar='FILE',
                    default=None,
                    help='save licking result',
                    dest='output')
    ap.add_argument('FILE')
    opt = ap.parse_args()
    main(opt.FILE, opt.output, opt.use_roi, opt.save_roi)
