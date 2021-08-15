import logging
import os.path
import time
from typing import Optional, Tuple

import cv2
import numpy as np

logging.basicConfig(
    level=logging.DEBUG
)

LOGGER = logging.getLogger('MiceLick')

CURRENT_WINDOW_HANDLE = None

COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)


def select_roi(t: int, frame: np.ndarray) -> np.ndarray:
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


class Main:
    def __init__(self, video_file: str):
        if not os.path.exists(video_file):
            raise FileNotFoundError(video_file)

        # file
        self.video_file = video_file
        self.output_file: Optional[str] = None
        self.roi_use_file: Optional[str] = None
        self.roi_output_file: Optional[str] = None

        # property
        self.window_title = 'MouseLicking'

        # video property
        self.video_capture: cv2.VideoCapture = None
        self.video_width: int = 0
        self.video_height: int = 0
        self.video_fps: int = 1
        self.video_total_frames: int = 0
        self.current_image_original = None
        self.current_image = None

        # control
        self._speed_factor: float = 1
        self._sleep_interval = 1
        self._is_playing = False
        self.show_time = True
        self.buffer = ''

    @property
    def speed_factor(self) -> float:
        return self._speed_factor

    @speed_factor.setter
    def speed_factor(self, value: float):
        value = min(32.0, max(0.25, value))
        self._speed_factor = value
        self._sleep_interval = 1 / self.video_fps / value
        LOGGER.debug(f'speed = {value}')

    @property
    def current_frame(self) -> int:
        vc = self.video_capture
        if vc is None:
            raise RuntimeError()
        return int(vc.get(cv2.CAP_PROP_POS_FRAMES))

    @current_frame.setter
    def current_frame(self, value: int):
        vc = self.video_capture
        if vc is None:
            raise RuntimeError()

        if not (0 <= value < self.video_total_frames):
            raise ValueError()

        vc.set(cv2.CAP_PROP_POS_FRAMES, value)

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    @is_playing.setter
    def is_playing(self, value: bool):
        self._is_playing = value
        if value:
            LOGGER.debug('play')
        else:
            LOGGER.debug('pause')

    def start(self, pause_on_start=False):
        LOGGER.debug(f'file = {self.video_file}')
        self.video_capture = vc = cv2.VideoCapture(self.video_file)
        self.video_width = w = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = h = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
        LOGGER.debug(f'width,height = {w},{h}')

        self.video_fps = fps = int(vc.get(cv2.CAP_PROP_FPS))
        LOGGER.debug(f'fps = {fps}')
        self.speed_factor = self._speed_factor  # update sleep_interval

        self.video_total_frames = f = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
        LOGGER.debug(f'total_frame = {f}')

        cv2.namedWindow(self.window_title, cv2.WINDOW_GUI_NORMAL)

        try:
            self._is_playing = not pause_on_start
            self._loop()
        except KeyboardInterrupt:
            pass
        finally:
            vc.release()
            cv2.destroyWindow(self.window_title)

    def _loop(self):
        while True:
            try:
                self._update()
            except KeyboardInterrupt:
                raise
            except BaseException as e:
                print(e)

    def _update(self):
        vc = self.video_capture
        if self._is_playing or self.current_image_original is None:
            ret, image = vc.read()
            if not ret:
                self._is_playing = False
                return

            self.current_image_original = image

        self.current_image = self.current_image_original.copy()

        if len(self.buffer):
            self._show_buffer()

        if self.show_time:
            self._show_time_bar()

        cv2.imshow(self.window_title, self.current_image)
        time.sleep(self._sleep_interval)

        k = cv2.waitKey(1)
        if k > 0:
            self.handle_key_event(k)

    def _show_buffer(self):
        buffer = self.buffer
        cv2.putText(self.current_image, buffer, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_RED, 2, cv2.LINE_AA)

    def _show_time_bar(self):
        w = self.video_width
        h = self.video_height
        frame = self.current_frame

        t_sec = self.video_total_frames // self.video_fps
        t_min, t_sec = t_sec // 60, t_sec % 60
        t_text = f'{t_min:02d}:{t_sec:02d}'
        cv2.putText(self.current_image, t_text, (w - 100, h), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_RED, 2, cv2.LINE_AA)

        t_sec = frame // self.video_fps
        t_min, t_sec = t_sec // 60, t_sec % 60
        t_text = f'{t_min:02d}:{t_sec:02d}'
        cv2.putText(self.current_image, t_text, (10, h), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_RED, 2, cv2.LINE_AA)

        s = 130
        cv2.line(self.current_image, (s, h - 10), (w - s, h - 10), COLOR_RED, 3, cv2.LINE_AA)
        x = int((w - 2 * s) * frame / self.video_total_frames) + s
        cv2.line(self.current_image, (x, h - 20), (x, h), COLOR_RED, 3, cv2.LINE_AA)

    def _print_key_shortcut(self):
        print('h       : print key shortcut help')
        print('q       : quit program')
        print('+/-     : in/decrease video playing speed')
        print('t       : show/hide progress bar')
        print('<space> : play/pause')
        print('0~9.    : input number')
        print('<backspace> : delete input number')
        print('c       : clear buffer')
        print('<num>j  : jump to time')

    def handle_key_event(self, k: int):
        if k == ord('q'):
            raise KeyboardInterrupt
        elif k == ord('h'):
            self._print_key_shortcut()
        elif k == ord('+'):
            self.speed_factor *= 2
        elif k == ord('-'):
            self.speed_factor /= 2
        elif k == ord('t'):
            self.show_time = not self.show_time
        elif k == ord(' '):
            self.is_playing = not self.is_playing
        elif 48 <= k < 58:  # 0-9
            self.buffer += '0123456789'[k - 48]
        elif k == ord('.'):
            self.buffer += '.'
        elif k == 8:  # backspace
            if len(self.buffer) > 0:
                self.buffer = self.buffer[:-1]
        elif k == ord('c'):
            self.buffer = ''
        elif k == ord('j'):
            if len(self.buffer) > 0:
                try:
                    t_min, t_sec = _decode_buffer_as_time(self.buffer)
                except BaseException as e:
                    print(e)
                else:
                    frame = (t_min * 60 + t_sec) * self.video_fps
                    self.current_frame = frame
                    LOGGER.debug(f'jump to {t_min:02d}:{t_sec:02d} (frame={frame})')
                    self.buffer = ''

        else:
            LOGGER.debug(f'key_input : {k}')


def _decode_buffer_as_time(buffer: str) -> Tuple[int, int]:
    if '.' in buffer:
        t_min, t_sec = buffer.split('.')
        return int(t_min), int(t_sec)
    else:
        t_sec = int(buffer)
        return t_sec // 60, t_sec % 60


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

    main = Main(opt.FILE)
    main.output_file = opt.output
    main.roi_use_file = opt.use_roi
    main.roi_output_file = opt.save_roi

    main.start()
