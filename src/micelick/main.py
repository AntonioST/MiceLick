import logging
import os.path
import sys
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
COLOR_YELLOW = (0, 255, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)
COLOR_WHITE = (0, 0, 0)


def calculate_value(video, mask: np.ndarray) -> np.ndarray:
    pass


def calculate_licking(t: np.ndarray, value: np.ndarray) -> np.ndarray:
    pass


def save_licking_result(file: str, data: np.ndarray):
    pass


class Main:
    MOUSE_STATE_FREE = 0
    MOUSE_STATE_MASKING = 1
    MOUSE_STATE_MASKED = 2

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
        self.mask: np.ndarray = np.zeros((0, 5), dtype=int)
        self.show_time = True
        self.show_mask = True
        self.mouse_stick_to_mask = True
        self.mouse_stick_distance = 5
        self.message_fade_time = 5

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
        self._current_operation_state = self.MOUSE_STATE_FREE
        self._current_mouse_hover_frame = None
        self._current_mouse_button = None
        self._current_masking_region = None
        self._message_queue = []
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

    def enqueue_message(self, text: str):
        self._message_queue.append((time.time(), text))

    @property
    def mask_count(self) -> int:
        return self.mask.shape[0]

    @property
    def current_mask(self) -> Optional[np.ndarray]:
        f = self.current_frame
        ret = None
        for mask in self.mask:
            t = mask[0]
            if t < f:
                ret = mask
        return ret

    def add_mask(self, x0: int, y0: int, x1: int, y1: int, t: int = None):
        if t is None:
            t = self.current_frame

        self.mask = np.sort(np.append(self.mask, [(t, x0, y0, x1, y1)], axis=0), axis=0)
        LOGGER.info(f'add mask [{t},{x0},{y0},{x1},{y1}]')

    def del_mask(self, index: int):
        t, x0, y0, x1, y1 = self.mask[index]
        self.mask = np.delete(self.mask, index, axis=0)
        LOGGER.info(f'del mask [{t},{x0},{y0},{x1},{y1}]')

    def clear_mask(self):
        self.mask = np.zeros((0, 5), dtype=int)

    def load_mask(self, file: str):
        LOGGER.debug(f'load mask = {file}')
        ret: np.ndarray = np.load(file)
        # shape: (N, (time, x, y, width, height))
        if not (ret.ndim == 2 and ret.shape[1] == 5):
            raise RuntimeError('not a roi mask')
        self.mask = ret

    def save_mask(self, file: str):
        np.save(file, self.mask)

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
        cv2.setMouseCallback(self.window_title, self.handle_mouse_event)

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
                LOGGER.warning(e)
                # raise

    def _update(self):
        vc = self.video_capture
        if self._is_playing or self.current_image_original is None:
            ret, image = vc.read()
            if not ret:
                self._is_playing = False
                return

            self.current_image_original = image

        self.current_image = self.current_image_original.copy()

        self._show_mouse_operator_text()
        self._show_queued_message()

        if len(self.buffer):
            self._show_buffer()

        if self._current_masking_region is not None:
            self._show_mask_tmp()
        elif self.show_mask:
            mask = self.current_mask
            if mask is not None:
                self._show_mask(mask)

        if self.show_time:
            self._show_time_bar()

        cv2.imshow(self.window_title, self.current_image)
        time.sleep(self._sleep_interval)

        k = cv2.waitKey(1)
        if k > 0:
            self.handle_key_event(k)

    def _show_mouse_operator_text(self):
        if self._current_operation_state == self.MOUSE_STATE_FREE:
            pass
        elif self._current_operation_state == self.MOUSE_STATE_MASKING:
            cv2.putText(self.current_image, 'select ROI', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_RED, 2, cv2.LINE_AA)

    def _show_queued_message(self):
        t = time.time()
        y = 70
        s = 25
        i = 0
        while i < len(self._message_queue):
            r, m = self._message_queue[i]
            if r + self.message_fade_time < t:
                del self._message_queue[i]
            else:
                cv2.putText(self.current_image, m, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_RED, 2, cv2.LINE_AA)
                i += 1
                y += s

    def _show_buffer(self):
        h = self.video_height
        buffer = self.buffer
        cv2.putText(self.current_image, buffer, (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_RED, 2, cv2.LINE_AA)

    def _show_mask(self, mask: np.ndarray):
        _, x0, y0, x1, y1 = mask
        cv2.rectangle(self.current_image, (x0, y0), (x1, y1), COLOR_YELLOW, 2, cv2.LINE_AA)

    def _show_mask_tmp(self):
        x0, y0, x1, y1 = self._current_masking_region
        cv2.rectangle(self.current_image, (x0, y0), (x1, y1), COLOR_GREEN, 2, cv2.LINE_AA)

    def _show_time_bar(self):
        s = 130
        w = self.video_width
        h = self.video_height
        frame = self.current_frame

        # total frame text
        cv2.putText(self.current_image, self._frame_to_text(self.video_total_frames), (w - 100, h),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_RED, 2, cv2.LINE_AA)

        # current frame text
        cv2.putText(self.current_image, self._frame_to_text(frame), (10, h),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_RED, 2, cv2.LINE_AA)

        #
        # mask frame
        mx = self._frame_to_time_bar_x(self.mask[:, 0])
        for x in mx:
            cv2.line(self.current_image, (x, h - 20), (x, h), COLOR_YELLOW, 3, cv2.LINE_AA)

        # current frame
        x = self._frame_to_time_bar_x(frame)
        cv2.line(self.current_image, (x, h - 20), (x, h), COLOR_RED, 3, cv2.LINE_AA)

        # mouse hover
        if self._current_mouse_hover_frame is not None:
            t = self._current_mouse_hover_frame
            x = self._frame_to_time_bar_x(t)

            color = COLOR_GREEN
            if self.mouse_stick_to_mask and len(mx) > 0:
                i = np.argmin(np.abs(mx - x))
                if abs(mx[i] - x) < self.mouse_stick_distance:
                    x = mx[i]
                    t = self.mask[i, 0]
                    color = COLOR_YELLOW

            cv2.line(self.current_image, (x, h - 20), (x, h), color, 3, cv2.LINE_AA)

            # text
            cv2.putText(self.current_image, self._frame_to_text(t), (x - s // 2, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        # line
        cv2.line(self.current_image, (s, h - 10), (w - s, h - 10), COLOR_RED, 3, cv2.LINE_AA)

    def _set_mouse_hover_frame(self, x, y):
        w = self.video_width
        h = self.video_height
        t = self.video_total_frames
        s = 130

        if (h - 20 <= y <= h) and (s <= x <= w - s):
            self._current_mouse_hover_frame = int((x - s) / (w - 2 * s) * t)
        else:
            self._current_mouse_hover_frame = None

    def _frame_to_time_bar_x(self, frame):
        w = self.video_width
        t = self.video_total_frames
        s = 130

        if isinstance(frame, (int, float)):
            return int((w - 2 * s) * frame / t) + s
        elif isinstance(frame, np.ndarray):
            return ((w - 2 * s) * frame.astype(float) / t).astype(int) + s
        else:
            raise TypeError()

    def _frame_to_text(self, frame: int):
        t_sec = frame // self.video_fps
        t_min, t_sec = t_sec // 60, t_sec % 60
        return f'{t_min:02d}:{t_sec:02d}'

    def _print_key_shortcut(self):
        print('h       : print key shortcut help')
        print('q       : quit program')
        print('+/-     : in/decrease video playing speed')
        print('t       : show/hide progress bar')
        print('m       : show/hide masks')
        print('n       : print all masks')
        print('s       : save masks')
        print('<space> : play/pause')
        print('0~9.    : input number')
        print('<backspace> : delete input number')
        print('c       : clear buffer')
        print('<num>j  : jump to time')
        print('<num>d  : delete mask')
        print('dd      : delete all maskes')

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
        elif k == ord('m'):
            self.show_mask = not self.show_mask
        elif k == ord('n'):
            np.savetxt(sys.stdout, self.mask, fmt='%d', delimiter='\t',
                       header='\t'.join(['time', 'x0', 'y0', 'x1', 'x2']))
        elif k == ord('s'):
            if self.roi_output_file is not None:
                self.save_mask(self.roi_output_file)
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
                t_min, t_sec = _decode_buffer_as_time(self.buffer)
                frame = (t_min * 60 + t_sec) * self.video_fps
                self.current_frame = frame
                LOGGER.debug(f'jump to {t_min:02d}:{t_sec:02d} (frame={frame})')
                self.buffer = ''
        elif k == ord('d'):
            if self.buffer == 'd':
                self.clear_mask()
                self.buffer = ''
            elif len(self.buffer) > 0:
                self.del_mask(int(self.buffer))
                self.buffer = ''
            else:
                self.buffer = 'd'

        else:
            LOGGER.debug(f'key_input : {k}')

    def handle_mouse_event(self, event: int, x: int, y: int, flag: int, data):
        if event == cv2.EVENT_MOUSEMOVE:
            if self._current_operation_state == self.MOUSE_STATE_MASKING:
                x0, y0, _, _ = self._current_masking_region
                self._current_masking_region = [x0, y0, x, y]
            else:
                if self.show_time:
                    self._set_mouse_hover_frame(x, y)
                else:
                    self._current_mouse_hover_frame = None

        elif event == cv2.EVENT_LBUTTONUP:
            if self._current_operation_state == self.MOUSE_STATE_FREE:
                if self._current_mouse_hover_frame is not None:
                    self.current_frame = self._current_mouse_hover_frame
                else:
                    self.is_playing = not self.is_playing

        elif event == cv2.EVENT_RBUTTONDOWN:
            self._current_operation_state = self.MOUSE_STATE_MASKING
            self._current_masking_region = [x, y, x, y]
            self.is_playing = False

        elif event == cv2.EVENT_RBUTTONUP:
            if self._current_operation_state == self.MOUSE_STATE_MASKING:
                t = self.current_frame
                n = self.mask_count
                self.add_mask(*self._current_masking_region)
                self.enqueue_message(f'add mask[{n}] at ' + self._frame_to_text(t))
                self._current_masking_region = None
                self._current_operation_state = self.MOUSE_STATE_FREE
            else:
                self._current_operation_state = self.MOUSE_STATE_FREE


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
