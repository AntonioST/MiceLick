import logging
import os.path
import threading
import time
from typing import Optional, Tuple, List, Callable

import cv2
import numpy as np

logging.basicConfig(
    level=logging.DEBUG
)

LOGGER = logging.getLogger('MiceLick')


COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)
COLOR_WHITE = (0, 0, 0)
COLOR_BLACK = (255, 255, 255)


def rectangle_to_mask(w: int, h: int, roi: np.ndarray) -> np.ndarray:
    ret = np.zeros((h, w, 3), dtype=np.uint8)
    _, x0, y0, x1, y1 = roi
    cv2.rectangle(ret, (x0, y0), (x1, y1), COLOR_BLACK, -1, cv2.LINE_AA)
    return ret


class Main:
    MOUSE_STATE_FREE = 0
    MOUSE_STATE_MASKING = 1

    def __init__(self, video_file: str):
        if not os.path.exists(video_file):
            raise FileNotFoundError(video_file)

        # file
        self.video_file = video_file
        self.output_file: Optional[str] = None
        self.roi_use_file: Optional[str] = None
        self.roi_output_file: Optional[str] = None

        # lick properties
        self.current_value = 0
        self.lick_possibility: np.ndarray = None

        # display properties
        self.window_title = 'MouseLicking'
        self.roi: np.ndarray = np.zeros((0, 5), dtype=int)
        self.show_time = True
        self.show_lick = True
        self.show_lick_duration = 30
        self.show_roi = True
        self.eval_lick = True
        self.mouse_stick_to_roi = True
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
        self._current_mouse_hover_frame: Optional[int] = None
        self._current_roi_region: Optional[List[int]] = None
        self._message_queue: List[Tuple[float, str]] = []
        self._mask_cache: Optional[Tuple[int, np.ndarray, np.ndarray]] = None
        self._eval_task = None
        self.buffer = ''

    def _check_task(self) -> bool:
        return self._eval_task is None

    @property
    def speed_factor(self) -> float:
        return self._speed_factor

    @speed_factor.setter
    def speed_factor(self, value: float):
        value = min(32.0, max(0.25, value))
        self._speed_factor = value
        self._sleep_interval = 1 / self.video_fps / value
        self.enqueue_message(f'x{value}')
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
        self.current_image_original = None

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    @is_playing.setter
    def is_playing(self, value: bool):
        if self._check_task():
            self._is_playing = value
            if value:
                LOGGER.debug('play')
            else:
                LOGGER.debug('pause')

    def enqueue_message(self, text: str):
        self._message_queue.append((time.time(), text))

    @property
    def roi_count(self) -> int:
        return self.roi.shape[0]

    @property
    def current_roi(self) -> Optional[np.ndarray]:
        return self.get_roi(self.current_frame)

    def get_roi(self, frame: int) -> Optional[np.ndarray]:
        ret = None
        for mask in self.roi:
            t = mask[0]
            if t <= frame:
                ret = mask
        return ret

    def add_roi(self, x0: int, y0: int, x1: int, y1: int, t: int = None):
        if t is None:
            t = self.current_frame

        i = np.nonzero(self.roi[:, 0] == t)[0]
        if len(i) == 0:
            self.roi = np.sort(np.append(self.roi, [(t, x0, y0, x1, y1)], axis=0), axis=0)
        else:
            self.roi[i[0]] = (t, x0, y0, x1, y1)
        LOGGER.info(f'add roi [{t},{x0},{y0},{x1},{y1}]')

    def del_roi(self, index: int):
        t, x0, y0, x1, y1 = self.roi[index]
        self.roi = np.delete(self.roi, index, axis=0)
        LOGGER.info(f'del roi [{t},{x0},{y0},{x1},{y1}]')

    def clear_roi(self):
        self.roi = np.zeros((0, 5), dtype=int)

    def load_roi(self, file: str) -> np.ndarray:
        LOGGER.debug(f'load roi = {file}')
        ret: np.ndarray = np.load(file)

        # shape: (N, (time, x, y, width, height))
        if not (ret.ndim == 2 and ret.shape[1] == 5):
            raise RuntimeError('not a roi mask')

        self.roi = ret
        return ret

    def save_roi(self, file: str):
        np.save(file, self.roi)
        self.enqueue_message(f'save roi = {file}')
        LOGGER.debug(f'save roi = {file}')

    def eval_all_result(self):
        if self._eval_task is not None:
            return

        if self.output_file is None:
            raise RuntimeError('output file not set')

        roi = self.load_roi(self.roi_use_file)
        ric = roi.shape[0]
        if ric == 0:
            raise RuntimeError('empty ROI')

        frame = self.current_frame

        def before_task():
            self.enqueue_message('start eval')

        def after_task():
            self.enqueue_message('eval done')
            self.current_frame = frame

        self._eval_task = threading.Thread(
            name='eval thread',
            target=self._eval_all_result,
            args=(self.enqueue_message,),
            kwargs=dict(
                before=before_task,
                after=after_task
            )
        )
        self._eval_task.start()

    def _eval_all_result(self,
                         progress: Callable[[str], None] = print,
                         before: Callable[[], None] = None,
                         after: Callable[[], None] = None):

        if before is not None:
            before()

        vc = self.video_capture
        assert vc is not None

        self._is_playing = False
        self.eval_lick = False
        self.current_frame = 0

        roi = self.roi
        ric = roi.shape[0]
        assert ric != 0

        rii = -1

        def get_roi():
            nonlocal rii
            if rii + 1 >= ric:
                return roi[rii]

            rnt = roi[rii + 1, 0]
            if frame >= rnt:
                rii += 1
            if rii < 0:
                return None
            return roi[rii]

        lick_possibility = self.lick_possibility

        step = 10
        progress(f'eval 0% ...')
        for frame in range(self.video_total_frames):
            if 100 * frame / self.video_total_frames > step:
                progress(f'eval {step}% ...')
                step += 10

            ret, image = vc.read()
            self.current_image_original = image
            _roi = get_roi()
            if _roi is not None:
                lick_possibility[frame] = self.calculate_value(image, _roi)
            else:
                lick_possibility[frame] = 0

        progress(f'eval 100%')
        self._eval_task = None
        if after is not None:
            after()

    def clear_result(self):
        self.lick_possibility[:] = 0

    def save_result(self, file: str):
        if self.lick_possibility is None:
            raise RuntimeError()

        fps = self.video_fps
        size = self.video_total_frames
        data = np.zeros((size, 3), dtype=float)
        data[:, 0] = np.linspace(0, size / fps, num=size)
        data[:, 2] = self.lick_possibility

        data[:, 1] = -1
        for roi in self.roi:
            data[:, roi[0]:] += 1

        np.save(file, data)
        self.enqueue_message(f'save result = {file}')
        LOGGER.debug(f'save result = {file}')

    def load_result(self, file: str):
        size = self.video_total_frames

        LOGGER.debug(f'load result = {file}')
        data = np.load(file)
        if data.shape != (size, 3):
            raise RuntimeError('result file not match to current video')

        self.lick_possibility = data[:, 2].copy()

    def _get_mask_cache(self, roi: np.ndarray) -> np.ndarray:
        if self._mask_cache is None or self._mask_cache[0] != roi[0]:
            mask = rectangle_to_mask(self.video_width, self.video_height, roi)
            self._mask_cache = (roi[0], mask, mask[:, :, 0] == 255)
        return self._mask_cache[1]

    def calculate_value(self, image, roi: np.ndarray) -> float:
        self._get_mask_cache(roi)
        _, mask_b, mask_i = self._mask_cache
        img = cv2.cvtColor(cv2.bitwise_and(image, mask_b), cv2.COLOR_BGR2GRAY)

        return np.mean(img, where=mask_i)

    def start(self, pause_on_start=False):
        LOGGER.debug('start with GUI')
        vc = self._init_video()

        cv2.namedWindow(self.window_title, cv2.WINDOW_GUI_NORMAL)
        cv2.setMouseCallback(self.window_title, self.handle_mouse_event)

        if self.roi_use_file is not None:
            self.load_roi(self.roi_use_file)

        try:
            self._is_playing = not pause_on_start
            self._loop()
        except KeyboardInterrupt:
            pass
        finally:
            LOGGER.debug('closing')
            vc.release()
            cv2.destroyWindow(self.window_title)
            LOGGER.debug('closed')

    def start_no_gui(self):
        LOGGER.debug('start no GUI')
        if self.output_file is None:
            raise RuntimeError('output file not set')

        roi = self.load_roi(self.roi_use_file)
        ric = roi.shape[0]
        if ric == 0:
            raise RuntimeError('empty ROI')

        vc = self._init_video()

        try:
            self._eval_all_result(LOGGER.debug)
            self.save_result(self.output_file)
        except KeyboardInterrupt:
            pass
        finally:
            LOGGER.debug('closing')
            vc.release()
            LOGGER.debug('closed')

    def _init_video(self) -> cv2.VideoCapture:
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

        self.lick_possibility = np.zeros((self.video_total_frames,), float)

        return vc

    def _loop(self):
        while True:
            t = time.time()
            try:
                self._update()
            except KeyboardInterrupt:
                raise
            except BaseException as e:
                LOGGER.warning(e, exc_info=True)
                raise

            t = self._sleep_interval - (time.time() - t)
            if t > 0:
                time.sleep(t)

    def _update(self):
        vc = self.video_capture
        if self._is_playing or self.current_image_original is None:
            ret, image = vc.read()
            if not ret:
                self._is_playing = False
                return

            self.current_image_original = image

        frame = self.current_frame
        self.current_image = self.current_image_original.copy()
        roi = self.current_roi

        if frame < self.video_total_frames:
            if roi is not None and self.eval_lick:
                self.current_value = self.calculate_value(self.current_image_original, roi)
                self.lick_possibility[frame] = self.current_value
            else:
                self.current_value = self.lick_possibility[frame]

        self._show_queued_message()

        if len(self.buffer):
            self._show_buffer()

        if self._current_roi_region is not None:
            self._show_roi_tmp()
        elif self.show_roi:
            if roi is not None:
                self._show_roi(roi)

        if self.show_lick and self.lick_possibility is not None and roi is not None:
            self._show_lick_possibility()

        if self.show_time:
            self._show_time_bar()

        cv2.imshow(self.window_title, self.current_image)

        k = cv2.waitKey(1)
        if k > 0:
            self.handle_key_event(k)

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
        cv2.putText(self.current_image, buffer, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_RED, 2, cv2.LINE_AA)

    def _show_roi(self, roi: np.ndarray):
        _, x0, y0, x1, y1 = roi
        cv2.rectangle(self.current_image, (x0, y0), (x1, y1), COLOR_YELLOW, 2, cv2.LINE_AA)

    def _show_roi_tmp(self):
        x0, y0, x1, y1 = self._current_roi_region
        cv2.rectangle(self.current_image, (x0, y0), (x1, y1), COLOR_GREEN, 2, cv2.LINE_AA)

    def _show_lick_possibility(self):
        s = 130
        w = self.video_width
        h = self.video_height
        frame = self.current_frame
        y0 = h - 30
        length = w - 2 * s

        duration = (self.show_lick_duration * self.video_fps) // 2
        f0 = max(0, frame - duration)

        cv2.putText(self.current_image, f'-{self.show_lick_duration} s', (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_RED, 2, cv2.LINE_AA)

        if f0 != frame:
            # FIXME not work properly
            v = self.lick_possibility[f0:frame]
            length = min(length, len(v))
            y = np.histogram(v, bins=length)[0]
            y = y / max(1, np.max(y))
            y = y0 - 20 * y.astype(np.int32)
            x = np.linspace(s, w - s, length, dtype=np.int32)
            p = np.vstack((x, y)).transpose()
            cv2.polylines(self.current_image, [p], 0, COLOR_RED, 2, cv2.LINE_AA)
        else:
            cv2.line(self.current_image, (s, y0), (w - s, y0), COLOR_RED, 1, cv2.LINE_AA)

        if self.current_value is not None:
            cv2.putText(self.current_image, f'{self.current_value:01.2f}', (w - 100, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_RED, 2, cv2.LINE_AA)

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

        # line
        cv2.line(self.current_image, (s, h - 10), (w - s, h - 10), COLOR_RED, 3, cv2.LINE_AA)

        #
        # roi frame
        mx = self._frame_to_time_bar_x(self.roi[:, 0])
        for x in mx:
            cv2.line(self.current_image, (x, h - 20), (x, h), COLOR_YELLOW, 3, cv2.LINE_AA)

        # current frame
        x = self._frame_to_time_bar_x(frame)
        cv2.line(self.current_image, (x, h - 20), (x, h), COLOR_RED, 3, cv2.LINE_AA)

        # mouse hover
        if self._current_mouse_hover_frame is not None:
            x = self._frame_to_time_bar_x(self._current_mouse_hover_frame)

            color = COLOR_GREEN
            if self.mouse_stick_to_roi and len(mx) > 0:
                i = np.argmin(np.abs(mx - x))
                if abs(mx[i] - x) < self.mouse_stick_distance:
                    x = mx[i]
                    self._current_mouse_hover_frame = int(self.roi[i, 0])
                    color = COLOR_YELLOW

            cv2.line(self.current_image, (x, h - 20), (x, h), color, 3, cv2.LINE_AA)

            # text
            cv2.putText(self.current_image, self._frame_to_text(self._current_mouse_hover_frame), (x - s // 2, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

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
            raise TypeError(type(frame))

    def _frame_to_text(self, frame: int):
        t_sec = frame // self.video_fps
        t_min, t_sec = t_sec // 60, t_sec % 60
        return f'{t_min:02d}:{t_sec:02d}'

    def handle_key_event(self, k: int):
        if k == 27:  # escape:
            self.buffer = ''
        elif k == 8:  # backspace
            if len(self.buffer) > 0:
                self.buffer = self.buffer[:-1]
        elif len(self.buffer) == 0 and k == 32:  # space
            self.is_playing = not self.is_playing
        elif k == 81:  # left
            self.current_frame = max(0, self.current_frame - 5 * self.video_fps)
        elif k == 83:  # right
            self.current_frame = min(self.video_total_frames - 1, self.current_frame + 5 * self.video_fps)
        elif len(self.buffer) == 0 and k == ord('+'):
            self.speed_factor *= 2
        elif len(self.buffer) == 0 and k == ord('-'):
            self.speed_factor /= 2
        elif k == 13:  # enter:
            command = self.buffer
            self.buffer = ''
            try:
                self.handle_command(command)
            except KeyboardInterrupt:
                raise
            except BaseException as e:
                LOGGER.warning(f'command "{command}" : {e}', exc_info=True)
                self.enqueue_message(str(e))
        elif 32 <= k < 127:  # printable
            self.buffer += chr(k)

    def handle_command(self, command: str):
        LOGGER.debug(f'command : {command}')
        if command == 'h':
            self.enqueue_message('h : print key shortcut help')
            self.enqueue_message('q : quit program')
            self.enqueue_message('v : video ...')
            self.enqueue_message('j : jump to ...')
            self.enqueue_message('d : display ...')
            self.enqueue_message('r : roi ...')
            self.enqueue_message('o : result ...')

        elif command == 'q':
            raise KeyboardInterrupt

        elif command == 'o':
            self.enqueue_message('count [sec]  : lick count within sec')
            self.enqueue_message('eval[+-?]    : en/disable licking calculation')
            self.enqueue_message('eval-all     : eval all licking possibility and disable calculation')
            self.enqueue_message('clear        : clear result')
            self.enqueue_message('save <file>  : save result')
            self.enqueue_message('load <file>  : load result')
            self.enqueue_message('load?        : print result output file')
            self.enqueue_message('save?        : print result output file')

        elif command.startswith('count'):
            part: List[str] = list(filter(len, command.split(' ')))
            if len(part) == 1:
                t_sec = 10
            elif len(part) == 2:
                t_sec = int(part[1])
            else:
                raise ValueError()

            # TODO
            self.enqueue_message('not implemented')

        elif command == 'eval-all':
            self.eval_all_result()

        elif command.startswith('eval'):
            if command == 'eval':
                self.eval_lick = not self.eval_lick
            elif command == 'eval+':
                self.eval_lick = True
            elif command == 'eval-':
                self.eval_lick = False
            elif command == 'eval?':
                pass
            else:
                raise ValueError(command)

            self.enqueue_message('eval lick : ' + ('on' if self.eval_lick else 'off'))

        elif command == 'clear':
            self.clear_result()

        elif command == 'save?':
            self.enqueue_message(str(self.output_file))
        elif command == 'load?':
            self.enqueue_message(str(self.output_file))

        elif command.startswith('save'):
            part: List[str] = list(filter(len, command.split(' ')))
            if len(part) == 1:
                if self.output_file is not None:
                    self.save_result(self.output_file)
                else:
                    self.enqueue_message('None output_file')
            elif len(part) == 2:
                self.output_file = part[1]
                self.save_result(self.output_file)
            else:
                raise ValueError(command)

        elif command.startswith('load'):
            part: List[str] = list(filter(len, command.split(' ')))
            if len(part) == 1:
                if self.output_file is not None:
                    self.load_result(self.output_file)
                else:
                    self.enqueue_message('None output_file')
            elif len(part) == 2:
                self.output_file = part[1]
                self.load_result(self.output_file)
            else:
                raise ValueError(command)

        elif command == 'v':
            self.enqueue_message('va : print all video information below')
            self.enqueue_message('vp : print video file path')
            self.enqueue_message('vw : print video width')
            self.enqueue_message('vh : print video height')
            self.enqueue_message('vf : print video fps')
            self.enqueue_message('vd : print video duration')
            self.enqueue_message('vt : print video total frame')
            self.enqueue_message('vc : print current frame')

        elif command.startswith('v'):
            if command == 'va':
                command = 'pwhfdt'
            else:
                command = command[1:]

            if 'p' in command:
                self.enqueue_message(f'path : {self.video_file}')
            if 'w' in command:
                self.enqueue_message(f'width : {self.video_width}')
            if 'h' in command:
                self.enqueue_message(f'height : {self.video_height}')
            if 'f' in command:
                self.enqueue_message(f'fps : {self.video_fps}')
            if 'd' in command:
                self.enqueue_message(f'time : {self._frame_to_text(self.video_total_frames)}')
            if 't' in command:
                self.enqueue_message(f'total : {self.video_total_frames}')
            if 'c' in command:
                self.enqueue_message(f'frame : {self.current_frame}')

        elif command == 'j':
            self.enqueue_message('j<sec>        : jump to <sec> time')
            self.enqueue_message('j<min>:<sec>  : jump to <min>:<sec> time')
            self.enqueue_message('j[-+][<min>:]<sec> : jump to back/forward time')
            self.enqueue_message('jend          : jump to END')
            self.enqueue_message('jr            : jump to current roi time')
            self.enqueue_message('jr<idx>       : jump to <idx> roi time')

        elif command == 'jend':
            self.current_frame = self.video_total_frames - 1
            LOGGER.debug(f'jump to END')

        elif command == 'jr':
            self.handle_command('rj')

        elif command.startswith('jr'):
            self.handle_command(f'r{int(command[2:])}j')

        elif command.startswith('j'):
            if command.startswith('j+'):
                relative = 1
                command = command[2:]
            elif command.startswith('j-'):
                relative = -1
                command = command[2:]
            else:
                relative = 0
                command = command[1:]

            t_min, t_sec = _decode_buffer_as_time(command)
            frame = (t_min * 60 + t_sec) * self.video_fps
            if relative == 0:
                self.current_frame = frame
            else:
                frame = self.current_frame + relative * frame
                self.current_frame = frame
                t_min, t_sec = frame // 60, frame % 60

            LOGGER.debug(f'jump to {t_min:02d}:{t_sec:02d} (frame={frame})')

        elif command == 'd':
            self.enqueue_message('dtime[+-?] : display time line')
            self.enqueue_message('dlick[+-?] : display lick curve')
            self.enqueue_message('droi[+-?]  : display roi rectangle')

        elif command.startswith('dtime'):
            if command == 'dtime':
                self.show_time = not self.show_time
            elif command == 'dtime+':
                self.show_time = True
            elif command == 'dtime-':
                self.show_time = False
            elif command == 'dtime?':
                pass
            else:
                raise ValueError(command)

            self.enqueue_message('show time : ' + ('on' if self.show_time else 'off'))

        elif command.startswith('dlick'):
            if command == 'dlick':
                self.show_lick = not self.show_lick
            elif command == 'dlick+':
                self.show_lick = True
            elif command == 'dlick-':
                self.show_lick = False
            elif command == 'dlick?':
                pass
            else:
                raise ValueError(command)

            self.enqueue_message('show lick : ' + ('on' if self.show_lick else 'off'))

        elif command.startswith('droi'):
            if command == 'droi':
                self.show_roi = not self.show_roi
            elif command == 'droi+':
                self.show_roi = True
            elif command == 'droi-':
                self.show_roi = False
            elif command == 'droi?':
                pass
            else:
                raise ValueError(command)

            self.enqueue_message('show roi : ' + ('on' if self.show_roi else 'off'))

        elif command == 'r':
            self.enqueue_message('rp            : print current roi information')
            self.enqueue_message('ra            : print all roi information')
            self.enqueue_message('r[<idx>]j     : jump to current/<idx> roi time')
            self.enqueue_message('r[<idx>]d     : delete current/<idx> roi')
            self.enqueue_message('rad           : delete all roi')
            self.enqueue_message('r+ x y x y [t] : add roi')
            self.enqueue_message('rload <file>  : load roi')
            self.enqueue_message('rload?        : print use roi file')
            self.enqueue_message('rsave <file>  : save roi')
            self.enqueue_message('rsave?        : print save roi file')

        elif command == 'rp':
            roi = self.current_roi
            idx = np.nonzero(self.roi[:, 0] == roi[0])[0][0]
            self.enqueue_message(f'roi[{idx}] t={roi[0]} ({roi[1]},{roi[2]}).({roi[3]},{roi[4]})')

        elif command == 'ra':
            cnt_roi = self.current_roi
            for idx, roi in enumerate(self.roi):
                if cnt_roi[0] == roi[0]:
                    self.enqueue_message(f'*roi[{idx}] t={roi[0]} ({roi[1]},{roi[2]}).({roi[3]},{roi[4]})')
                else:
                    self.enqueue_message(f' roi[{idx}] t={roi[0]} ({roi[1]},{roi[2]}).({roi[3]},{roi[4]})')

        elif command.startswith('r') and command.endswith('j'):
            if command == 'rj':
                frame = self.current_roi[0]
                idx = np.nonzero(self.roi[:, 0] == frame)[0][0]
            else:
                idx = int(command[1:-1])
                frame = self.roi[idx, 0]

            self.current_frame = frame
            LOGGER.debug(f'jump to roi[{idx}] (frame={frame})')

        elif command == 'rd':
            frame = self.current_roi[0]
            idx = np.nonzero(self.roi[:, 0] == frame)[0][0]
            self.del_roi(idx)

        elif command.startswith('r') and command.endswith('d'):
            if command == 'rd':
                frame = self.current_roi[0]
                idx = np.nonzero(self.roi[:, 0] == frame)[0][0]
            else:
                idx = int(command[1:-1])

            self.del_roi(idx)

        elif command == 'rad':
            self.clear_roi()

        elif command.startswith('r+'):
            part: List[str] = list(filter(len, command.split(' ')))
            x0 = int(part[1])
            y0 = int(part[2])
            x1 = int(part[3])
            y1 = int(part[4])
            t = int(part[5]) if len(part) == 6 else self.current_frame
            if len(part) > 6:
                raise ValueError(command)
            self.add_roi(x0, y0, x1, y1, t)

        elif command == 'rsave?':
            self.enqueue_message(str(self.roi_output_file))
        elif command == 'rload?':
            self.enqueue_message(str(self.roi_use_file))

        elif command.startswith('rsave'):
            part: List[str] = list(filter(len, command.split(' ')))
            if len(part) == 1:
                if self.roi_output_file is not None:
                    self.save_roi(self.roi_output_file)
                else:
                    self.enqueue_message('None roi_output_file')
            elif len(part) == 2:
                self.roi_output_file = part[1]
                self.save_roi(self.roi_output_file)
            else:
                raise ValueError(command)

        elif command.startswith('rload'):
            part: List[str] = list(filter(len, command.split(' ')))
            if len(part) == 1:
                if self.roi_use_file is not None:
                    self.load_roi(self.roi_use_file)
                else:
                    self.enqueue_message('None roi_use_file')
            elif len(part) == 2:
                self.roi_use_file = part[1]
                self.load_roi(self.roi_use_file)
            else:
                raise ValueError(command)

        else:
            # self.enqueue_message('q             : quit program')
            self.enqueue_message(f'unknown command : {command}')

    def handle_mouse_event(self, event: int, x: int, y: int, flag: int, data):
        if event == cv2.EVENT_MOUSEMOVE:
            if self._current_operation_state == self.MOUSE_STATE_MASKING:
                x0, y0, _, _ = self._current_roi_region
                self._current_roi_region = [x0, y0, x, y]
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
            self._current_roi_region = [x, y, x, y]
            self.is_playing = False

        elif event == cv2.EVENT_RBUTTONUP:
            if self._current_operation_state == self.MOUSE_STATE_MASKING:
                t = self.current_frame
                n = self.roi_count
                self.add_roi(*self._current_roi_region)
                self.enqueue_message(f'add roi[{n}] at ' + self._frame_to_text(t))
                self._current_roi_region = None
                self._current_operation_state = self.MOUSE_STATE_FREE
            else:
                self._current_operation_state = self.MOUSE_STATE_FREE


def _decode_buffer_as_time(buffer: str) -> Tuple[int, int]:
    if ':' in buffer:
        t_min, t_sec = buffer.split(':')
        return int(t_min), int(t_sec)
    else:
        t_sec = int(buffer)
        return t_sec // 60, t_sec % 60


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument('-x', '--execute',
                    action='store_true',
                    help='start no GUI mode',
                    dest='execute')
    ap.add_argument('--roi',
                    metavar='FILE',
                    default=None,
                    help='use ROI file',
                    dest='use_roi')
    ap.add_argument('--save-roi',
                    metavar='FILE',
                    default=None,
                    help='save roi path, default as same as --roi',
                    dest='save_roi')
    ap.add_argument('-o', '--output', '--output-data-path',
                    metavar='FILE',
                    default=None,
                    help='save licking result',
                    dest='output')
    ap.add_argument('-p', '--pause-on-start',
                    action='store_true',
                    help='pause when start',
                    dest='pause_on_start')
    ap.add_argument('FILE')
    opt = ap.parse_args()

    main = Main(opt.FILE)
    main.output_file = opt.output
    main.roi_use_file = opt.use_roi
    main.roi_output_file = opt.save_roi if opt.save_roi is not None else opt.use_roi

    if opt.execute:
        main.start_no_gui()
    else:
        main.start(pause_on_start=opt.pause_on_start)
