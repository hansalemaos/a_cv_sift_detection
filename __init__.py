import random
from collections import defaultdict
from threading import Lock
from time import sleep
from typing import Union
import keyboard
import kthread
import pandas as pd
from a_cv_imwrite_imread_plus import open_image_in_cv
import cv2
from a_pandas_ex_group_coordinates_by_distance import group_coordinates_by_distance
from a_cv2_imshow_thread import add_imshow_thread_to_cv2
from shapely.geometry import MultiPoint
from windows_adb_screen_capture import ScreenShots

nested_dict = lambda: defaultdict(nested_dict)

add_imshow_thread_to_cv2()
sift = cv2.SIFT_create()


def join_columns_as_tuple(frame, columns):
    coordstog = frame[columns].to_records(index=False).tolist()
    return coordstog


def get_representative_coord_list_of_coords(list_):
    points = MultiPoint(list_)

    reprx, repry = points.representative_point().coords.xy
    reprx, repry = reprx[-1], repry[-1]
    return reprx, repry


def get_centroid_from_list_of_coords(list_):

    points = MultiPoint(list_)

    centroidx, centroidy = points.centroid.coords.xy
    centroidx, centroidy = centroidx[-1], centroidy[-1]
    return centroidx, centroidy


def cv_rectangle_around_1_point(
    img, start=(50, 100), width=50, height=100, thickness=3, color=(255, 0, 0)
):
    anfangbild, breite, hoehe, dicke, color = start, width, height, thickness, color
    imgOriginalScene = img.copy()
    color = list(reversed(color))
    p2fRectPoints = (
        [anfangbild[0], anfangbild[1]],
        [anfangbild[0] + breite, anfangbild[1]],
        [anfangbild[0] + breite, anfangbild[1] + hoehe],
        [anfangbild[0], anfangbild[1] + hoehe],
    )
    cv2.line(
        imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), color, dicke
    )
    cv2.line(
        imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), color, dicke
    )
    cv2.line(
        imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), color, dicke
    )
    cv2.line(
        imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), color, dicke
    )
    return imgOriginalScene.copy()


def calculate_sift(img):
    kp1, des1 = sift.detectAndCompute(img, None)
    return kp1, des1


def read_image_grayscale(img):
    return open_image_in_cv(img, channels_in_output=2)


def get_random_rgb_color(start=50, end=200):
    return (
        random.randrange(start, end),
        random.randrange(start, end),
        random.randrange(start, end),
    )


def gray_to_rgb(im):
    return cv2.cvtColor(im.copy(), cv2.COLOR_GRAY2BGR)


def convert_cv_image_to_3_channels(img):
    image = img.copy()
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    elif len(image.shape) == 2:
        image = gray_to_rgb(image)
    return image


def reverse_color(color):
    if len(color) == 3:
        return list(reversed(color))
    elif len(color) == 4:
        return list(reversed(color[:3])) + [color[-1]]
    return color


def draw_dot_and_text_with_cv(img, xy, color, text):
    img4 = img.copy()
    pt1_, pt2_ = xy
    img4 = cv_rectangle_around_1_point(
        img4, start=(pt1_, pt2_), width=15, height=15, color=color
    )
    img4 = cv_rectangle_around_1_point(
        img4, start=(pt1_, pt2_), width=10, height=10, color=(0, 0, 0)
    )
    img4 = cv_rectangle_around_1_point(
        img4, start=(pt1_, pt2_), width=5, height=5, color=color
    )
    img4 = cv2.putText(
        img4,
        str((text)),
        (pt1_ - 15, pt2_ - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        3,
    )
    img4 = cv2.putText(
        img4,
        str((text)),
        (pt1_ - 15, pt2_ - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        reverse_color(color),
        1,
    )
    return img4


def sift_detection(
    img1,
    img2,
    kp1=None,
    des1=None,
    kp2=None,
    des2=None,
    checks=50,
    trees=5,
    debug=False,
    max_distance=70,
    minimum_matches_per_group=3,
):
    img1 = read_image_grayscale(img1)
    img2 = read_image_grayscale(img2)

    if kp1 is None or des1 is None:
        kp1, des1 = sift.detectAndCompute(img1, None)
    if kp2 is None or des2 is None:
        kp2, des2 = sift.detectAndCompute(img2, None)
    img3 = None
    img4 = None
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=trees)
    search_params = dict(checks=checks)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    if debug is True:
        matchesMask = [[0, 0] for i in range(len(matches))]
        for i, (m, n) in enumerate(matches):
            if m.distance < max_distance / 100 * n.distance:
                matchesMask[i] = [1, 0]
        draw_params = dict(
            matchColor=(0, 255, 0),
            singlePointColor=(255, 0, 0),
            matchesMask=matchesMask,
            flags=cv2.DrawMatchesFlags_DEFAULT,
        )
        img3 = cv2.drawMatchesKnn(
            img1.copy(), kp1, img2.copy(), kp2, matches, None, **draw_params
        )

    piccoords = []
    for no in range(len(matches)):

        piccoords.append(kp2[matches[no][0].trainIdx].pt + (matches[no][0].distance,))

    df = (
        pd.DataFrame.from_records(piccoords, columns=["x", "y", "distance"])
        .drop_duplicates()
        .reset_index(drop=True)
    )
    df = group_coordinates_by_distance(
        df[["x", "y"]].copy(), max_euclidean_distance=max_distance
    )
    upcounter = 0
    goodones = []
    for name, group in df.groupby("item"):
        if len(group) >= minimum_matches_per_group:
            groupcp = group.copy()
            groupcp["item"] = upcounter
            groulist = join_columns_as_tuple(groupcp, columns=["x", "y"])
            groupcp["x_y"] = groulist.copy()
            groupcp["repr_point"] = [
                get_representative_coord_list_of_coords(groulist)
            ] * len(groupcp)
            groupcp["centroid"] = [get_centroid_from_list_of_coords(groulist)] * len(
                groupcp
            )

            goodones.append(groupcp.copy())
            upcounter += 1

    df = pd.concat(goodones.copy(), ignore_index=True)

    if debug is True:
        img4 = convert_cv_image_to_3_channels(img2.copy())
        for name, group in df.groupby("item"):
            color = get_random_rgb_color()
            color_rev = reverse_color(color)
            for key, item in group.iterrows():
                pt1_ = int(item["x"])
                pt2_ = int(item["y"])
                middle = (int(item["repr_point"][0]), int(item["repr_point"][1]))
                img4 = draw_dot_and_text_with_cv(
                    img=img4, xy=(pt1_, pt2_), color=color, text=" "
                )

                img4 = draw_dot_and_text_with_cv(
                    img=img4,
                    xy=middle,
                    color=color_rev,
                    text=str(middle) + " " + str(int(item["item"])),
                )
    return df, img3, img4


def get_needle_images(needle_images, scale_percent=50):
    if not isinstance(needle_images, list):
        needle_images = [needle_images]
    needle_images_ready = []
    for icon in needle_images:
        readim = open_image_in_cv(icon, channels_in_output=2)
        if scale_percent != 100:
            width = int(readim.shape[1] * scale_percent / 100)
            height = int(readim.shape[0] * scale_percent / 100)
            dim = (width, height)
            resized = cv2.resize(readim.copy(), dim, interpolation=cv2.INTER_LANCZOS4)
        else:
            resized = readim.copy()
        kp, des = calculate_sift(resized)
        needle_images_ready.append((icon, kp, des, resized.copy()))
    return needle_images_ready


def get_haystack_image(haystack_image):
    haystack_image = open_image_in_cv(haystack_image, channels_in_output=2)
    kp2, des2 = calculate_sift(haystack_image)
    return haystack_image, kp2, des2


class SiftMatchingOnScreen:
    def __init__(self):

        self.sc = None  # = ScreenShots()
        self.monitor = None
        self.adb_path = None
        self.adb_serial = None
        self.hwnd = None
        self.needle_images = []
        self.image_original = None
        self.df = pd.DataFrame()
        self.untouchedimage = None
        self.lock = Lock()
        self.show_results_thread = None
        self.debug_images1 = []
        self.debug_images2 = []

    def get_needle_images(self, path, scale_percent=50):
        self.needle_images = get_needle_images(path, scale_percent=scale_percent)
        return self

    def get_haystack_image(self):
        self.get_screenshot()
        haystack_image = self.image_original.copy()
        return haystack_image

    def configure_monitor(self, monitor=1):
        self.sc = ScreenShots()
        self.monitor = monitor
        self.sc.choose_monitor_for_screenshot(monitor)
        return self

    def configure_adb(
        self, adb_path=r"C:\ProgramData\adb\adb.exe", adb_serial="localhost:5555"
    ):
        self.adb_path = adb_path
        self.adb_serial = adb_serial
        self.sc = ScreenShots(hwnd=None, adb_path=adb_path, adb_serial=adb_serial)
        return self

    def configure_window(self, regular_expression=None, hwnd=None):
        self.sc = ScreenShots(hwnd=hwnd)
        if hwnd is None and regular_expression is not None:
            self.sc.find_window_with_regex(regular_expression)
        self.hwnd = self.sc.hwnd
        return

    def get_screenshot(self):
        if self.adb_path is not None and self.adb_serial is not None:
            self.image_original = self.sc.imget_adb().copy()
        elif self.hwnd is not None:

            self.image_original = self.sc.imget_hwnd().copy()
        elif self.monitor is not None:
            self.image_original = self.sc.imget_monitor().copy()
        self.image_original = open_image_in_cv(
            self.image_original, channels_in_output=3
        )
        self.untouchedimage = self.image_original.copy()
        return self

    def _show_results_as_video(
        self, quit_key="q", sleep_time: Union[float, int] = 0.05,
    ):
        def activate_stop():
            nonlocal stop
            stop = True

        stop = False
        keyboard.add_hotkey(quit_key, activate_stop)
        cv2.destroyAllWindows()
        sleep(1)
        while not stop:
            self.get_screenshot_and_start_detection(
                show_results=True,
                sleep_time_for_results=sleep_time,
                quit_key_for_results=quit_key,
            )
            sleep(sleep_time)
        keyboard.remove_all_hotkeys()

    def draw_results(self):
        dfn = self.df.copy()
        try:
            newind = pd.MultiIndex.from_frame(dfn[["needle", "item"]])
        except Exception:
            return convert_cv_image_to_3_channels(self.untouchedimage.copy())
        dfn = dfn.set_index(newind)
        dfnoduplicates = dfn.drop_duplicates(subset=["item", "repr_point", "needle"])

        img4 = convert_cv_image_to_3_channels(self.untouchedimage.copy())
        for key, item in dfnoduplicates.iterrows():
            color = get_random_rgb_color()
            pt1_ = int(item["x"])
            pt2_ = int(item["y"])
            middle = (int(item["repr_point"][0]), int(item["repr_point"][1]))

            img4 = cv_rectangle_around_1_point(
                img4, start=(pt1_, pt2_), width=15, height=15, color=color
            )
            img4 = cv_rectangle_around_1_point(
                img4, start=(pt1_, pt2_), width=10, height=10, color=(0, 0, 0)
            )
            img4 = cv_rectangle_around_1_point(
                img4, start=(pt1_, pt2_), width=5, height=5, color=color
            )
            img4 = cv2.putText(
                img4,
                str(middle),
                (pt1_ - 15, pt2_ - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                3,
            )
            img4 = cv2.putText(
                img4,
                str(middle),
                (pt1_ - 15, pt2_ - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                reverse_color(color),
                1,
            )
            img4 = cv2.putText(
                img4,
                str((item["needle"])),
                (pt1_ - 45, pt2_ - 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                3,
            )
            img4 = cv2.putText(
                img4,
                str((item["needle"])),
                (pt1_ - 45, pt2_ - 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                reverse_color(color),
                1,
            )
        return img4

    def _show_results(
        self, quit_key="q", sleep_time: Union[float, int] = 0.05,
    ):
        def activate_stop():
            nonlocal stop
            stop = True

        stop = False
        keyboard.add_hotkey(quit_key, activate_stop)
        cv2.destroyAllWindows()
        sleep(1)
        while not stop:
            screenshot_window = self.draw_results()
            # screenshot_window = self.untouchedimage.copy()

            if cv2.waitKey(1) & 0xFF == ord(quit_key):
                cv2.waitKey(0)

            cv2.imshow("", screenshot_window)
            sleep(sleep_time)
        keyboard.remove_all_hotkeys()
        return self

    def _scale_image(self, img, scale_percent):
        if scale_percent == 100:
            return img
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(img.copy(), dim, interpolation=cv2.INTER_LANCZOS4)
        return resized

    def _start_detect(
        self,
        checks=30,
        trees=5,
        debug=False,
        max_distance=100,
        minimum_matches_per_group=12,
        scale_percent=50,
    ):
        self.get_screenshot()
        img = self.untouchedimage.copy()

        resized = self._scale_image(img, scale_percent)

        haystack_image = open_image_in_cv(
            resized.copy(), channels_in_output=2, bgr_to_rgb=False
        )

        haystack_image, kp2, des2 = get_haystack_image(haystack_image)
        needle_images_ready = self.needle_images
        allresas = []
        for icon, kp, des, readim in needle_images_ready:
            try:
                df, img3, img4 = sift_detection(
                    img1=readim.copy(),
                    img2=haystack_image.copy(),
                    kp1=kp,
                    des1=des,
                    kp2=kp2,
                    des2=des2,
                    checks=checks,
                    trees=trees,
                    debug=debug,
                    max_distance=max_distance,
                    minimum_matches_per_group=minimum_matches_per_group,
                )
                allresas.append([df.assign(needle=icon), img3, img4])
            except Exception as fe:
                continue

        dfn = pd.concat([x[0] for x in allresas], ignore_index=True)
        self.df = dfn.copy()
        scale = scale_percent
        self.df.x = self.df.x * 100 / scale
        self.df.y = self.df.y * 100 / scale
        self.df.x_y = self.df.x_y.apply(
            lambda x: (x[0] * 100 / scale, x[1] * 100 / scale)
        )
        self.df.repr_point = self.df.repr_point.apply(
            lambda x: (x[0] * 100 / scale, x[1] * 100 / scale)
        )
        self.df.centroid = self.df.centroid.apply(
            lambda x: (x[0] * 100 / scale, x[1] * 100 / scale)
        )
        debugim1 = [x[1] for x in allresas]
        debugim2 = [x[2] for x in allresas]
        self.debug_images1 = debugim1.copy()
        self.debug_images2 = debugim2.copy()
        return self

    def get_screenshot_and_start_detection(
        self,
        checks=30,
        trees=5,
        debug=False,
        max_distance=100,
        minimum_matches_per_group=12,
        show_results=False,
        sleep_time_for_results=0.1,
        quit_key_for_results="q",
        scale_percent=50,
    ):
        if show_results:
            self.lock.acquire()
        try:
            self._start_detect(
                checks=checks,
                trees=trees,
                debug=debug,
                max_distance=max_distance,
                minimum_matches_per_group=minimum_matches_per_group,
                scale_percent=scale_percent,
            )
        except Exception as fa:
            print(fa)
        if show_results:
            self.lock.release()
            if self.show_results_thread is None:
                self.show_results_thread = kthread.KThread(
                    target=self._show_results,
                    name="results",
                    args=(quit_key_for_results, sleep_time_for_results),
                )
                self.show_results_thread.start()
            elif not self.show_results_thread.is_alive():
                self.show_results_thread = kthread.KThread(
                    target=self._show_results,
                    name="results",
                    args=(quit_key_for_results, sleep_time_for_results),
                )
        return self

