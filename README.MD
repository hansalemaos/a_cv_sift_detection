# Detecting objects using openCV and sift


<img src="https://github.com/hansalemaos/screenshots/raw/main/sift.png"/>


```python

from a_cv_sift_detection import SiftMatchingOnScreen
needle_images = [
    r"C:\detectiontest\media_manager_icon--744x194--1250x738.png",
    r"C:\detectiontest\chrome_icon--643x199--1140x734.png",
    r"C:\detectiontest\einstellungen_icon--537x200--1038x735.png",
    r"C:\detectiontest\kamera_icon--426x203--931x738.png",
    r"C:\detectiontest\spiele_und_gewinne_icon--1347x0--1920x452.png",
    r"C:\detectiontest\bluestacks_x_icon--1101x0--1643x449.png",
    r"C:\detectiontest\roblox_icon--833x0--1342x448.png",
    r"C:\detectiontest\systemapps_icon--528x0--1067x448.png",
    r"C:\detectiontest\gamecenter_icon--244x0--781x451.png",
    r"C:\detectiontest\playstore_icon--0x0--478x451.png",
]



siftdetect = SiftMatchingOnScreen()
siftdetect.configure_monitor(monitor=1)
siftdetect.get_needle_images(needle_images, scale_percent=100)
while True:
    siftdetect.get_screenshot_and_start_detection(
        checks=50,
        trees=5,
        debug=False,
        max_distance=100,
        minimum_matches_per_group=5,
        show_results=True,
        sleep_time_for_results=0.1,
        quit_key_for_results="q",
        scale_percent=100,
    )
    sleep(5)


while True:
    siftdetect.get_screenshot_and_start_detection(
        checks=50,
        trees=5,
        debug=False,
        max_distance=100,
        minimum_matches_per_group=5,
        show_results=False,
        sleep_time_for_results=0.1,
        quit_key_for_results="q",
        scale_percent=100,
    )
    print(siftdetect.df)


0    956.575073  ...  C:\detectiontest\media_manager_icon--744x194--...
1    960.470520  ...  C:\detectiontest\media_manager_icon--744x194--...
2    961.128601  ...  C:\detectiontest\media_manager_icon--744x194--...
3    964.380310  ...  C:\detectiontest\media_manager_icon--744x194--...
4    965.097656  ...  C:\detectiontest\media_manager_icon--744x194--...
..          ...  ...                                                ...
420  257.061249  ...  C:\detectiontest\playstore_icon--0x0--478x451.png
421  263.247589  ...  C:\detectiontest\playstore_icon--0x0--478x451.png
422  263.265106  ...  C:\detectiontest\playstore_icon--0x0--478x451.png
423  263.682190  ...  C:\detectiontest\playstore_icon--0x0--478x451.png
424  267.074219  ...  C:\detectiontest\playstore_icon--0x0--478x451.png
[425 rows x 7 columns]
              x  ...                                             needle
0    762.877319  ...  C:\detectiontest\media_manager_icon--744x194--...
1    781.889160  ...  C:\detectiontest\media_manager_icon--744x194--...
2    816.500549  ...  C:\detectiontest\media_manager_icon--744x194--...
3    816.339050  ...  C:\detectiontest\media_manager_icon--744x194--...
4    816.586853  ...  C:\detectiontest\media_manager_icon--744x194--...
..          ...  ...                                                ...
344  257.061249  ...  C:\detectiontest\playstore_icon--0x0--478x451.png
345  263.247589  ...  C:\detectiontest\playstore_icon--0x0--478x451.png
346  263.265106  ...  C:\detectiontest\playstore_icon--0x0--478x451.png
347  263.682190  ...  C:\detectiontest\playstore_icon--0x0--478x451.png
348  267.074219  ...  C:\detectiontest\playstore_icon--0x0--478x451.png
[349 rows x 7 columns]
No objects to concatenate
              x  ...                                             needle
0    881.638245  ...  C:\detectiontest\media_manager_icon--744x194--...
1    935.431885  ...  C:\detectiontest\media_manager_icon--744x194--...
2    955.275879  ...  C:\detectiontest\media_manager_icon--744x194--...
3    941.385254  ...  C:\detectiontest\media_manager_icon--744x194--...
4    817.367920  ...  C:\detectiontest\media_manager_icon--744x194--...
..          ...  ...                                                ...
302  392.155945  ...  C:\detectiontest\playstore_icon--0x0--478x451.png
303  428.587921  ...  C:\detectiontest\playstore_icon--0x0--478x451.png
304  431.010773  ...  C:\detectiontest\playstore_icon--0x0--478x451.png
305  476.075165  ...  C:\detectiontest\playstore_icon--0x0--478x451.png
306  323.301575  ...  C:\detectiontest\playstore_icon--0x0--478x451.png
[307 rows x 7 columns]
              x  ...                                             needle
0    881.638245  ...  C:\detectiontest\media_manager_icon--744x194--...
1    937.933167  ...  C:\detectiontest\media_manager_icon--744x194--...
2    935.431885  ...  C:\detectiontest\media_manager_icon--744x194--...
3    955.450562  ...  C:\detectiontest\media_manager_icon--744x194--...
4    941.385254  ...  C:\detectiontest\media_manager_icon--744x194--...
..          ...  ...                                                ...
303  417.567749  ...  C:\detectiontest\playstore_icon--0x0--478x451.png
304  429.982422  ...  C:\detectiontest\playstore_icon--0x0--478x451.png
305  438.710541  ...  C:\detectiontest\playstore_icon--0x0--478x451.png
306  445.110016  ...  C:\detectiontest\playstore_icon--0x0--478x451.png
307  458.563171  ...  C:\detectiontest\playstore_icon--0x0--478x451.png
[308 rows x 7 columns]
               x  ...                                             needle
0     996.570251  ...  C:\detectiontest\media_manager_icon--744x194--...
1    1011.908936  ...  C:\detectiontest\media_manager_icon--744x194--...
2    1015.692566  ...  C:\detectiontest\media_manager_icon--744x194--...
3    1056.677734  ...  C:\detectiontest\media_manager_icon--744x194--...
4    1071.151245  ...  C:\detectiontest\media_manager_icon--744x194--...
..           ...  ...                                                ...
345   257.061249  ...  C:\detectiontest\playstore_icon--0x0--478x451.png
346   263.247589  ...  C:\detectiontest\playstore_icon--0x0--478x451.png
347   263.265106  ...  C:\detectiontest\playstore_icon--0x0--478x451.png
348   263.682190  ...  C:\detectiontest\playstore_icon--0x0--478x451.png
349   267.074219  ...  C:\detectiontest\playstore_icon--0x0--478x451.png
[350 rows x 7 columns]
               x  ...                                             needle
0     996.570251  ...  C:\detectiontest\media_manager_icon--744x194--...
1    1011.908936  ...  C:\detectiontest\media_manager_icon--744x194--...
2    1015.692566  ...  C:\detectiontest\media_manager_icon--744x194--...
3    1056.677734  ...  C:\detectiontest\media_manager_icon--744x194--...
4    1071.151245  ...  C:\detectiontest\media_manager_icon--744x194--...
..           ...  ...                                                ...
352   257.061249  ...  C:\detectiontest\playstore_icon--0x0--478x451.png
353   263.247589  ...  C:\detectiontest\playstore_icon--0x0--478x451.png
354   263.265106  ...  C:\detectiontest\playstore_icon--0x0--478x451.png
355   263.682190  ...  C:\detectiontest\playstore_icon--0x0--478x451.png
356   267.074219  ...  C:\detectiontest\playstore_icon--0x0--478x451.png
[357 rows x 7 columns]
              x  ...                                             needle
0    762.877319  ...  C:\detectiontest\media_manager_icon--744x194--...
1    781.889160  ...  C:\detectiontest\media_manager_icon--744x194--...
2    816.500549  ...  C:\detectiontest\media_manager_icon--744x194--...
3    816.339050  ...  C:\detectiontest\media_manager_icon--744x194--...
4    816.586853  ...  C:\detectiontest\media_manager_icon--744x194--...
..          ...  ...                                                ...
343  257.061249  ...  C:\detectiontest\playstore_icon--0x0--478x451.png
344  263.247589  ...  C:\detectiontest\playstore_icon--0x0--478x451.png
345  263.265106  ...  C:\detectiontest\playstore_icon--0x0--478x451.png
346  263.682190  ...  C:\detectiontest\playstore_icon--0x0--478x451.png
347  267.074219  ...  C:\detectiontest\playstore_icon--0x0--478x451.png
[348 rows x 7 columns]


```