import numpy as np

from vision.utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors


image_size = 640
image_mean = np.array([123, 117, 104])  # RGB layout
image_std = 1.0

iou_threshold = 0.45
center_variance = 0.1
size_variance = 0.2


specs = [
    SSDSpec(80, 8, SSDBoxSizes(15, 30), [2, 3]),
    SSDSpec(40, 16, SSDBoxSizes(30, 60), [2, 3]),
    SSDSpec(20, 32, SSDBoxSizes(60, 105), [2, 3]),
    SSDSpec(10, 64, SSDBoxSizes(105, 150), [2, 3]),
    SSDSpec(5, 128, SSDBoxSizes(150, 200), [2, 3]),
    SSDSpec(3, 214, SSDBoxSizes(250, 340), [2, 3])
]
#specs = [
#    SSDSpec(38, 16, SSDBoxSizes(15, 30), [1, 2]),
#    SSDSpec(19, 32, SSDBoxSizes(30, 60), [1, 2]),
#    SSDSpec(10, 64, SSDBoxSizes(60, 105), [1, 2]),
#    SSDSpec(5, 100, SSDBoxSizes(105, 150), [1, 2]),
#    SSDSpec(3, 150, SSDBoxSizes(150, 195), [1, 2]),
#    SSDSpec(1, 300, SSDBoxSizes(195, 240), [1, 2])
#]

#specs = [ # index-2
#    SSDSpec(38, 8, SSDBoxSizes(15, 30), [2]),
#    SSDSpec(19, 16, SSDBoxSizes(60, 111), [2, 3]),
#    SSDSpec(10, 32, SSDBoxSizes(111, 162), [2, 3]),
#    SSDSpec(5, 64, SSDBoxSizes(162, 213), [2, 3]),
#    SSDSpec(3, 100, SSDBoxSizes(213, 264), [2]),
#    SSDSpec(1, 300, SSDBoxSizes(264, 315), [2])
#]
# orig
#specs = [
#    SSDSpec(38, 8, SSDBoxSizes(30, 60), [2]),
#    SSDSpec(19, 16, SSDBoxSizes(60, 111), [2, 3]),
#    SSDSpec(10, 32, SSDBoxSizes(111, 162), [2, 3]),
#    SSDSpec(5, 64, SSDBoxSizes(162, 213), [2, 3]),
#    SSDSpec(3, 100, SSDBoxSizes(213, 264), [2]),
#    SSDSpec(1, 300, SSDBoxSizes(264, 315), [2])
#]

priors = generate_ssd_priors(specs, image_size)
