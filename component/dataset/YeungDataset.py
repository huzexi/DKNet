from .BaseDataset import BaseDataset
from .LytroDataset import LytroDataset

__all__ = ['YeungDataset']


class YeungDataset(LytroDataset):
    """
    The same dataset with the following work:
    Yeung, Henry Wing Fung, Junhui Hou, Xiaoming Chen, Jie Chen, Zhibo Chen, and Yuk Ying Chung.
    “Light Field Spatial Super-Resolution Using Deep Efficient Spatial-Angular Separable Convolution.”
    IEEE Transactions on Image Processing 28, no. 5 (2019): 2319–30.
    """

    # Configuration
    path_h5 = {
        BaseDataset.MODE_TRAIN: 'Yeung_Train.h5',
        BaseDataset.MODE_TEST: 'Yeung_Test.h5'
    }

    list = {
        BaseDataset.MODE_TRAIN: {
            'Stanford': [
                'bikes/bikes_10_eslf', 'bikes/bikes_11_eslf', 'bikes/bikes_13_eslf', 'bikes/bikes_15_eslf',
                'bikes/bikes_16_eslf', 'bikes/bikes_1_eslf', 'bikes/bikes_20_eslf', 'bikes/bikes_2_eslf',
                'bikes/bikes_5_eslf', 'bikes/bikes_6_eslf', 'bikes/bikes_7_eslf',
                'buildings/buildings_14_eslf', 'buildings/buildings_1_eslf', 'buildings/buildings_24_eslf',
                'buildings/buildings_26_eslf', 'buildings/buildings_27_eslf', 'buildings/buildings_32_eslf',
                'buildings/buildings_33_eslf', 'buildings/buildings_4_eslf', 'buildings/buildings_7_eslf',
                'buildings/buildings_8_eslf', 'buildings/buildings_9_eslf',
                'cars/cars_11_eslf', 'cars/cars_13_eslf', 'cars/cars_17_eslf', 'cars/cars_21_eslf', 'cars/cars_24_eslf',
                'cars/cars_26_eslf', 'cars/cars_2_eslf', 'cars/cars_34_eslf', 'cars/cars_37_eslf', 'cars/cars_3_eslf',
                'cars/cars_41_eslf', 'cars/cars_42_eslf', 'cars/cars_44_eslf', 'cars/cars_49_eslf', 'cars/cars_4_eslf',
                'cars/cars_50_eslf', 'cars/cars_52_eslf', 'cars/cars_53_eslf', 'cars/cars_56_eslf', 'cars/cars_57_eslf',
                'cars/cars_9_eslf',
                'flowers_plants/flowers_plants_10_eslf', 'flowers_plants/flowers_plants_12_eslf',
                'flowers_plants/flowers_plants_13_eslf', 'flowers_plants/flowers_plants_16_eslf',
                'flowers_plants/flowers_plants_17_eslf', 'flowers_plants/flowers_plants_18_eslf',
                'flowers_plants/flowers_plants_20_eslf', 'flowers_plants/flowers_plants_22_eslf',
                'flowers_plants/flowers_plants_23_eslf', 'flowers_plants/flowers_plants_25_eslf',
                'flowers_plants/flowers_plants_26_eslf', 'flowers_plants/flowers_plants_27_eslf',
                'flowers_plants/flowers_plants_28_eslf', 'flowers_plants/flowers_plants_2_eslf',
                'flowers_plants/flowers_plants_31_eslf', 'flowers_plants/flowers_plants_32_eslf',
                'flowers_plants/flowers_plants_35_eslf', 'flowers_plants/flowers_plants_36_eslf',
                'flowers_plants/flowers_plants_38_eslf', 'flowers_plants/flowers_plants_3_eslf',
                'flowers_plants/flowers_plants_41_eslf', 'flowers_plants/flowers_plants_42_eslf',
                'flowers_plants/flowers_plants_47_eslf', 'flowers_plants/flowers_plants_4_eslf',
                'flowers_plants/flowers_plants_55_eslf', 'flowers_plants/flowers_plants_60_eslf',
                'flowers_plants/flowers_plants_63_eslf', 'flowers_plants/flowers_plants_64_eslf',
                'flowers_plants/flowers_plants_6_eslf', 'flowers_plants/flowers_plants_7_eslf',
                'flowers_plants/flowers_plants_8_eslf',
                'fruits_vegetables/fruits_vegetables_13_eslf', 'fruits_vegetables/fruits_vegetables_14_eslf',
                'fruits_vegetables/fruits_vegetables_15_eslf', 'fruits_vegetables/fruits_vegetables_16_eslf',
                'fruits_vegetables/fruits_vegetables_17_eslf', 'fruits_vegetables/fruits_vegetables_20_eslf',
                'fruits_vegetables/fruits_vegetables_3_eslf',
                'occlusions/occlusions_15_eslf', 'occlusions/occlusions_1_eslf', 'occlusions/occlusions_21_eslf',
                'occlusions/occlusions_24_eslf', 'occlusions/occlusions_25_eslf', 'occlusions/occlusions_27_eslf',
                'occlusions/occlusions_2_eslf', 'occlusions/occlusions_3_eslf', 'occlusions/occlusions_40_eslf',
                'occlusions/occlusions_49_eslf', 'occlusions/occlusions_4_eslf', 'occlusions/occlusions_6_eslf',
                'people/people_5_eslf',
                'reflective/reflective_12_eslf', 'reflective/reflective_1_eslf', 'reflective/reflective_22_eslf',
                'reflective/reflective_29_eslf', 'reflective/reflective_4_eslf', 'reflective/reflective_5_eslf'
            ],
            'SIGASIA16': [
                'TrainingSet/OURS/IMG_0288_eslf', 'TrainingSet/OURS/IMG_0466_eslf', 'TrainingSet/OURS/IMG_0518_eslf',
                'TrainingSet/OURS/IMG_0681_eslf', 'TrainingSet/OURS/IMG_1016_eslf', 'TrainingSet/OURS/IMG_1410_eslf',
                'TrainingSet/OURS/IMG_1414_eslf', 'TrainingSet/OURS/IMG_1419_eslf', 'TrainingSet/OURS/IMG_1471_eslf',
                'TrainingSet/OURS/IMG_1477_eslf', 'TrainingSet/OURS/IMG_1479_eslf', 'TrainingSet/OURS/IMG_1483_eslf',
                'TrainingSet/OURS/IMG_1484_eslf', 'TrainingSet/OURS/IMG_1501_eslf', 'TrainingSet/OURS/IMG_1504_eslf',
                'TrainingSet/OURS/IMG_1522_eslf', 'TrainingSet/OURS/IMG_1544_eslf', 'TrainingSet/OURS/IMG_1546_eslf',
                'TrainingSet/OURS/IMG_1566_eslf', 'TrainingSet/OURS/IMG_1582_eslf', 'TrainingSet/OURS/IMG_1594_eslf',
                'TestSet/EXTRA/IMG_1187_eslf', 'TestSet/EXTRA/IMG_1306_eslf', 'TestSet/EXTRA/IMG_1321_eslf',
                'TestSet/EXTRA/IMG_1411_eslf', 'TestSet/EXTRA/IMG_1541_eslf', 'TestSet/EXTRA/IMG_1554_eslf',
                'TestSet/PAPER/Rock', 'TestSet/PAPER/Cars', 'TestSet/PAPER/Flower1'
            ]
        },
        BaseDataset.MODE_TEST: {
            'Stanford': [
                'general/general_1_eslf', 'general/general_2_eslf', 'general/general_3_eslf', 'general/general_4_eslf',
                'general/general_5_eslf', 'general/general_6_eslf', 'general/general_7_eslf', 'general/general_8_eslf',
                'general/general_9_eslf', 'general/general_10_eslf', 'general/general_11_eslf',
                'general/general_12_eslf', 'general/general_13_eslf', 'general/general_14_eslf',
                'general/general_15_eslf', 'general/general_16_eslf', 'general/general_17_eslf',
                'general/general_18_eslf', 'general/general_19_eslf', 'general/general_20_eslf',
                'general/general_21_eslf', 'general/general_22_eslf', 'general/general_23_eslf',
                'general/general_24_eslf', 'general/general_25_eslf', 'general/general_26_eslf',
                'general/general_27_eslf', 'general/general_28_eslf', 'general/general_29_eslf',
                'general/general_30_eslf', 'general/general_31_eslf', 'general/general_32_eslf',
                'general/general_33_eslf', 'general/general_34_eslf', 'general/general_35_eslf',
                'general/general_36_eslf', 'general/general_37_eslf', 'general/general_38_eslf',
                'general/general_39_eslf', 'general/general_40_eslf', 'general/general_41_eslf',
                'general/general_42_eslf', 'general/general_43_eslf', 'general/general_44_eslf',
                'general/general_45_eslf', 'general/general_46_eslf', 'general/general_47_eslf',
                'general/general_48_eslf', 'general/general_49_eslf', 'general/general_50_eslf',
                'general/general_51_eslf', 'general/general_52_eslf', 'general/general_53_eslf',
                'general/general_54_eslf', 'general/general_55_eslf', 'general/general_56_eslf',
                'general/general_57_eslf'
            ]
        }
    }
