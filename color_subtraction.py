import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
import libs.method.QcImage as QcImage
from libs.method.PolynomialRegression3D import PolynomialRegression3D
from libs.model.TrainingSet import TrainingSet
from libs.model.Num3 import Num3

CONFIG_PATH = 'config_red.json'

RECT_SCALE = 1000
REFERENCE_BLANK_BGR = [135.0, 143.0, 136.0]
REFERENCE_BLK_BLANK_BGR = [11.0, 10.0, 9.0]
PARAMS = np.array([0.0681, 0.371, 0.417])
ABSORB_RATE = 1.0
VIS = False


def subtract_reference(image, reference_bgr, blk_reference_bgr):
    image = image.astype(np.float)
    b, g, r = cv2.split(image)
    res_b = b
    res_g = g
    res_r = r

    # The new color subtraction algorithm
    slope1, intercept1, _, _, _ = ss.linregress(
        [REFERENCE_BLANK_BGR[0], REFERENCE_BLK_BLANK_BGR[0]],
        [reference_bgr[0] / REFERENCE_BLANK_BGR[0],
         blk_reference_bgr[0] / REFERENCE_BLK_BLANK_BGR[0]])
    slope2, intercept2, _, _, _ = ss.linregress(
        [REFERENCE_BLANK_BGR[1], REFERENCE_BLK_BLANK_BGR[1]],
        [reference_bgr[1] / REFERENCE_BLANK_BGR[1],
         blk_reference_bgr[1] / REFERENCE_BLK_BLANK_BGR[1]])
    slope3, intercept3, _, _, _ = ss.linregress(
        [REFERENCE_BLANK_BGR[2], REFERENCE_BLK_BLANK_BGR[2]],
        [reference_bgr[2] / REFERENCE_BLANK_BGR[2],
         blk_reference_bgr[2] / REFERENCE_BLK_BLANK_BGR[2]])
    for i in range(3):
        res_b = b * ABSORB_RATE / \
            (slope1 * res_b + intercept1)
        res_g = g * ABSORB_RATE / \
            (slope2 * res_g + intercept2)
        res_r = r * ABSORB_RATE / \
            (slope3 * res_r + intercept3)

    # The old color subtraction algorithm
    # res_b = b * REFERENCE_BLANK_BGR[0] / reference_bgr[0]
    # res_g = g * REFERENCE_BLANK_BGR[1] / reference_bgr[1]
    # res_r = r * REFERENCE_BLANK_BGR[2] / reference_bgr[2]

    res_b = np.array(res_b, np.uint8)
    res_g = np.array(res_g, np.uint8)
    res_r = np.array(res_r, np.uint8)
    image = cv2.merge([res_b, res_g, res_r])
    image = image.astype(np.uint8)
    return image


if __name__ == "__main__":

    with open(CONFIG_PATH) as config_json:
        config = json.load(config_json)
        trainJsonPath = config.get('trainJsonPath')
        predictJsonPath = config.get('predictJsonPath')
        trainImagePath = config.get('trainImagePath')
        predictImagePath = config.get('predictImagePath')

    # train
    with open(trainJsonPath) as json_data:
        objs = json.load(json_data)

    predictions_l = []
    predictions_u = []
    predictions_v = []
    results = []

    pr3 = PolynomialRegression3D(4)

    first_bg = None
    first_bg2 = None

    for obj in objs:

        trainingSet = TrainingSet(obj)

        cv_image = cv2.imread(
            trainImagePath + trainingSet.imagePath, cv2.IMREAD_COLOR)
        background_image = cv2.imread(
            trainImagePath + trainingSet.bgImageInfoId, cv2.IMREAD_COLOR)

        if cv_image is None or background_image is None:
            print('Training image: ' + trainingSet.imagePath + ' cannot be found.')
            continue

        cv_normalized = QcImage.SNIC_EMOR(
            background_image, cv_image, PARAMS)

        dis_image = cv_normalized.copy()

        height, width, channels = cv_normalized.shape

        for i in range(len(trainingSet.references)):
            anno = trainingSet.references[i]
            if anno.type != '0':
                continue
            try:
                result = float(anno.result)
            except:
                _ = 0
            colour_area = QcImage.crop_image_by_position_and_rect(
                cv_normalized, anno.position, anno.rect)
            sample_bgr = QcImage.get_average_rgb(colour_area)
            colour_area = cv2.cvtColor(colour_area, cv2.COLOR_BGR2Lab)
            sample_lab = QcImage.get_average_rgb(colour_area)
            predictions_l.append(sample_lab[0])
            predictions_u.append(sample_lab[1])
            predictions_v.append(sample_lab[2])
            results.append(result)

        # display training image and label
        if VIS:
            dis_image = cv2.cvtColor(dis_image, cv2.COLOR_BGR2RGB)

            plt.imshow(dis_image)
            plt.title(trainingSet.imagePath)
            plt.show()

    if len(results) < 2:
        print('Sample number less than 2!')
        raw_input("Press Enter to exit...")
        exit(0)

    poly_params, error = pr3.train(
        predictions_l, predictions_u, predictions_v, results)

    # predict
    with open(predictJsonPath) as json_data:
        objs = json.load(json_data)

    count = 0
    totalError = 0

    for obj in objs:
        testSet = TrainingSet(obj)

        cv_image = cv2.imread(
            predictImagePath + testSet.imagePath, cv2.IMREAD_COLOR)
        background_image = cv2.imread(
            predictImagePath + testSet.bgImageInfoId, cv2.IMREAD_COLOR)

        if cv_image is None or background_image is None:
            print('Test image: ' + testSet.imagePath + ' cannot be found.')
            continue

        cv_normalized = QcImage.SNIC_EMOR(
            background_image, cv_image, PARAMS)

        dis_image = cv_normalized.copy()

        height, width, channels = cv_normalized.shape

        reference_bgr = None
        blk_reference_bgr = None

        for annotation in testSet.annotations:
            if annotation.type == '20':
                crop_image = QcImage.crop_image_by_position_and_rect(
                    cv_normalized, annotation.position, annotation.rect)
                reference_bgr = QcImage.get_average_rgb(crop_image)
            if annotation.type == '18':
                crop_image = QcImage.crop_image_by_position_and_rect(
                    cv_normalized, annotation.position, annotation.rect)
                blk_reference_bgr = QcImage.get_average_rgb(crop_image)

        for anno in testSet.annotations:
            if anno.type != '0':
                continue
            colour_area = QcImage.crop_image_by_position_and_rect(
                cv_normalized, anno.position, anno.rect)
            sample_bgr = QcImage.get_average_rgb(colour_area)
            colour_area = subtract_reference(
                colour_area, reference_bgr, blk_reference_bgr)
            colour_area = cv2.cvtColor(colour_area, cv2.COLOR_BGR2Lab)
            sample_lab = QcImage.get_average_rgb(colour_area)
            result = pr3.predict(
                sample_lab[0], sample_lab[1], sample_lab[2], poly_params)
            str_result = str(result)
            print(testSet.imagePath + ': ' + str_result + ', ' + anno.result)

            count += 1
            if anno.result is not None:
                totalError += abs(result - float(anno.result))

        # display predicting image and label
        if VIS:
            dis_image = cv2.cvtColor(dis_image, cv2.COLOR_BGR2RGB)

            plt.imshow(dis_image)
            plt.title(testSet.imagePath)
            plt.show()

    if totalError >= 0 and count is not 0:
        print('Average error: ' + str(totalError / count))

    input("Press Enter to exit...")
