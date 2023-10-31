import cv2
import numpy as np


def resizeImage(image, scale):

    scale_percent = scale
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)

    return cv2.resize(image, dim)

def detect_corners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    corners = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    corners = cv2.dilate(corners, None)
    image[corners > 0.01 * corners.max()] = [0, 0, 255]

    return image


# Función para detectar y emparejar parches utilizando el algoritmo ORB
def detect_and_match(image, object_template):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(object_template, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()

    kp_image, des_image = orb.detectAndCompute(gray_image, None)
    kp_template, des_template = orb.detectAndCompute(gray_template, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_template, des_image)
    matches = sorted(matches, key=lambda x: x.distance)

    matched_image = cv2.drawMatches(object_template, kp_template, image, kp_image, matches[:10], outImg=None)

    return matched_image


if __name__ == "__main__":
    main_image = cv2.imread('./Images/base.jpg')
    object_template = cv2.imread('./Images/test1.jpg')
    main_image_with_corners = detect_corners(main_image)
    matched_image = detect_and_match(main_image, object_template)

    matchImageSmall = resizeImage(matched_image, 60)
    cornerImgSmall = resizeImage(main_image_with_corners, 60)

    cv2.imshow('Main Image with Corners', cornerImgSmall)
    cv2.imshow('Matched Image', matchImageSmall)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
