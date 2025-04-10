import cv2

alteredImages = ["images/Altered/20__M_Left_thumb_finger_Obl.BMP",
                 "images/Altered/21__M_Left_index_finger_Zcut.BMP",
                 "images/Altered/22__M_Left_index_finger_CR.BMP",
                 "images/Altered/23__M_Right_index_finger_rotate.BMP"]

realImages = ["images/Real/20__M_Left_thumb_finger.BMP",
              "images/Real/21__M_Left_index_finger.BMP",
              "images/Real/22__M_Left_index_finger.BMP",
              "images/Real/23__M_Right_index_finger.BMP",]


def compareImages(img1, img2):
    fingerprint_image1 = cv2.imread(img1)
    fingerprint_image2 = cv2.imread(img2)

    sift = cv2.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(fingerprint_image1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(fingerprint_image2, None)

    matches = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 10}, {}).knnMatch(descriptors_1, descriptors_2, k=2)

    match_points = []

    for p, q in matches:
        if p.distance < 0.7 * q.distance:
            match_points.append(p)

    keypoints = 0
    if len(keypoints_1) < len(keypoints_2):
        keypoints = len(keypoints_1)
    else:
        keypoints = len(keypoints_2)

    score = len(match_points) / keypoints * 100

    print("\nОтпечаток 1: " + img1)
    print("Отпечаток 2: " + img2)
    print("Результат сравнения отпечатков: " + str(score))

    if score < 20:
        print("Отпечатки не совпадают")
    elif score < 40:
        print("Требуется дополнительный анализ")
    else:
        print("Отпечатки совпадают")


    fingerprint1_resize = cv2.resize(fingerprint_image1, None, fx=3, fy=3)
    fingerprint2_resize = cv2.resize(fingerprint_image2, None, fx=3, fy=3)

    cv2.imshow("Fingerprint 1", fingerprint1_resize)
    cv2.imshow("Fingerprint 2", fingerprint2_resize)

    result = cv2.drawMatches(fingerprint_image1, keypoints_1, fingerprint_image2, keypoints_2, match_points, None)
    result = cv2.resize(result, None, fx=3, fy=3)
    cv2.imshow("keypoints comparison", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return score

compareImages(realImages[0], alteredImages[0])