import cv2    

def dispalayfingerPrint(winname, img, scale):
    fingerprint = cv2.resize(cv2.imread(img), None, fx=scale, fy=scale)
    cv2.imshow(winname, fingerprint)


def dispayComparisonResult(img1, img2, score, fingerprint1, keypoints_1, fingerprint2, keypoints_2, match_points):
    
    print("\nОтпечаток 1: " + img1)
    print("Отпечаток 2: " + img2)
    print("Результат сравнения отпечатков: " + str(score))

    if score < 20:
        print("Отпечатки не совпадают")
    elif score < 40:
        print("Требуется дополнительный анализ")
    else:
        print("Отпечатки совпадают")


    dispalayfingerPrint("Fingerpirnt1", img1, 3)
    dispalayfingerPrint("Fingerprint2", img2, 3)

    result = cv2.drawMatches(fingerprint1, keypoints_1, fingerprint2, keypoints_2, match_points, None)
    result = cv2.resize(result, None, fx=3, fy=3)

    cv2.imshow("keypoints comparison", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 


def serialize_keypoints(keypoints):
    return [
        {
            'pt': kp.pt,
            'size': kp.size,
            'angle': kp.angle,
            'response': kp.response,
            'octave': kp.octave,
            'class_id': kp.class_id
        } for kp in keypoints
    ]

def deserialize_keypoints(data):
    return [
        cv2.KeyPoint(
            kp['pt'][0],           # x
            kp['pt'][1],           # y
            kp['size'],            # size
            kp['angle'],           # angle
            kp['response'],        # response
            kp['octave'],          # octave
            kp['class_id']         # class_id
        ) for kp in data
    ]
