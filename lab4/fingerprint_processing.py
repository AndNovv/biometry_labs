import cv2
from utils import dispayComparisonResult

def findMatchPoints(descriptors_1, descriptors_2):
    matches = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 10}, {}).knnMatch(descriptors_1, descriptors_2, k=2)
    match_points = []
    for p, q in matches:
        if p.distance < 0.7 * q.distance:
            match_points.append(p)

    return match_points

def calculateKeypointsAndDescriptors(fingerprint):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(fingerprint, None)
    return keypoints, descriptors


def compareFingerprints(img1, img2, displayResults=False):

    fingerprint1 = cv2.imread(img1)
    fingerprint2 = cv2.imread(img2)

    keypoints_1, descriptors_1 = calculateKeypointsAndDescriptors(fingerprint1)
    keypoints_2, descriptors_2 = calculateKeypointsAndDescriptors(fingerprint2)

    match_points = findMatchPoints(descriptors_1, descriptors_2)

    keypoints = min(len(keypoints_1), len(keypoints_2))
    score = len(match_points) / keypoints * 100

    if displayResults: 
        dispayComparisonResult(img1, img2, score, fingerprint1, keypoints_1, fingerprint2, keypoints_2, match_points)

    return score