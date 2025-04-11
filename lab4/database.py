import json
import os
import cv2
import numpy as np

from fingerprint_processing import calculateKeypointsAndDescriptors, findMatchPoints
from utils import deserialize_keypoints, dispayComparisonResult, serialize_keypoints

def addFingerPrintsToDB(start_index=0, count=10, real_folder="Real", db_path="fingerprint_db.json"):

    all_files = [
        f for f in os.listdir(real_folder)
    ]

    selected_files = all_files[start_index:start_index + count]

    if not selected_files:
        print("Нет файлов для обработки.")
        return

    for i, filename in enumerate(selected_files, start=1):
        
        image_path = os.path.join(real_folder, filename)
        saveFingerprintToDatabase(image_path, db_path)

    print("\nОбработка завершена.")


def saveFingerprintToDatabase(image_path, db_path='fingerprint_db.json'):

    db = load_database(db_path)
    if is_already_processed(db, image_path):
        print(f"Отпечаток уже есть в базе: {image_path}")
        return
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Ошибка: изображение '{image_path}' не найдено.")
        return

    keypoints, descriptors = calculateKeypointsAndDescriptors(image)
    serialized_kp = serialize_keypoints(keypoints)
    serialized_desc = descriptors.tolist() if descriptors is not None else []

    new_entry = {
        "image_path": image_path,
        "keypoints": serialized_kp,
        "descriptors": serialized_desc
    }
    db.append(new_entry)
    save_database(db, db_path)
    print(f"Отпечаток сохранён в базе: {image_path}")



def findMostSimilarFingerprint(image_path, db_path='fingerprint_db.json'):
    if not os.path.exists(db_path):
        print("База данных не найдена.")
        return

    fingerprint1 = cv2.imread(image_path)
    if fingerprint1 is None:
        print(f"Ошибка: изображение '{image_path}' не найдено.")
        return

    keypoints_1, descriptors_1 = calculateKeypointsAndDescriptors(fingerprint1)
    if descriptors_1 is None or len(keypoints_1) == 0:
        print("Не удалось извлечь дескрипторы из изображения.")
        return

    db = load_database()

    best_score = 0
    best_match = None

    print(f"\nНачинаем поиск отпечатка для {image_path}")

    for entry in db:
        if not entry['descriptors']:
            continue

        des_db = np.array(entry['descriptors'], dtype=np.float32)
        kp_db = deserialize_keypoints(entry['keypoints'])

        match_points = findMatchPoints(descriptors_1, des_db)
        keypoints_count = min(len(keypoints_1), len(kp_db))

        if keypoints_count == 0:
            continue

        score = len(match_points) / keypoints_count * 100

        if score > best_score:
            best_score = score
            best_match = {
                "image_path": entry["image_path"],
                "keypoints": kp_db,
                "descriptors": des_db,
                "match_points": match_points
            }

    if best_match and best_score > 40:
        print(f'Наиболее похожий отпечаток: {best_match["image_path"]}')
        print(f"Результаты сравнения отпечатков:")

        fingerprint2 = cv2.imread(best_match["image_path"])
        dispayComparisonResult(
            image_path,
            best_match["image_path"],
            best_score,
            fingerprint1,
            keypoints_1,
            fingerprint2,
            best_match["keypoints"],
            best_match["match_points"]
        )
    else:
        print(f"Совпадений в базе данных не найдено.")


def list_fingerprints_in_db():
    db = load_database()
    if not db:
        print("База данных пуста.")
        return
    
    print("Файлы в базе данных:")
    for entry in db:
        print(entry["image_path"])

def load_database(db_path="fingerprint_db.json"):
    if os.path.exists(db_path):
        with open(db_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_database(db, db_path="fingerprint_db.json"):
    with open(db_path, 'w', encoding='utf-8') as f:
        json.dump(db, f, indent=4)

def is_already_processed(db, image_path):
    return any(entry["image_path"] == image_path for entry in db)
