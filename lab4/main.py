from database import addFingerPrintsToDB, findMostSimilarFingerprint, list_fingerprints_in_db, saveFingerprintToDatabase


if __name__ == "__main__":
    print("Начинаем обработку\n")

    addFingerPrintsToDB(0, 50)
    list_fingerprints_in_db()

    # saveFingerprintToDatabase("Real\\500__M_Left_ring_finger.BMP")
    findMostSimilarFingerprint("Altered/Altered-Easy/400__M_Left_ring_finger_Obl.BMP")
    # findMostSimilarFingerprint("Altered/Altered-Hard/100__M_Left_middle_finger_Obl.BMP")
