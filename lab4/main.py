from database import addFingerPrintsToDB, findMostSimilarFingerprint, list_fingerprints_in_db, saveFingerprintToDatabase


if __name__ == "__main__":
    print("Начинаем обработку\n")

    # addFingerPrintsToDB(0, 30)
    list_fingerprints_in_db()

    # saveFingerprintToDatabase("Real\\500__M_Left_ring_finger.BMP")
    findMostSimilarFingerprint("Altered/Altered-Easy/22__M_Left_ring_finger_Obl.BMP")
    findMostSimilarFingerprint("Altered/Altered-Easy/101__M_Left_ring_finger_CR.BMP")
