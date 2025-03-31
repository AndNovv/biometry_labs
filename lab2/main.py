import cv2
import numpy as np 

import dlib

# Загрузка предобученной модели для обнаружения ключевых точек лица 
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Загрузка изображения
image = cv2.imread('image2.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Обнаружение лиц на изображении
detector = dlib.get_frontal_face_detector()
faces = detector(gray_image)

# Для каждого обнаруженного лица 

for face in faces:
    # Обнаружение ключевых точек лица  
    landmarks = predictor(gray_image, face)


    # Определение координат области глаз
    left_eye_region = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
    right_eye_region = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])


    # Определение минимальных и максимальных координат каждого глаза
    left_x_min, left_y_min = left_eye_region[:, 0].min(), left_eye_region[:, 1].min() 
    left_x_max, left_y_max = left_eye_region[:, 0].max(), left_eye_region[:, 1].max() 
    right_x_min, right_y_min = right_eye_region[:, 0].min(), right_eye_region[:, 1].min() 
    right_x_max, right_y_max = right_eye_region[:, 0].max(), right_eye_region[:, 1].max()

    # Определение центра каждого глаза
    left_eye_center = ((left_x_max + left_x_min) // 2, (left_y_max + left_y_min) // 2) 
    right_eye_center = ((right_x_max + right_x_min) // 2, (right_y_max + right_y_min) //2)


    left_roi = image[left_y_min:left_y_max, left_x_min:left_x_max] 
    right_roi = image[right_y_min:right_y_max, right_x_min:right_x_max] 
    rows,cols, _ = left_roi.shape
    row,col, _ = right_roi.shape
    # Преобразование ROI в оттенки серого и размытие Гаусса для левого глаза 
    gray_left_roi = cv2.cvtColor(left_roi, cv2.COLOR_BGR2GRAY)

    # Преобразование ROI в оттенки серого и размытие Гаусса для правого глаза 
    gray_right_roi = cv2.cvtColor(right_roi, cv2.COLOR_BGR2GRAY)

    _, threshold_left = cv2.threshold(gray_left_roi, 25, 255, cv2.THRESH_BINARY_INV) # Бинаризация изображения для правого глаза
    _, threshold_right = cv2.threshold(gray_right_roi, 25, 255, cv2.THRESH_BINARY_INV) 
    contours_left, _ = cv2.findContours(threshold_left, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours_left = sorted(contours_left, key=lambda x: cv2.contourArea(x), reverse=True)


    largest_contour_left = max(contours_left, key=cv2.contourArea)
 
    x_left, y_left, w_left, h_left = cv2.boundingRect(largest_contour_left) 
    center_left = (x_left + w_left // 2, y_left + h_left // 2)
    
    # Сравнить центр глаза с центром области интереса
    if left_x_min + center_left[0] <= left_x_min + 0.4 * (left_x_max - left_x_min): 
        print("Направление левого глаза: лево")
    elif left_x_min + center_left[0] >= left_x_max - 0.4 * (left_x_max - left_x_min): 
        print("Направление левого глаза: право")
    else:
        print("Направление левого глаза: центр")



    # Отображение изображения с контурами 
    for cnt_left in contours_left:
        (x, y, w, h) = cv2.boundingRect(cnt_left) 
        cv2.rectangle(left_roi, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.line(left_roi, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 2)
        cv2.line(left_roi, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 2) 
        break # Рассматриваем только самый большой контур радужки

    contours_right, _ = cv2.findContours(threshold_right, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_right = sorted(contours_right, key=lambda x: cv2.contourArea(x), reverse=True)
    largest_contour_right = max(contours_right, key=cv2.contourArea)


    x_right, y_right, w_right, h_right = cv2.boundingRect(largest_contour_right) 
    center_right = (x_right + w_right // 2, y_right + h_right // 2)


    if right_x_min + center_right[0] <= right_x_min + 0.3 * (right_x_max - right_x_min): 
        print("Направление правого глаза: лево")
    elif right_x_min + center_right[0] >= right_x_max - 0.3 * (right_x_max - right_x_min):
        print("Направление правого глаза: право") 
    else:
        print("Направление правого глаза: центр")


    for cnt_right in contours_right:
        (x, y, w, h) = cv2.boundingRect(cnt_right) 
        cv2.rectangle(right_roi, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.line(right_roi, (x + int(w/2), 0), (x + int(w/2), row), (0, 255, 0), 2)
        cv2.line(right_roi, (0, y + int(h/2)), (col, y + int(h/2)), (0, 255, 0), 2) 
        break # Рассматриваем только самый большой контур радужки

    # Отображение изображения с выделенными глазами 
    cv2.imshow('Detected Eyes', image) 
    cv2.waitKey(0)
    key = cv2.waitKey(30) 
    if key == 27:
        cv2.destroyAllWindows()
