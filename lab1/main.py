import cv2

# Загрузка классификатора для обнаружения лиц
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Загрузка изображений
image = cv2.imread('image1.jpg')
image_2 = cv2.imread('image2.jpg')

# Проверка загрузки изображений
if image is None or image_2 is None:
    print("Ошибка: не удалось загрузить одно или оба изображения.")
    exit()

# Преобразование изображения в оттенки серого
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
# Обнаружение лиц на изображении
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5,
minSize=(30, 30))
faces_2 = face_cascade.detectMultiScale(gray_image_2, scaleFactor=1.1, minNeighbors=5,
minSize=(30, 30))
# Отрисовка прямоугольника вокруг обнаруженных лиц
for (x, y, w, h) in faces:
 cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
for (x, y, w, h) in faces_2:
 cv2.rectangle(image_2, (x, y), (x+w, y+h), (155, 0, 0), 2)
# Отображение изображения с выделенными лицами
4
cv2.imshow('Detected 1 Faces', image)
cv2.imshow('Detected 2 Faces', image_2)
cv2.waitKey(0)
cv2.destroyAllWindows()