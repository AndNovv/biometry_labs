import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def binary(img_path):
    img = Image.open(img_path).convert('RGB')
    bImg = []
    for i in range(img.size[0]):
        tmp = []
        for j in range(img.size[1]):
            t = img.getpixel((i,j))
            p = t[0] * 0.3 + t[1] * 0.59 + t[2] * 0.11
            if p > 128:
                p = 1
            else:
                p = 0
            tmp.append(p)
        bImg.append(tmp)
    return bImg

def tmpDelete(img):
    w = len(img)
    h = len(img[0])
    count = 1
    while count != 0:
        count = delete(img, w, h)
        if count:
            delete2(img, w, h)
    return img # Возвращаем измененное изображение

def delete(img, w, h):
    count = 0
    for i in range(1, h-1):
        for j in range(1, w-1):
            if img[j][i] == 0:
                if deletable(img, j, i):
                    img[j][i] = 1
                    count += 1
    return count

def delete2(img, w, h):
    for i in range(1, h-1):
        for j in range(1, w-1):
            if img[j][i] == 0:
                if deletable2(img, j, i):
                    img[j][i] = 1

def fringe(a):
    t = [[1,1,1,1,0,1,1,1,1],
    [1,1,1,1,0,1,1,0,0],
    [1,1,1,0,0,1,0,1,1],
    [0,0,1,1,0,1,1,1,1],
    [1,1,0,1,0,0,1,1,1],
    [1,1,1,1,0,1,0,0,1],
    [0,1,1,0,0,1,1,1,1],
    [1,0,0,1,0,1,1,1,1],
    [1,1,1,1,0,0,1,1,0],
    [1,1,1,1,0,1,0,0,0],
    [0,1,1,0,0,1,0,1,1],
    [0,0,0,1,0,1,1,1,1],
    [1,1,0,1,0,0,1,1,0]]

    for i in t:
        if a == i:
            return True
    
def check(a):
    t123457 = [1,1,0,0,1,0]
    t013457 = [1,1,1,0,0,0]
    t134567 = [0,1,0,0,1,1]
    t134578 = [0,0,0,1,1,1]
    t0123457 = [1,1,1,0,0,0,0]
    t0134567 = [1,0,1,0,0,1,0]
    t1345678 = [0,0,0,0,1,1,1]
    t1234578 = [0,1,0,0,1,0,1]
    
    t = [a[1],a[2],a[3],a[4],a[5],a[7]]
    if t == t123457:
        return True
    t = [a[0],a[1],a[3],a[4],a[5],a[7]]
    if t == t013457:
        return True
    t = [a[1],a[3],a[4],a[5],a[6],a[7]]
    if t == t134567:
        return True
    t = [a[1],a[3],a[4],a[5],a[7],a[8]]
    if t == t134578:
        return True
    t = [a[0],a[1],a[2],a[3],a[4],a[5],a[7]]
    if t == t0123457:
        return True
    t = [a[1],a[3],a[4],a[5],a[6],a[7],a[8]]
    if t == t1345678:
        return True
    t = [a[0],a[1],a[3],a[4],a[5],a[6],a[7]]
    if t == t0134567:
        return True
    t = [a[1],a[2],a[3],a[4],a[5],a[7],a[8]]
    if t == t1234578:
        return True
    
def deletable(img, x, y):
    a = []
    for i in range(y-1, y+2):
        for j in range(x-1, x+2):
            a.append(img[j][i])
    return check(a)

def deletable2(img, x, y):
    a = []
    for i in range(y-1, y+2):
        for j in range(x-1, x+2):
            a.append(img[j][i])
    return fringe(a)

def checkThisPoint(img, x, y):
    c = 0
    for i in range(x-1, x+1):
        for j in range(y-1, y+1):
    # for i in range(max(0, x-1), min(len(img), x+1)):
    #     for j in range(max(0, y-1), min(len(img[0]), y+1)):
            if img[i][j] == 0:
                c += 1
    return c - 1

def findCheckPoint(img):
    x, y = img.shape
    branchPoint = []
    endPoint = []
    for i in range(x):
        for j in range(y):
            if img[i][j] == 0:
                t = checkThisPoint(img, i, j)
                if t == 1:
                    endPoint.append((i,j))
                if t == 3:
                    branchPoint.append((i,j))
    return (branchPoint, endPoint)

def __removeDouble(x,y):
    z = []
    for i in x:
        c = True
        for j in y:
            if i == j:
                c = False
        if c:
            z.append(i)
    for i in y:
        c = True
        for j in x:
            if i == j:
                c = False
        if c:
            z.append(i)
    return z

def delNoisePoint(branch_points, end_points):
    tmp = []
    tmp2 = []
    for i in end_points:
        # x = range(i[0]-100, i[0]+101)
        # y = range(i[1]-100, i[1]+101)
        x = range(i[0]-20, i[0]+21)
        y = range(i[1]-20, i[1]+21)
        
        for j in branch_points:
            if j[0] in x and j[1] in y:
                tmp.append(i)
                tmp2.append(j)
    return (__removeDouble(branch_points, tmp2), __removeDouble(end_points, tmp))

pointRadius = 2

def drawKeyPoints(img, key_points):
    for point in key_points:
        cv2.circle(img, (point[1], point[0]), pointRadius, (0, 0, 255), -1)

def drawCleanKeyPoints(img, branch_points, end_points):
    img_with_points = np.copy(img)

    # Рисуем чистые точки в ветвлениях
    for point in branch_points:
        cv2.circle(img_with_points, (point[1], point[0]), pointRadius, (0, 0, 255), -1)

    # Рисуем чистые конечные точки
    for point in end_points:
        cv2.circle(img_with_points, (point[1], point[0]), pointRadius, (0, 0, 255), -1)

    # Отображаем изображение с чистыми точками
    plt.figure()

    plt.imshow(img_with_points, cmap='gray')
    plt.show(block=False)

def matchingPoint(r, v): #вход: кортеж точек эталона и кортеж проверяемого; выход (совпало, всего)
    all=0
    match=0
    for i in v[0]:
        # x=range(i[0]-15,i[0]+15)
        # y=range(i[1]-15,i[1]+15)
        x=range(i[0]-200,i[0]+200)
        y=range(i[1]-200,i[1]+200)
        all+=1
        for j in r[0]:
            if j[0] in x and j[1] in y:
                match+=1
                # break

    for i in v[1]:
        x=range(i[0]-15,i[0]+15)
        y=range(i[1]-15,i[1]+15)
        # x=range(i[0]-100,i[0]+100)
        # y=range(i[1]-100,i[1]+100)
        all+=1
    
        for j in r[1]:
            if j[0] in x and j[1] in y:
                match+=1
                break

    return (match,all)

# Предполагается, что у вас уже есть изображения img1 и img2
# Вместо них используйте свои изображения и не забудьте их бинаризовать

# Бинаризация изображений
binary_img1 = binary('images/101_1.png')
binary_img2 = binary('images/101_2.png')

plt.figure()
plt.imshow(binary_img1, cmap='gray')
plt.show(block=False)

plt.figure()
plt.imshow(binary_img2, cmap='gray')
plt.show(block=False)

# Создание копии изображений для скелетизации
binary_img1_copy = np.array(binary_img1, dtype=np.uint8)

binary_img2_copy = np.array(binary_img2, dtype=np.uint8)
# Применение алгоритма скелетизации
a1 = tmpDelete(binary_img1_copy)
a2 = tmpDelete(binary_img2_copy)

plt.figure()
plt.imshow(a1, cmap='gray')
plt.show(block=False)
plt.figure()
plt.imshow(a2, cmap='gray')
plt.show(block=False)

branch_points, end_points = findCheckPoint(a1)
branch_points2, end_points2 = findCheckPoint(a2)

drawKeyPoints(a1, branch_points + end_points)
drawKeyPoints(a2, branch_points2 + end_points2)

cleaned_points1 = delNoisePoint(branch_points, end_points)
cleaned_points2 = delNoisePoint(branch_points2, end_points2)

# Выводим изображение с чистыми точками для изображения a1
drawCleanKeyPoints(a1, cleaned_points1[0], cleaned_points1[1])

# Выводим изображение с чистыми точками для изображения a2
drawCleanKeyPoints(a2, cleaned_points2[0], cleaned_points2[1])

# Отображение исходного изображения с ключевыми точками
matched, total = matchingPoint(cleaned_points1, cleaned_points2)
print("Совпало точек: ", matched)
print("Всего точек в эталоне: ", total)

plt.show()
