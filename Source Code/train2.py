import cv2
import numpy as np
from PIL import Image
import os

# Khai báo thư mục dataset
path = 'dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default (1).xml")

#Tạo hàm lấy nhãn của hình ảnh
def getImgAndLabels(path):
    # Lấy id ảnh
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []

    # DÙng for để lấy id từng bức ảnh
    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L') #convert it to grayscale
        img_numpy = np.array(PIL_img, 'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)

    return faceSamples, ids
print("\nTraining...")
faces, ids = getImgAndLabels(path)
recognizer.train(faces, np.array(ids))
# Lưu dữ liệu vào file trainer sau khi lấy được id ảnh
recognizer.write('trainer/trainer.yml')

print("\nKhuôn mặt đã được train. ".format(len(np.unique(ids))))