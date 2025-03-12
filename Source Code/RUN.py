import cv2
import numpy as np
import os

recognizer = cv2.face.LBPHFaceRecognizer_create() #Khởi tạo hàm
recognizer.read('trainer/trainer.yml') # Đọc d liệu từ file đã train

cascadePath = "haarcascade_frontalface_default (1).xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX

# Khởi tạo id = 0
id = 0

# Danh sách tên tương ứng với ID [0,1,2,3,...]
# Đổi tên trong chuỗi names phù hợp với mục đích sử dụng
names = ['0', '1', '2', '3']

# Khởi tạo camera
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # Chiều rộng video
cam.set(4, 480)  # Chiều cao video

# Xác định kích thước tối thiểu của cửa sổ phát hiện khuôn mặt
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)
#Bắt đầu 1 khối có thể có ngoại lệ
try:
    while True:
        ret, img = cam.read() # Đọc khung hình và lưu vào biến img, ret chỉ ra là dữ liệu được lưu hay chưa
        if not ret:
            print("Không thể nhận khung hình từ camera")
            break

        # Lật hình ảnh theo trục y (lật ngang)
        img = cv2.flip(img, 1)

        # Chuyển ảnh sang màu xám
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Phát hiện khuôn mặt có trong hình ảnh xám
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )

        for (x, y, w, h) in faces:
            # Vẽ hình chữ nhật lên ảnh
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Dự đoán id và mức độ tin cậy của khuôn mặt được nhận diện
            id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

            if confidence < 100:
                id = names[id]
                # Hiển thị mức độ tin cậy
                confidence_text = "{0}%".format(round(100 - confidence))
            else:
                #Không nhận diện được
                id = "unknown"
                confidence_text = "{0}%".format(round(100 - confidence))

            # vẽ lên ảnh, id của người nhận diện. Tọa độ (x,y) của góc dưới cùng bên trái
            cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, str(confidence_text), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

        cv2.imshow("Nhan dien khuon mat", img)

        k = cv2.waitKey(10) & 0xff  # Chờ 1 phím được ấn trong 10ms. Nhấn 'ESC' để thoát
        if k == 27:
            break
#xử lý ngoại lệ
except KeyboardInterrupt:
    print("Ngắt chương trình")

#Thực hiện cho dù có ngoại lệ hay không
finally:
    print("\nThoat")
    cam.release()
    cv2.destroyAllWindows()
