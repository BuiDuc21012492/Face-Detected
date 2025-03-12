import cv2
import os

# Khởi tạo camera
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # Chiều rộng video (đơn vị pixel)
cam.set(4, 480)  # Chiều cao video (đơn vị pixel)

# Kiểm tra nếu camera không mở được
if not cam.isOpened():
    print("Không thể mở camera")
    exit()

# Kiểm tra tệp cascade classifier
cascade_path = 'haarcascade_frontalface_default (1).xml'
if not os.path.isfile(cascade_path):
    print(f"Tệp {cascade_path} không tồn tại")
    exit()

face_detector = cv2.CascadeClassifier(cascade_path)

face_id = input('\nNhập ID khuôn mặt ( Các ID không được trùng nhau ): ')
print("\nĐang khởi tạo Camera....")
count = 0

#Bắt đầu 1 khối có thể có ngoại lệ
try:
    while True:
        ret, img = cam.read() # Đọc khung hình và lưu vào biến img, ret chỉ ra là dữ liệu được lưu hay chưa
        if not ret:
            print("Không thể nhận khung hình từ camera")
            break

        img = cv2.flip(img, 1)  # Lật hình ảnh theo trục y (lật ngang)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Khoanh vùng khuôn mặt nếu được phát hiện
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1

            # Lưu ảnh vào thư mục dataset dưới điịnh dạng jpn và ảnh xám
            cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h, x:x+w])
            cv2.imshow('Image', img)

        k = cv2.waitKey(100) & 0xff
        if k == 27:  # Nhấn ESC để thoát
            break
        elif count >= 50:  # Lưu 50 hình ảnh khuôn mặt rồi thoát
            break

#xử lý ngoại lệ
except KeyboardInterrupt:
    print("Ngắt chương trình")

#Thực hiện cho dù có ngoại lệ hay không
finally:
    print("\nThoát")
    cam.release()
    cv2.destroyAllWindows()
