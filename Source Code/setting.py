#Test camera có vấn đề gì không
import cv2

# Khởi tạo camera
cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

while True:
    ret, img = cam.read()
    if not ret:
        print("Không thể nhận khung hình từ camera")
        break

    cv2.imshow('Image', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Ấn 'q' để thoát
cam.release()
cv2.destroyAllWindows()
