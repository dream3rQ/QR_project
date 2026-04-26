#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
from rknnlite.api import RKNNLite
from pyzbar.pyzbar import decode

# Настройки
MODEL_PATH = '/home/user/best.rknn'
W, H = 1280, 736

def order_points(pts):
    rect = np.zeros((4,2), dtype='float32')
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[2] = pts[np.argmax(diff)]
    rect[3] = pts[np.argmax(s)]
    return rect

# Инициализация NPU
rknn = RKNNLite()
print("Загрузка модели в NPU...")
if rknn.load_rknn(MODEL_PATH) != 0:
    print("Ошибка загрузки"); exit()
if rknn.init_runtime() != 0:
    print("Ошибка NPU"); exit()

cap = cv2.VideoCapture(0) 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Подготовка для NPU
    img = cv2.resize(frame, (W, H))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ИНФЕРЕНС (Работает NPU)
    outputs = rknn.inference(inputs=[img])
    
    # Пост-обработка тензора YOLOv8-pose
    preds = np.squeeze(outputs[0])
    scores = preds[4, :]
    best_idx = np.argmax(scores)
    
    if scores[best_idx] > 0.5:
        # Извлекаем 4 точки [x, y, v]
        kpts = preds[5:, best_idx].reshape(4, 3)
        h_orig, w_orig = frame.shape[:2]
        src_pts = []
        for x, y, v in kpts:
            src_pts.append([x * w_orig / W, y * h_orig / H])
        
        src_pts = order_points(np.array(src_pts, dtype="float32"))

        # Выпрямление
        side = 300
        dst_pts = np.float32([[0, 0], [side, 0], [side, side], [0, side]])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(frame, M, (side, side))

        # Чтение QR
        qr_data = decode(warped)
        if qr_data:
            msg = qr_data[0].data.decode('utf-8')
            print(f"ПРОЧИТАНО: {msg}")
            cv2.putText(frame, msg, (50, 50), 1, 2, (0, 255, 0), 2)

        for pt in src_pts:
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)

    cv2.imshow('NPU QR scanner', frame)
    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()
rknn.release()

