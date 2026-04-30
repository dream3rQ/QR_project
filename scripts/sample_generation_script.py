#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import qrcode
import numpy as np
import random
import os
from pathlib import Path


# In[2]:


output_dir = Path('QR_dataset')
images_dir = output_dir / 'train/images'
labels_dir = output_dir / 'train/labels'
os.makedirs(images_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)


# In[3]:


def generate_random_qr():
    # Генерируем случайную строку для qr
    data = "".join(random.choices('qwertyuiopasdfghjklzxcvbnm1234567890', k=10))
    qr = qrcode.QRCode(box_size = 10, border = 1)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color='black', back_color='white').convert('RGB')
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# In[4]:


import matplotlib.pyplot as plt


# In[5]:


chiki = generate_random_qr()
chiki.shape
plt.imshow(chiki)
plt.axis('off')


# In[6]:


def create_yolo_line(dst_pts, bg_width=640, bg_height=640, class_id=0):

    x_coords = np.clip(dst_pts[:, 0], 0, bg_width)
    y_coords = np.clip(dst_pts[:, 1], 0, bg_height)

    xmin, xmax = np.min(x_coords), np.max(x_coords)
    ymin, ymax = np.min(y_coords), np.max(y_coords)

    w = (xmax - xmin) / bg_width
    h = (ymax - ymin) / bg_height
    xc = (xmin + (xmax - xmin)/2) / bg_width
    yc = (ymin + (ymax - ymin)/2) / bg_height

    line = f'{class_id} {np.clip(xc, 0, 1):.6f} {np.clip(yc, 0, 1):.6f} {np.clip(w, 0, 1):.6f} {np.clip(h, 0, 1):.6f}'

    for i in dst_pts:

        if 0 <= i[0] <= bg_width and 0 <= i[1] <= bg_height:
            v = 2
        else:
            v = 0

        px_n = np.clip((i[0] / bg_width), 0, 1)
        py_n = np.clip((i[1] / bg_height), 0, 1)

        line += f' {px_n:.6f} {py_n:.6f} {v}'

    return line


# In[7]:


# Просто получаем список путей к файлам
folder_path = './nature_images'
bg_image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.jpg')]

print(f"Найдено фонов: {len(bg_image_paths)}")


# In[11]:


def create_sample(bg_width=640, bg_height=640):

    BASE_DIR = 'QR_dataset'
    BG_IMAGES = f'./nature_images/*.jpg'
    NUM_IMAGES = 3000
    TRAIN_RATIO = 0.8

    # создаем папки
    for split in ['train', 'val']:
        os.makedirs(f'{BASE_DIR}/{split}/images', exist_ok=True)
        os.makedirs(f'{BASE_DIR}/{split}/labels', exist_ok=True)



    for i in range(NUM_IMAGES):

        split = 'train' if i < NUM_IMAGES * TRAIN_RATIO else 'val'

        # цвентой фон
        random_path = random.choice(bg_image_paths)
        bg = np.zeros((bg_height, bg_width, 3), dtype=np.uint8)
        if cv2.imread(random_path) is not None:
            bg[:] = cv2.resize(cv2.imread(random_path), (bg_width, bg_height)) 
        else:
            bg[:] = np.random.randint(50, 200, (3,))
        # Генерируем чистый QR code
        qr = generate_random_qr()
        q_h, q_w = qr.shape[:2] # высота и ширина qr кода
        # Выбираем случайную точку, куда поместим QR на фоне
        size = random.randint(100, 250)
        cy = random.randint(size//2, bg_height - size//2)
        cx = random.randint(size//2, bg_width - size//2)
        # Выставляем углы на фоне
        off = size//4

        dst_pts = np.float32([
          [cx - size//2 + random.randint(-off, off), cy - size//2 + random.randint(-off, off)], # TL
          [cx + size//2 + random.randint(-off, off), cy - size//2 + random.randint(-off, off)], # TR
          [cx - size//2 + random.randint(-off, off), cy + size//2 + random.randint(-off, off)], # BL
          [cx + size//2 + random.randint(-off, off), cy + size//2 + random.randint(-off, off)], # BR
        ])

        src_pts = np.float32([[0, 0], [q_w, 0], [0, q_h], [q_w, q_h]])

        change = cv2.getPerspectiveTransform(src_pts, dst_pts)
        changed_qr = cv2.warpPerspective(qr, change, (bg_width, bg_height), borderValue=(0, 0, 0))
        # Маска
        mask = cv2.warpPerspective(np.ones_like(qr)*255, change, (bg_width, bg_height)) # белый 4угольник на черном фоне
        masked_bg = cv2.bitwise_and(bg, cv2.bitwise_not(mask)) # цветной фон с черным 4угольником
        final = cv2.add(masked_bg, changed_qr)

        # добавляем говнецо
        # блюр
        if random.random() > 0.5:
            k_size = random.choice([11, 17])
            final = cv2.GaussianBlur(final, (k_size, k_size), 0)

        if random.random() > 0.1:
            noise = np.random.randint(0, 30, final.shape, dtype='uint8')
            final = cv2.add(final, noise)

        if random.random() > 0.7:
            overlay = final.copy()
            overlay = cv2.circle(overlay, (random.randint(0, bg_width), random.randint(0, bg_height)),
                                 random.randint(50, 300), [255, 255, 255], -1)
            overlay = cv2.GaussianBlur(overlay, (99, 99), 0)
            final = cv2.addWeighted(final, 0.4, overlay, 0.6, 1)


        file_name = f'qr_sample_{i}'
        cv2.imwrite(f'{BASE_DIR}/{split}/images/{file_name}.jpg', final)

        label_str = create_yolo_line(dst_pts)
        with open(f'{BASE_DIR}/{split}/labels/{file_name}.txt', 'w') as f:
            f.write(label_str)


    return print('готово')



# In[12]:


create_sample()


# In[13]:


pass_to_train_image = ('./QR_dataset/train/images')
pass_to_val_image = ('./QR_dataset/val/images')
train_image_paths = [os.path.join(pass_to_train_image, f) for f in os.listdir(pass_to_train_image) if f.endswith('.jpg')]
val_image_paths = [os.path.join(pass_to_val_image, f) for f in os.listdir(pass_to_val_image) if f.endswith('.jpg')]
print(f'В трейне: {len(train_image_paths)}')
print(f'В вале: {len(val_image_paths)}')


# In[ ]:




