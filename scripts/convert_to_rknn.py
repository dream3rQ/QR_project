#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from rknn.api import RKNN
import os

# Настройки путей
ONNX_MODEL = 'model/best.onnx'  # Путь к исходнику
RKNN_MODEL = 'model/best.rknn'  # Куда сохранить результат

def convert():
    # Создаем объект RKNN
    rknn = RKNN(verbose=True)

    # Конфигурация для Orange Pi 5 (чип RK3588)
    rknn.config(
        mean_values=[[0, 0, 0]], 
        std_values=[[255, 255, 255]], 
        target_platform='rk3588'
    )

    # Загрузка ONNX модели
    print('Loading ONNX model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load model failed!')
        return

    # Построение модели 
    print('Building model')
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print('Build model failed!')
        return

    # Экспорт в формат RKNN
    print('Exporting RKNN model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export failed!')
        return

    print('Model converted successfully.')
    rknn.release()

if __name__ == '__main__':
    convert()

