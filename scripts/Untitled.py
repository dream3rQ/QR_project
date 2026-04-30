#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from rknn.api import RKNN
import os

# 1. Настройки путей
ONNX_MODEL = 'model/best.onnx'  # Путь к исходнику
RKNN_MODEL = 'model/best.rknn'  # Куда сохранить результат

def convert():
    # Создаем объект RKNN
    rknn = RKNN(verbose=True)

    # 2. Конфигурация для Orange Pi 5 (чип RK3588)
    print('--> Config model for RK3588')
    rknn.config(
        mean_values=[[0, 0, 0]], 
        std_values=[[255, 255, 255]], 
        target_platform='rk3588'
    )

    # 3. Загрузка ONNX модели
    print('--> Loading ONNX model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load model failed!')
        return

    # 4. Построение модели (Building)
    # do_quantization=False означает экспорт в FP16 (высокая точность)
    print('--> Building model...')
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print('Build model failed!')
        return

    # 5. Экспорт в формат RKNN
    print('--> Exporting RKNN model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export failed!')
        return

    print('DONE! Model converted successfully.')
    rknn.release()

if __name__ == '__main__':
    convert()

