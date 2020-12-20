#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2020/12/20 11:58
# @Author  : dly
# @File    : predict.py
# @Desc    :
from train import create_model, data_path, TARGET_SIZE
import pandas as pd
import numpy as np
import os
from PIL import Image

model = create_model()
model.load_weights('model/ep02-loss0.484-val_loss0.021-val_acc_0.85.h5')

ss = pd.read_csv(os.path.join(data_path, "sample_submission.csv"))
preds = []

print('-----------------')
for image_id in ss.image_id:
    image = Image.open(os.path.join(data_path, "test_images", image_id))
    image = image.resize((TARGET_SIZE, TARGET_SIZE))
    image = np.expand_dims(image, axis=0)
    preds.append(np.argmax(model.predict(image)))

ss['label'] = preds
ss.to_csv('submission.csv', index=False)
