#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2020/12/19 14:50
# @Author  : dly
# @File    : train.py
# @Desc    :
from keras_preprocessing.image import ImageDataGenerator

import os
import efficientnet.keras as efn
from keras import layers, models
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 数据基础目录
data_path = 'C:/Users/Dooooooooo21/Desktop/project/cassava-leaf-disease-classification/'

# labels
train_labels = pd.read_csv(data_path + 'train.csv')


# print(train_labels.head())


# 分类情况统计
def plt_stat():
    sns.countplot(train_labels.label, edgecolor='black',
                  palette=reversed(sns.color_palette("viridis", 5)))
    plt.xlabel('Classes', fontfamily='serif', size=15)
    plt.ylabel('Count', fontfamily='serif', size=15)
    plt.show()


# batch 大小
BATCH_SIZE = 4
# 每个世代 step数
STEPS_PER_EPOCH = len(train_labels) * 0.8 / BATCH_SIZE
VALIDATION_STEPS = len(train_labels) * 0.2 / BATCH_SIZE
# 训练世代
EPOCHS = 20
# 图片目标大小
TARGET_SIZE = 512

# 训练数据和验证数据生成器
train_labels.label = train_labels.label.astype('str')

train_datagen = ImageDataGenerator(validation_split=0.2,
                                   preprocessing_function=None,
                                   rotation_range=45,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   fill_mode='nearest',
                                   shear_range=0.1,
                                   height_shift_range=0.1,
                                   width_shift_range=0.1)

train_generator = train_datagen.flow_from_dataframe(train_labels,
                                                    directory=os.path.join(data_path, "train_images"),
                                                    subset="training",
                                                    x_col="image_id",
                                                    y_col="label",
                                                    target_size=(TARGET_SIZE, TARGET_SIZE),
                                                    batch_size=BATCH_SIZE,
                                                    class_mode="sparse")

validation_datagen = ImageDataGenerator(validation_split=0.2)

validation_generator = validation_datagen.flow_from_dataframe(train_labels,
                                                              directory=os.path.join(data_path, "train_images"),
                                                              subset="validation",
                                                              x_col="image_id",
                                                              y_col="label",
                                                              target_size=(TARGET_SIZE, TARGET_SIZE),
                                                              batch_size=BATCH_SIZE,
                                                              class_mode="sparse")


# 模型
def create_model():
    conv_base = efn.EfficientNetB0(weights='imagenet', include_top=False, input_shape=(TARGET_SIZE, TARGET_SIZE, 3))
    model = conv_base.output
    model = layers.GlobalAveragePooling2D()(model)
    model = layers.Dense(5, activation="softmax")(model)
    model = models.Model(conv_base.input, model)

    model.compile(optimizer=Adam(lr=0.001),
                  loss="sparse_categorical_crossentropy",
                  metrics=["acc"])
    return model


model = create_model()
model.load_weights('model/ep02-loss0.484-val_loss0.021-val_acc_0.85.h5')
# model.summary()

# 训练
model_save = ModelCheckpoint('model/' + 'ep{epoch:02d}-l{loss:.3f}-a{acc:.3f}-v_l{val_loss:.3f}-v_a{val_acc:.3f}.h5',
                             save_best_only=True,
                             save_weights_only=True,
                             monitor='val_acc',
                             mode='auto', verbose=1)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001,
                           patience=5, mode='min', verbose=1,
                           restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                              patience=2, min_delta=0.001,
                              mode='min', verbose=1)

history = model.fit(
    train_generator,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=VALIDATION_STEPS,
    callbacks=[model_save, early_stop, reduce_lr]
)
