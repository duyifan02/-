"""
太原理工大学信息与光电工程学院《多媒体通信技术》课程
项目三 基于深度学习的Udacity无人驾驶系统

 学号：2020002030
 班级：信息2002
 姓名：杜艺帆
 完成工作：训练模型；
"""
# import third libs
import cv2, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from data_preprocessing import image_height, image_width, image_channels, image_preprocessing, image_normalized
from build_model import build_model1, build_model2, build_model3

# initialize variable
data_path = './lake_data/'
# data_path = './mountain_data/'
test_ration = 0.1

# batch_size = 10
# batch_num = 30
# epochs = 10

batch_size = 100
batch_num = 200
epochs = 100


# import data
def data_load(data_path):
    data_csv = pd.read_csv(data_path + 'driving_log.csv',
                           names=['center', 'left', 'right', 'steering', '_', '__', '___'])
    # print(data_csv)
    print(data_csv.shape)
    X = data_csv[['center', 'left', 'right']].values
    Y = data_csv['steering'].values
    # print(X)
    # print(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_ration, random_state=0)
    return X_train, X_test, Y_train, Y_test


# create data generator
def data_generator(data_path, batch_size, X_data, Y_data, train_flag):
    image_container = np.empty([batch_size, image_height, image_width, image_channels])
    steer_container = np.empty(batch_size)
    while True:
        i = 0
        for index in np.random.permutation(X_data.shape[0]):
            center, left, right = data_path + X_data[index]
            # print(X_data[index])
            steering_angle = Y_data[index]
            if train_flag and np.random.rand() < 0.4:
                image, steering_angle = image_preprocessing(center, left, right, steering_angle)
            else:
                image = cv2.imread(center)
            image_container[i] = image_normalized(image)
            steer_container[i] = steering_angle
            i += 1
            if i == batch_size:
                break
        yield image_container, steer_container


# train model
X_train, X_test, Y_train, Y_test = data_load(data_path)

model = build_model1()
# model = build_model2()
# model = build_model3()

# learning_rate=0.0001
model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=['accuracy'])
checkpoint = ModelCheckpoint('duyifan_lake_model1_{epoch:03d}.h5', monitor='val_loss', verbose=1, save_best_only=True,
                             mode='auto')
stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=300, verbose=1, mode='auto')
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=False)
model.fit(data_generator(data_path, batch_size, X_train, Y_train, True),
          # steps_per_epoch=len(X_train) // batch_size,
          steps_per_epoch=batch_num,
          epochs=epochs,
          verbose=1,
          validation_data=data_generator(data_path, batch_size, X_test, Y_test, False),
          # validation_steps=len(X_test) // batch_size,
          validation_steps=1,
          max_queue_size=1,
          callbacks=[checkpoint, stopping, tensorboard],
          )

# save model
model.save('duyifan_lake_model1.h5')
