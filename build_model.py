"""
太原理工大学信息与光电工程学院《多媒体通信技术》课程
项目三 基于深度学习的Udacity无人驾驶系统

 学号：2020002030
 班级：信息2002
 姓名：杜艺帆
 完成工作：构建神经网络模型；
"""

# import third libs
from keras.models import Sequential
from keras.layers import Conv2D, Dropout, Flatten, Dense, Lambda, MaxPool2D
from data_preprocessing import image_height, image_width, image_channels

Input_size = (image_height, image_width, image_channels)
from keras.utils import plot_model


# build CNN model
def build_model1():  # 参数数量 K*K*C*O+O（K:kernel_size,C:输入通道数，O:输出通道数）
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1, input_shape=Input_size))
    model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2),
                     activation='relu'))
    model.add(MaxPool2D())
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), strides=(2, 2),
                     activation='relu'))
    model.add(MaxPool2D())
    model.add(Flatten())
    model.add(Dense(32))
    model.add(Dropout(0.20))
    model.add(Dense(16))
    model.add(Dense(1))
    model.summary()
    return model

def build_model2():
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1, input_shape=Input_size))
    model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2),
                     activation='elu'))
    model.add(Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2),
                     activation='elu'))
    model.add(Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2),
                     activation='elu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
                     activation='elu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
                     activation='elu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()
    return model

def build_model3():
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1, input_shape=Input_size))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),
                     padding='same', activation='elu'))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),
                     padding='same', activation='elu'))
    model.add(MaxPool2D((2, 2), padding='same'))
    model.add(Dropout(0.5))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
                     padding='same', activation='elu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
                     padding='same', activation='elu'))
    model.add(MaxPool2D((2, 2), padding='same'))
    model.add(Dropout(0.5))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),
                     padding='same', activation='elu'))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),
                     padding='same', activation='elu'))
    model.add(MaxPool2D((2, 2), padding='same'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512, activation='elu'))
    model.add(Dense(64, activation='elu'))
    model.add(Dense(16, activation='elu'))
    model.add(Dense(1))
    model.summary()
    return model

# define main function
if __name__ == '__main__':
    model1 = build_model1()
    plot_model(model1, to_file='model1.png', show_shapes=True)
    model2 = build_model2()
    plot_model(model2, to_file='model2.png', show_shapes=True)
    model3 = build_model3()
    plot_model(model3, to_file='model3.png', show_shapes=True)
