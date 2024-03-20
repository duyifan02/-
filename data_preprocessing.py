"""
太原理工大学信息与光电工程学院《多媒体通信技术》课程
项目三 基于深度学习的Udacity无人驾驶系统

 学号：2020002030
 班级：信息2002
 姓名：杜艺帆
 完成工作：0、扩充、丰富、平衡数据；1、归一化数据；2、模拟一些场景；
"""

# import third libs
import cv2
import numpy as np

# initialize variable
image_height, image_width, image_channels = 66, 200, 3
center, left, right = './test/center.jpg', './test/left.jpg', './test/right.jpg'
steering_angle = 0.0


# select data
def image_choose(center, left, right, steering_angle):
    choice = np.random.choice(3)
    if choice == 0:
        image_name = center
        bias = 0.0
    elif choice == 1:
        image_name = left
        bias = 0.2
    else:
        image_name = right
        bias = -0.2
    image = cv2.imread(image_name)
    steering_angle += bias
    # cv2.imshow('image choose', image)
    # cv2.waitKey(0)
    return image, steering_angle


# flip data
def image_flip(image, steering_angle):
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    # cv2.imshow('image_flip', image)
    # cv2.waitKey(0)
    return image, steering_angle


# translate data
def image_translate(image, steering_angle):
    range_x, range_y = 100, 10
    tran_x = int(range_x * (np.random.rand() - 0.5))
    tran_y = int(range_y * (np.random.rand() - 0.5))
    steering_angle += tran_x * 0.002
    tran_m = np.float32([[1, 0, tran_x], [0, 1, tran_y]])
    image = cv2.warpAffine(image, tran_m, (image.shape[1], image.shape[0]))
    # cv2.imshow('image_translate', image)
    # cv2.waitKey(0)
    return image, steering_angle


# normalize data
def image_normalized(image):
    image = image[60:-25, :, :]
    image = cv2.resize(image, (image_width, image_height), cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    # cv2.imshow('image_normalized', image)
    # cv2.waitKey(0)
    return image


# define pre-processing function
def image_preprocessing(center, left, right, steering_angle):
    image, steering_angle = image_choose(center, left, right, steering_angle)
    image, steering_angle = image_flip(image, steering_angle)
    image, steering_angle = image_translate(image, steering_angle)
    # image = image_normalized(image)
    return image, steering_angle


# define main function
if __name__ == '__main__':
    image, steering_angle = image_preprocessing(center, left, right, steering_angle)
    image = image_normalized(image)
    print('steering_angle = ', steering_angle)
    cv2.imshow('image_normalized', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()