"""
太原理工大学信息与光电工程学院《多媒体通信技术》课程
项目三 基于深度学习的Udacity无人驾驶系统

 学号：2020002030 
 班级：信息2002
 姓名：杜艺帆
 完成工作：0、采集数据；1、连接模拟器；2、驾驶车辆；3、验证模型；
"""

# import third libs
# (1) websocket
import socketio
from flask import Flask
import eventlet.wsgi
# (2) image processing
import base64
import cv2
from io import BytesIO
from PIL import Image
import numpy as np
from keras.models import load_model
from data_preprocessing import image_normalized

model = load_model('duyifan_lake_model1.h5')
# model = load_model('best_model.h5')
# initialize variable
steering_angle = -0.02
throttle = 0.2


# define controller
class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement

        # integral error
        self.integral += self.error
        # print(self.integral)
        return self.Kp * self.error + self.Ki * self.integral


controller = SimplePIController(0.1, 0.002)
set_speed = 9
controller.set_desired(set_speed)


# send control
def send_control(steering_angle, throttle):
    sio.emit(
        'steer',
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        }
    )


# create net connection
sio = socketio.Server()
app = Flask(__name__)
app = socketio.WSGIApp(sio, app)


# drive car
@sio.on('connect')
def on_connect(sid, environ):
    # print('successfully connected ','sid= ', sid,'environ= ', environ)
    print('successfully connected', sid)


@sio.on('telemetry')
def on_telemetry(sid, data):
    # print('message = ', data)
    speed = float(data['speed'])
    print(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    # display by opencv
    cv2.imshow('Pic from Udacity', cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
    # press q to quit
    cv2.waitKey(1)
    throttle = controller.update(speed)
    image = image_normalized(np.asarray(image))
    # image=image_normalized(image)
    steering_angle = float(model.predict(np.array([image])))

    send_control(steering_angle, throttle)


@sio.on('disconnect')
def on_disconnect(sid):
    print('disconnect ', sid)


# start service
eventlet.wsgi.server(eventlet.listen(('', 4567)), app)



@sio.on('connect')  # 当收到 'connect' 事件时执行下面的函数
def on_connect(sid, environ):  # 函数定义，接收 sid 和 environ 两个参数
    print('successfully connected', sid)  # 打印成功连接信息


@sio.on('telemetry')  # 当收到 'telemetry' 事件时执行下面的函数
def on_telemetry(sid, data):  # 函数定义，接收 sid 和 data 两个参数
    speed = float(data['speed'])  # 从 data 中获取 'speed' 字段的值并转换为浮点数
    print(data['speed'])  # 打印 'speed' 字段的值
    image = Image.open(BytesIO(base64.b64decode(data['image'])))  # 解码图像数据并转换为图像对象
    cv2.imshow('Pic from Udacity', cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))  # 在窗口中显示图像
    cv2.waitKey(1)  # 等待按键按下的最长时间为1毫秒
    throttle = controller.update(speed)  # 更新油门信息
    image = image_normalized(np.asarray(image))  # 对图像进行归一化处理
    steering_angle = float(model.predict(np.array([image])))  # 使用模型预测方向盘转角
    send_control(steering_angle, throttle)  # 发送方向盘转角和油门信息


@sio.on('disconnect')  # 当收到 'disconnect' 事件时执行下面的函数
def on_disconnect(sid):  # 函数定义，接收 sid 参数
    print('disconnect', sid)  # 打印断开连接信息
