import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, LearningRateScheduler
from data_preprocessing import image_height, image_width, image_channels, image_preprocessing, image_normalized
from build_model import build_model1, build_model2, build_model3


class DeepLearningModelTrainer:
    def __init__(self, data_path, test_ratio=0.1, batch_size=32, epochs=50, initial_learning_rate=0.0001):
        self.data_path = data_path
        self.test_ratio = test_ratio
        self.batch_size = batch_size
        self.epochs = epochs
        self.initial_learning_rate = initial_learning_rate

    def load_data(self):
        data_csv = pd.read_csv(self.data_path + 'driving_log.csv',
                               names=['center', 'left', 'right', 'steering', '_', '__', '___'])
        X = data_csv[['center', 'left', 'right']].values
        Y = data_csv['steering'].values
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=self.test_ratio, random_state=0)
        return X_train, X_test, Y_train, Y_test

    def data_generator(self, X_data, Y_data, train_flag):
        image_container = np.empty([self.batch_size, image_height, image_width, image_channels])
        steer_container = np.empty(self.batch_size)
        while True:
            i = 0
            for index in np.random.permutation(X_data.shape[0]):
                center, left, right = self.data_path + X_data[index]
                # print(X_data[index])
                steering_angle = Y_data[index]
                if train_flag and np.random.rand() < 0.4:
                    image, steering_angle = image_preprocessing(center, left, right, steering_angle)
                else:
                    image = cv2.imread(center)
                image_container[i] = image_normalized(image)
                steer_container[i] = steering_angle
                i += 1
                if i == self.batch_size:
                    break
            yield image_container, steer_container

    def lr_schedule(self, epoch):
        if epoch < 30:
            return self.initial_learning_rate
        elif epoch < 60:
            return self.initial_learning_rate * 0.1
        else:
            return self.initial_learning_rate * 0.01

    def train_model(self):
        X_train, X_test, Y_train, Y_test = self.load_data()
        model = build_model3()  # Define your advanced model here
        model.compile(optimizer=Adam(learning_rate=self.initial_learning_rate), loss='mse', metrics=['accuracy'])
        checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
        stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=300, verbose=1, mode='auto')
        tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=False)
        lr_scheduler = LearningRateScheduler(self.lr_schedule)
        model.fit(self.data_generator(X_train, Y_train, True), steps_per_epoch=len(X_train) // self.batch_size,
                  epochs=self.epochs, verbose=1,
                  validation_data=self.data_generator(X_test, Y_test, False),
                  validation_steps=len(X_test) // self.batch_size,
                  callbacks=[checkpoint, stopping, tensorboard, lr_scheduler])
        model.save('final_model.h5')


# Example usage
trainer = DeepLearningModelTrainer(data_path='./lake_data/', test_ratio=0.1, batch_size=32, epochs=75,
                                   initial_learning_rate=0.001)
trainer.train_model()
