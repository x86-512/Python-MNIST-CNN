import tensorflow
import numpy as num
import matplotlib.pyplot as plt


class Conv_NN():
    def __init__(self):
        self.model = tensorflow.keras.models.Sequential()
        mnist = tensorflow.keras.datasets.mnist
        (self.train_img, self.train_lbl), (self.test_img, self.test_lbl) = mnist.load_data()

    def setup(self):
        self.model.add(tensorflow.keras.layers.Conv2D(32,(3,3), activation="relu", input_shape = (28,28,1)))
        self.model.add(tensorflow.keras.layers.MaxPooling2D(pool_size=(2,2)))
        self.model.add(tensorflow.keras.layers.Conv2D(64,(3,3), activation="relu"))
        self.model.add(tensorflow.keras.layers.MaxPooling2D(pool_size=(2,2)))
        self.model.add(tensorflow.keras.layers.Conv2D(64,(3,3), activation="relu"))
        self.model.add(tensorflow.keras.layers.Flatten())
        self.model.add(tensorflow.keras.layers.Dense(64,activation="relu"))
        self.model.add(tensorflow.keras.layers.Dense(10,activation="softmax"))


    def compile(self):
        loss = tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.model.compile(optimizer=tensorflow.keras.optimizers.Adam(), loss=loss, metrics=["accuracy"])

    def train(self):
        self.model.fit(self.train_img, self.train_lbl, epochs=5, batch_size=128)
        self.model.evaluate(self.test_img, self.test_lbl, batch_size=128, verbose=2)

def main():
    nn = Conv_NN()
    nn.setup()
    nn.compile()
    nn.train()

if __name__=="__main__":
    main()
