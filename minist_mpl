#encoding:utf-8
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.optimizers import SGD,Adadelta
from keras.models import save_model
import matplotlib.pyplot as plt
(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train = x_train.reshape(60000,28*28).astype('float32')  #转换数据格式
x_test = x_test.reshape(10000,28*28).astype('float32')

x_train /= 255   #训练数据归一化
x_test /= 255

y_train = keras.utils.to_categorical(y_train,10)    #one-hot编码
y_test = keras.utils.to_categorical(y_test,10)

print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
modle = Sequential()
#第一层隐层，64个神经元
modle.add(Dense(256,activation='relu',input_dim=28*28))
#第二层隐层，64个神经元
modle.add(Dense(256,activation='relu'))
modle.add(Dropout(0.5))
#输出层，10个神经元
modle.add(Dense(10,activation='softmax'))

sgd = SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)
modle.compile(loss='categorical_crossentropy',optimizer='adagrad',metrics=['accuracy'])

modle.fit(x_train,y_train,epochs=10,batch_size=128)
score = modle.evaluate(x_test,y_test,batch_size=128)
print(score)
modle.save('MLP_minist.h5')
