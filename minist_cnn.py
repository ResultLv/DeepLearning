#encoding:utf-8
import keras
from keras.datasets import mnist
from keras.models import Sequential,save_model
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from keras.optimizers import SGD,Adadelta

(x_train,y_train),(x_test,y_test) = mnist.load_data()   #加载数据
print(x_train.shape,x_test.shape)
x_train = x_train.reshape(60000,28,28,1).astype('float32')  #二维数据
x_test = x_test.reshape(10000,28,28,1).astype('float32')

x_train /= 255  #训练数据归一化
x_test /= 255

y_train = keras.utils.to_categorical(y_train)    #one-hot编码
y_test = keras.utils.to_categorical(y_test)
num_classes = y_test.shape[1]

model = Sequential()    #创建序列模型
model.add(Conv2D(64,(3,3),activation='relu',input_shape=(28,28,1)))  #第一层卷积层
model.add(MaxPooling2D(pool_size=(2,2)))    #池化层

model.add(Conv2D(64,(3,3),activation='relu'))  #第二层卷积层
model.add(MaxPooling2D(pool_size=(2,2)))    #池化层

model.add(Flatten())    #铺平当前节点

model.add(Dense(128,activation='relu'))     #全连接层
model.add(Dropout(0.5)) #随机失活
model.add(Dense(num_classes,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])   #编译模型
model.fit(x_train,y_train,batch_size=128,epochs=10)     #训练模型
score = model.evaluate(x_test,y_test,batch_size=128)    #评价模型

print(score)    #打印分类准确率

model.save('CNN_minist.h5')
