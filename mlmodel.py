from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import sequential
model=Sequential()
model.add(Convolution2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=(64,64,3))
train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
training_set = train_datagen.flow_from_directory('~/train/',target_size=(64, 64),batch_size=32,class_mode='binary')
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory('~/test/',target_size=(64,64),batch_size=32,class_mode=binary)
