import keras
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
import sp_googlemaps


batch_size = 64 # 128
num_classes = 2
epochs = 10

# input image dimensions
image_pixels = 50
img_x, img_y = image_pixels, image_pixels

# load the MNIST data set, which already splits into train and test sets for us
x_all, y_all = sp_googlemaps.load_data('2018-02-20_16-42 filenames.csv', 'images/thumbs', 50)

# use 1 picture as validation and test
x_validation = x_all[0:1][:][:][:]
y_validation = y_all[0:1]
x_test = x_all[1:2][:][:][:]
y_test = y_all[1:2]

x_train = x_all[2:][:][:][:]
y_train = y_all[2:]

input_shape = (img_x, img_y, 3)

# convert the data to the right type
print('x_all shape:', x_all.shape)
print('y_all shape:', y_all.shape)
print(x_train.shape[0], 'train samples')
print(x_validation.shape[0], 'validation samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices - this is for use in the
# categorical_crossentropy loss below

y_train = keras.utils.to_categorical(y_train, num_classes)
y_validation = keras.utils.to_categorical(y_validation, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#print(y_train)

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])


model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_validation, y_validation))
score = model.evaluate(x_validation, y_validation, verbose=0)
print('Validation loss:', score[0])
print('Validation accuracy:', score[1])
# plt.plot(range(1, 11), history.acc)
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.show()
