import keras
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
import sp_googlemaps

# for development, reload the package every time
#import importlib
#importlib.reload(sp_googlemaps)

batch_size = 64 # 128
num_classes = 2
epochs = 30

# input image dimensions
image_pixels = 50
img_x, img_y = image_pixels, image_pixels

x_all, y_all = sp_googlemaps.load_data('test_cnn_m_l_j.csv', 'images/thumbs', image_pixels,
                                       skip_headline=False)

# use n pictures as validation and test
test_samples = int(0)
validation_samples = int(0)
x_validation = x_all[0:validation_samples][:][:][:]
y_validation = y_all[0:validation_samples]
x_test = x_all[validation_samples:(validation_samples+test_samples)][:][:][:]
y_test = y_all[validation_samples:(validation_samples+test_samples)]

x_train = x_all[(validation_samples+test_samples):][:][:][:]
y_train = y_all[(validation_samples+test_samples):]

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
model.add(Conv2D(4, kernel_size=(10, 10), strides=(3, 3),
                 padding='same',
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Conv2D(8, (5, 5), activation='relu'))
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
          validation_split=0.1,
          #validation_data=(x_validation, y_validation)
          )
#score = model.evaluate(x_validation, y_validation, verbose=0)
#print('Validation loss:', score[0])
#print('Validation accuracy:', score[1])
# plt.plot(range(1, 11), history.acc)
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.show()
