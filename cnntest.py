# Use this to temporarily disable GPU. Important: set BEFORE importing keras/tf!
#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import keras
from keras import layers
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import sp_googlemaps
import datetime
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt


# for development, reload the package every time
#import importlib
#importlib.reload(sp_googlemaps)

batch_size = 112
num_classes = 2
epochs = 60
loss_function = keras.losses.categorical_crossentropy

# input image dimensions
image_pixels = 64
img_x, img_y = image_pixels, image_pixels


def label_map(image_label):
    if image_label == 1 or image_label == 3:
        return True
    else:
        return False


x_all, y_all = sp_googlemaps.load_data('doc/labels/label_final.csv', r'images/dropbox/thumbs',
                                       image_pixels,
                                       label_map,
                                       4,  # The "uncertain"/"exclude" label
                                       equalize_labels=True,
                                       skip_headline=False,
                                       horizontal_flip=False,
                                       vertical_flip=False,
                                       YCbCr=False, #'BT601'/'JPEG'
                                       featurewise_center=False,
                                       featurewise_std_normalization=False)

# use a percentage of pictures as validation and test
test_ratio = 0
validation_ratio = 0.2

x_test = None
y_test = None
test_samples = 0
if test_ratio > 0:
    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=test_ratio)
    test_samples = y_test.shape[0]
else:
    x_train = x_all
    y_train = y_all

x_validation = None
y_validation = None
validation_samples = 0
if validation_ratio > 0:
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=validation_ratio)
    validation_samples = y_validation.shape[0]

input_shape = (img_x, img_y, 3)

# convert the data to the right type
print('x_all shape:', x_all.shape)
print('y_all shape:', y_all.shape)
print(x_train.shape[0], 'train samples')
print(validation_samples, 'validation samples')
print(test_samples, 'test samples')

# convert class vectors to binary class matrices - this is for use in the
# categorical_crossentropy loss below

y_train = keras.utils.to_categorical(y_train, num_classes)
if y_validation is not None:
    y_validation = keras.utils.to_categorical(y_validation, num_classes)
if y_test is not None:
    y_test = keras.utils.to_categorical(y_test, num_classes)

#print(y_train)

# Use dropout to reduce overfitting
# http://www.jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf
dropout_ratio = 0.3

model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3),
                 padding='same',
                 activation='relu',
                 input_shape=input_shape))
model.add(Dropout(dropout_ratio))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Dropout(dropout_ratio))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Dropout(dropout_ratio))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Dropout(dropout_ratio))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Have an existing weights file? Load before compiling!
#model.load_weights('2018-03-13_16-32 cnntest_weights.h5')

# This number does not change any calculation, just the labels in the plots
resume_from_epoch = 0

model.compile(loss=loss_function,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])


# Creates scheme of the model
# plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False)


# Use the data generator:
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    fill_mode='reflect',
    horizontal_flip=True,
)

# Use fit to get proper featurewise_center and featurewise_std_normalization
datagen.fit(x_train)
# Generate and save n*32 sample images
flow_countdown = 0
if flow_countdown:
    for x_gen, y_gen in datagen.flow(x_train, y_train, save_to_dir='images/kerasgenerated', batch_size=32):
        flow_countdown -= 1
        if not flow_countdown:
            break

datagen_valid = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    #rotation_range=10,
    #width_shift_range=0.1,
    #height_shift_range=0.1,
    #horizontal_flip=True,
)
datagen_valid.fit(x_validation)

now = datetime.datetime.now()
model_filename = '{:0>4}-{:0>2}-{:0>2}_{:0>2}-{:0>2} cnntest' \
    .format(now.year, now.month, now.day, now.hour, now.minute)

# define the checkpoint
checkpoint_filepath = model_filename + "_weights{epoch:03d}-{loss:.4f}.h5"
checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

fit_history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                  steps_per_epoch=x_train.shape[0] // batch_size, epochs=epochs+resume_from_epoch,
                                  validation_data=datagen_valid.flow(x_validation, y_validation, batch_size=batch_size),
                                  validation_steps=x_validation.shape[0]//batch_size,
                                  verbose=2,
                                  initial_epoch=resume_from_epoch,
                                  callbacks=callbacks_list)

# Without generator:
#model.fit(x_train, y_train,
#          batch_size=batch_size,
#          epochs=epochs,
#          verbose=1,
#          validation_split=0.2,
#          )
# plt.plot(range(1, 11), history.acc)
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.show()

model.save(model_filename + '_weights.h5')

with open(model_filename + '_layers.txt', 'w') as layer_file:
    layer_file.write(datagen.__class__.__name__ + '\n' + str(datagen.__dict__) + '\n\n\n')
    for layer in model.layers:
        config = layer.get_config()
        layer_file.write(
            ' '.join([
                layer.__class__.__name__,
                'in:' + str(layer.input_shape),
                'out:' + str(layer.output_shape)])
            + '\n'
            + str(config) + '\n')
plot_x_range = range(resume_from_epoch + 1, resume_from_epoch + epochs + 1)

plt.plot(plot_x_range, fit_history.history['acc'], label='Training')
plt.plot(plot_x_range, fit_history.history['val_acc'], label='Validation')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(model_filename + '_acc.png')
plt.clf()

plt.plot(plot_x_range, fit_history.history['loss'], label='Training')
plt.plot(plot_x_range, fit_history.history['val_loss'], label='Validation')
plt.xlabel('Epochs')
plt.ylabel('Loss ({})'.format(loss_function.__name__))
plt.legend()
plt.savefig(model_filename + '_loss.png')
plt.clf()

# ToDo
# - ImageDataGenerator: https://keras.io/preprocessing/image/
# - Zuerst Featurewise normalisieren, dann Samplewise
# - Dropout erhöhen (z.B. bis 0,4)
# - MEHR DATEN
# - RMSProp statt Adam
# - Wichtig! Balancing der Klassen
# - Californiadaten zum Balancen? Eher nicht.
