import keras
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
import numpy as np
import sp_googlemaps


def build_model(input_shape=(64, 64, 3), dropout_ratio=0.3, convolution_layers=(64, 64, 64, 64), num_classes=2,
                loss_function=keras.losses.categorical_crossentropy,
                dense_layers=(256,), weights_file=None):
    model = Sequential()
    model.add(Conv2D(convolution_layers[0], kernel_size=(3, 3),
                     padding='same',
                     activation='relu',
                     input_shape=input_shape))
    if dropout_ratio > 0:
        model.add(Dropout(dropout_ratio))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    for kernel_count in convolution_layers[1:]:
        model.add(Conv2D(kernel_count, (3, 3), activation='relu', padding='same'))
        if dropout_ratio > 0:
            model.add(Dropout(dropout_ratio))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    for num_dense in dense_layers:
        model.add(Dense(num_dense, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # Have an existing weights file? Load before compiling!
    if weights_file:
        model.load_weights(weights_file)

    model.compile(loss=loss_function,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])


def model_predict(model, filenames_csv, folder, image_size):
    x_all, y_all = sp_googlemaps.load_data(filenames_csv, folder, image_size,
                      skip_headline=False,
                      featurewise_center=True,
                      featurewise_std_normalization=True)
    y_predict = model.predict(x_all)
    y_predict_labels = np.argmax(y_predict, 1).astype('bool')

    #for i in range(y_predict_labels.shape[0]):
    #    if y_all[i] and (not y_predict_labels[i]):
    #        shutil.copyfile(os.path.join(fs_path, filenames[i][0]), os.path.join('images/false_neg', filenames[i][0]))
    #    if (not y_all[i]) and y_predict_labels[i]:
    #        shutil.copyfile(os.path.join(fs_path, filenames[i][0]), os.path.join('images/false_pos', filenames[i][0]))
    return y_predict, y_predict_labels
