import keras
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
import numpy as np
import sp_googlemaps
import os
import shutil
from PIL import Image
from PIL import ImageFont, ImageDraw


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
    return model


def model_predict(model, filenames_csv, predict_folder, fullsize_folder, output_folder, image_size, label_map, exclude_index=4):
    x_all, y_all = sp_googlemaps.load_data(filenames_csv, predict_folder, image_size,
                                           label_map,
                                           exclude_index,
                                           skip_headline=False,
                                           featurewise_center=True,
                                           featurewise_std_normalization=True)
    y_predict = model.predict(x_all)
    y_predict_labels = np.argmax(y_predict, 1).astype('bool')

    filenames = sp_googlemaps.load_filenames(filenames_csv, 0, exclude_index)

    false_neg_folder = os.path.join(output_folder, 'false_neg')
    sp_googlemaps.create_folder_if_not_exists(false_neg_folder)
    false_pos_folder = os.path.join(output_folder, 'false_pos')
    sp_googlemaps.create_folder_if_not_exists(false_pos_folder)

    font = ImageFont.truetype('arialbd', 16)

    for i in range(y_predict_labels.shape[0]):
        target_path = None
        if y_all[i] and (not y_predict_labels[i]):
            target_path = os.path.join(false_neg_folder, filenames[i][0])
            shutil.copyfile(os.path.join(fullsize_folder, filenames[i][0]),
                            target_path)
        if (not y_all[i]) and y_predict_labels[i]:
            target_path = os.path.join(false_pos_folder, filenames[i][0])
            shutil.copyfile(os.path.join(fullsize_folder, filenames[i][0]),
                            target_path)
        if target_path:
            img = Image.open(target_path)
            draw = ImageDraw.Draw(img)
            draw.text((5, 5), "Neg={:.2f} : Pos={:.2f}".format(y_predict[i][0], y_predict[i][1]),
                      (255, 0, 255), font=font)
            img.save(target_path)

    return y_predict, y_predict_labels
