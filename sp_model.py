import keras
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
import numpy as np
import sp_googlemaps
import os
import shutil
import PIL
from PIL import Image
from PIL import ImageFont, ImageDraw
from vis.visualization import visualize_saliency, overlay
import imageio


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


def precision_recall(y_pred, y_label):
    pos = np.sum(y_label).astype(int)
    neg = y_label.shape[0] - pos
    tp = sum([1 for pred, true in zip(y_pred, y_label) if pred == true == 1])
    fp = sum([1 for pred, true in zip(y_pred, y_label) if pred != true and true == 0])
    tn = sum([1 for pred, true in zip(y_pred, y_label) if pred == true == 0])
    fn = sum([1 for pred, true in zip(y_pred, y_label) if pred != true and true == 1])

    precision = float(tp)/(tp+fp)
    recall = float(tp)/(tp+fn)
    f1_score = 2/(1/recall + 1/precision)

    print('''True positive: {}
True negative: {}
False positive: {}
False negative: {}
Precision: {}
Recall: {}
F1-Score: {}'''.format(tp, tn, fp, fn, precision, recall, f1_score))


def model_predict(model, filenames_csv, predict_folder, fullsize_folder, output_folder, image_size, label_map,
                  exclude_index=4, positive_bias=None):
    x_all, y_all = sp_googlemaps.load_data(filenames_csv, predict_folder, image_size,
                                           label_map,
                                           exclude_index,
                                           skip_headline=False,
                                           featurewise_center=True,
                                           featurewise_std_normalization=True)
    y_predict = model.predict(x_all)
    if positive_bias:
        y_predict[:, 0] -= positive_bias
        y_predict[:, 1] += positive_bias
    y_predict_labels = np.argmax(y_predict, 1).astype('bool')

    filenames = sp_googlemaps.load_filenames(filenames_csv, 0, exclude_index)

    sp_googlemaps.create_folder_if_not_exists(output_folder)
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

    precision_recall(y_predict_labels, y_all)
    return y_predict, y_predict_labels, y_all


def model_saliency(model, filenames_csv, predict_folder, fullsize_folder, output_folder, image_size, full_image_size,
                   layer_idx=-1, filter_indices=0, exclude_index=4):
    test_img, _ = sp_googlemaps.load_data(filenames_csv, predict_folder, image_size, bool, exclude_index,
                                          featurewise_center=True,
                                          featurewise_std_normalization=True)

    vis_img, _ = sp_googlemaps.load_data(filenames_csv, fullsize_folder, full_image_size, bool, exclude_index,
                                         featurewise_center=False,
                                         featurewise_std_normalization=False)

    vis_img *= 255.

    filenames = sp_googlemaps.load_filenames(filenames_csv, 0, exclude_index)
    sp_googlemaps.create_folder_if_not_exists(output_folder)
    for i, f in enumerate(filenames):
        filename = f[0]
        sal = visualize_saliency(model, layer_idx=layer_idx, filter_indices=filter_indices, seed_input=test_img[i, :, :, :])
        result_filename = os.path.join(output_folder, 'sal_'+filename)
        overlay_result_filename = os.path.join(output_folder, 'sal_ovl_'+filename)
        imageio.imwrite(result_filename, sal)
        if full_image_size != image_size:
            sal = Image.open(result_filename)
            sal = sal.resize((full_image_size, full_image_size), resample=PIL.Image.LANCZOS)
            sal = np.array(sal)[:, :, :3]
        imageio.imwrite(overlay_result_filename, overlay(sal, vis_img[i, :, :, :], 0.5))
