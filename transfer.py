# import
import numpy as np
import pandas as pd
import os.path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score

import os

# label
# =========================
def class_2_type(root):
    label = ""
    if "正常" in root:
        label = "0"
    else:
        label = "1"
    return label

def class_3_type(root):
    label = ""
    if "正常" in root:
        label = "0"
    elif "雙踝" in root:
        label = "1"
    elif "三踝" in root:
        label = "2"
    return label
# =========================


def load_path(path, class_count):
    dataset = []
    class_type = ''
    if class_count == 2:
        class_type = class_2_type
    elif class_count == 3:
        class_type = class_3_type       
    for root, dirs, files in os.walk(path):
        for file in files:
            label = class_type(root)
            dataset.append(
                            {   
                                'uuid': root.split("\\")[-1],
                                'label': label,
                                'image_path': os.path.join(root, file)
                            }
                        )

    return dataset


def trainByPart(image_dir,class_count=2, maru_part=None, save_path="./weights/"):

    ## load data and  labels
    # =========================
    data = load_path(image_dir, class_count)
    labels = []
    filepaths = []
    for row in data:
        labels.append(row['label'])
        filepaths.append(row['image_path'])

    filepaths = pd.Series(filepaths, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')

    images = pd.concat([filepaths, labels], axis=1)
    # =========================


    ## front(AP+Mortise) or side(Lateral) 標籤
    # =========================
    part = ""
    if image_dir.split("//")[-1] == "front":
        part = "AP+Mortise"
    else:
        part = "Lateral"
    # =========================


    # 哪個部位的權重
    # =========================
    part2="imagenet"
    if maru_part is not None:
        part2 = maru_part.split("_")[2]
    # =========================

    ## split image
    # =========================
    train_df, test_df = train_test_split(images, train_size=0.9, shuffle=True, random_state=1, stratify=images['Label'])
    print("Training set label distribution:\n", train_df['Label'].value_counts(normalize=True))
    print("Test set label distribution:\n", test_df['Label'].value_counts(normalize=True))
    
    train_generator = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True,
                                                                      preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
                                                                      validation_split=0.2)
    test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.resnet50.preprocess_input)

    train_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=64,
        shuffle=True,
        seed=42,
        subset='training'
    )

    val_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=64,
        shuffle=True,
        seed=42,
        subset='validation'
    )

    test_images = test_generator.flow_from_dataframe(
        dataframe=test_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=False
    )
    # =========================


    # load model
    # =========================
    if maru_part is not None:
        pretrained_model = tf.keras.models.load_model(maru_part)
    else:
        pretrained_model = tf.keras.applications.resnet50.ResNet50(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet',
            pooling='avg')

    pretrained_model.trainable = False

    inputs = pretrained_model.input
    x = tf.keras.layers.Dense(128, activation='relu', name='dense_128')(pretrained_model.output)
    x = tf.keras.layers.Dense(50, activation='relu', name='dense_50')(x)

    outputs = tf.keras.layers.Dense(class_count, activation='softmax', name='output_layer')(x)
    model = tf.keras.Model(inputs, outputs)
    # print(model.summary())
    # =========================


    ## compile and evaluate
    # =========================
    print("-------Training " + part + part2 + "-------")
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    ## early stop 
    # callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    ## no early stop
    history = model.fit(train_images, validation_data=val_images, epochs=25)

    results = model.evaluate(test_images, verbose=0)
    # =========================


    ## save model to this path
    # =========================
    model.save(save_path + part2 + "_" + part + "_" + str(class_count) + "class" + "_frac.h5")
    # =========================


    ## print results
    # =========================
    print(part + "_" + part2 + "_Results:")
    pred = model.predict(test_images)
    predicted_labels = np.argmax(pred, axis=1)
    f1 = f1_score(test_images.labels, predicted_labels, average='macro')
    print(results)
    print(f"Test Accuracy: {np.round(results[1], 2)}")
    print(f"f1 score: {np.round(f1, 2)}")
    # =========================


    # create plots for accuracy and save it
    # =========================
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(part + " " + part2 +' model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    figAcc = plt.gcf()
    my_file = os.path.join("./plots/" + part + "_" + part2 + "_" + str(class_count) + "class_Accuracy.jpeg")
    figAcc.savefig(my_file)
    plt.clf()
    # =========================


    ## create plots for loss and save it
    # =========================
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(part2 +"_"+ part + ' model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    figAcc = plt.gcf()
    my_file = os.path.join("./plots/" + part + "_" + part2 + "_" + str(class_count) + "class_Loss.jpeg")
    figAcc.savefig(my_file)
    plt.clf()
    # =========================


    ## plot confusion matrix
    # =========================
    if class_count == 2:
        display_labels = [0, 1]
    elif class_count == 3:
        display_labels = [0, 1, 2]

    cm = confusion_matrix(test_images.labels, predicted_labels)
    cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = display_labels)
    cm_display.plot()
    plt.title(part2 +"_"+ part + ' Confusion Matrix')
    figAcc = plt.gcf()
    my_file = os.path.join("./plots/" + part + "_" + part2 + "_" + str(class_count) + "class_Confusion Matrix.jpeg")
    figAcc.savefig(my_file)
    plt.clf()
    # =========================



###################################
## 單次訓練
## save_path: 儲存模型的目錄
###################################


## 訓練front or side
# =========================
path = "E://data_bone//all//side"
save_path = "./weights/中榮/side/transfer_imagenet/all/"
# =========================

## 所有部位
# =========================
# maru_part_arr = ["XR_ELBOW", "XR_FINGER", "XR_FOREARM", "XR_HAND", "XR_HUMERUS", "XR_SHOULDER", "XR_WRIST"]
# for a in maru_part_arr:
#     maru_part = f"D://reaserch//Bone-Fracture-Detection//weights2//ResNet50_{a}_frac.h5"
#     trainByPart(path, maru_part)
# =========================

## imagenet
# =========================
trainByPart(image_dir=path, class_count=2, save_path=save_path)
# =========================
