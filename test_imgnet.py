# import
import numpy as np
import pandas as pd
import os.path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

# 載入腳踝骨折照片
import os
def load_path(path):
    dataset = []
    for root, dirs, files in os.walk(path):
        for file in files:
            # label
            label = ""
            if "正常組" in root:
                label = "nor"
            elif "三踝" in root:
                label = "tri"
            elif "雙踝" in root:
                label = "bi"
            # else:
            #     label = "fra"


            dataset.append(
                            {   
                                'uuid': root.split("\\")[-1],
                                'label': label,
                                'image_path': os.path.join(root, file)
                            }
                        )


    return dataset


# 載不同區的模型
def trainByPart(image_dir):
    data = load_path(image_dir)
    labels = []
    filepaths = []
    # front or side
    part = image_dir.split("//")[-1]

    # add labels for dataframe for each category 0-fractured, 1- normal
    for row in data:
        labels.append(row['label'])
        filepaths.append(row['image_path'])

    filepaths = pd.Series(filepaths, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')

    images = pd.concat([filepaths, labels], axis=1)

    train_df, test_df = train_test_split(images, train_size=0.9, shuffle=True, random_state=1)

    train_generator = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True,
                                                                      preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
                                                                      validation_split=0.2)

    # maru as transfer
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

    # pretrained_model = tf.keras.models.load_model(maru_part)
    pretrained_model = tf.keras.applications.resnet50.ResNet50(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg')

    # for faster performance
    pretrained_model.trainable = False

    inputs = pretrained_model.input
    x = tf.keras.layers.Dense(128, activation='relu', name='dense_128')(pretrained_model.output)
    x = tf.keras.layers.Dense(50, activation='relu', name='dense_50')(x)

    # outputs Dense '2' because of 2 classes, fratured and normal
    # outputs = tf.keras.layers.Dense(2, activation='softmax', name='output_layer')(x)
    outputs = tf.keras.layers.Dense(3, activation='softmax', name='output_layer')(x)
    model = tf.keras.Model(inputs, outputs)
    # print(model.summary())
    print("-------Training " + part + "-------")

    # Adam optimizer with low learning rate for better accuracy
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    # early stop when our model is over fit or vanishing gradient, with restore best values
    # callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(train_images, validation_data=val_images, epochs=25)

    # save model to this path
    model.save("./weights3/" +"imagenet_3part_"+ part+"_frac.h5")
    results = model.evaluate(test_images, verbose=0)
    print("imagenet_3part_"+ part + " Results:")
    print(results)
    print(f"Test Accuracy: {np.round(results[1] * 100, 2)}%")
    # test_acc.append("Test Accuracy of " + part2 +"_"+ part + f": {np.round(results[1] * 100, 2)}%")

    # create plots for accuracy and save it
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title("imagenet_3part_"+ part + ' model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    # plt.show()
    figAcc = plt.gcf()
    my_file = os.path.join("./plots3/" + "imagenet_3part_"+ part + "_Accuracy.jpeg")
    figAcc.savefig(my_file)
    plt.clf()

    # create plots for loss and save it
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title("imagenet_3part_"+ part + ' model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    # plt.show()
    figAcc = plt.gcf()
    my_file = os.path.join("./plots3/" + "imagenet_3part_"+ part + "_Loss.jpeg")
    figAcc.savefig(my_file)
    plt.clf()

# path = "E://data_bone//雲端//雲端_clean2//side"
path = "E://data_bone//雲端//雲端_clean2//front"

trainByPart(path)