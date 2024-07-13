# import
import numpy as np
import pandas as pd
import os.path
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt
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


# def class_2_type(root):
#     label = ""
#     if "雙踝" in root:
#         label = "0"
#     elif "三踝" in root:
#         label = "1"
#     return label

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

def train_cross_validation(image_dir, n_splits=5, class_count=2, maru_part=None):

    ## load data and  labels
    # =========================
    labels = []
    filepaths = []
    data = load_path(image_dir, class_count)
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
    
    print("-------Training " + part + "-------")
    train_df, test_df = train_test_split(images, train_size=0.9, shuffle=True, random_state=1, stratify=images['Label'])
    ## test  image
    # =========================
    test_generator = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet50.preprocess_input)
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

    ## train model, k fold cross validation
    # ==================================================
    fold_no = 1
    test_acc = []
    test_f1 = []
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)
    for train_index, val_index in kfold.split(train_df, train_df['Label']):
        k_fold_train = train_df.iloc[train_index]
        k_fold_val = train_df.iloc[val_index]
        # print("K-fold training set label distribution:\n", k_fold_train['Label'].value_counts(normalize=True))
        # print("K-fold validation set label distribution:\n", k_fold_val['Label'].value_counts(normalize=True))
        
        ## train, validation image
        # =========================
        train_generator = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True,preprocessing_function=tf.keras.applications.resnet50.preprocess_input)
        val_generator = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet50.preprocess_input)
        train_images = train_generator.flow_from_dataframe(
            dataframe=k_fold_train,
            x_col='Filepath',
            y_col='Label',
            target_size=(224, 224),
            color_mode='rgb',
            class_mode='categorical',
            batch_size=64,
            shuffle=True,
            seed=42
        )
        val_images = val_generator.flow_from_dataframe(
            dataframe=k_fold_val,
            x_col='Filepath',
            y_col='Label',
            target_size=(224, 224),
            color_mode='rgb',
            class_mode='categorical',
            batch_size=64,
            shuffle=False,
            seed=42
        )
        # =========================
        
        ## load model
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
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

        ## early stop 
        # callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        
        ## no early stop
        model.fit(train_images, validation_data=val_images, epochs=25)

        results = model.evaluate(test_images,  verbose=0)
        # =========================

        ## print each results
        # =========================
        test_acc.append(np.round(results[1], 2))
        pred = model.predict(test_images)
        predicted_labels = np.argmax(pred, axis=1)
        f1 = f1_score(test_images.labels, predicted_labels, average='macro')
        test_f1.append(np.round(f1, 2))
        fold_no += 1
        # =========================
    # ==================================================
        
    ## print  mean results
    # =========================
    print(f"acc mean = {np.round(np.mean(test_acc), 2)}, std = {np.round(np.std(test_acc), 2)}")
    print(test_acc)
    print(f"f1  mean = {np.round(np.mean(test_f1), 2)}, std = {np.round(np.std(test_f1), 2)}")
    print(test_f1)
    # =========================


#####################################
# train_cross_validation: 訓練
#############
## image_dir: 影像來源
## class_count: 分幾個class
## n_splits: 幾折
## maru_part: 用maru哪個部位的模型pretrain
#############

# load_path: load image
#############
## path: 影像來源
## class_count: 分幾個class
## n_splits: 幾折
#############

# class_2_type, class_3_type: 種類標籤, 回傳label種類標籤, 回傳label

#####################################


## 訓練front or side
# =========================
path = "E://data_bone//all//front"
# =========================

train_cross_validation(image_dir=path, class_count=2)