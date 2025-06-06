import os
import os.path

import tensorflow as tf
import pandas as pd
import numpy as np

from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt

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


def load_path(path, class_count):
    dataset = []
    class_type = ''
    if class_count == 2:
        class_type = class_2_type
    elif class_count == 3:
        class_type = class_3_type   

    for root, dirs, files in os.walk(path):
        for file in files:
            if file.lower().endswith('.jpg'):
                label = class_type(root)
                # if label != "":
                dataset.append(
                                {   
                                    'uuid': root.split("\\")[-1],
                                    'label': label,
                                    'image_path': os.path.join(root, file)
                                }
                            )

    return dataset




def train_original_cross(image_dir, n_splits=5, class_count=2, maru_part=None,  chosen_model='resnet50'):


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


    ## define common params, same as transfer
    # =========================
    comon_params={
            ## fully CV layer
            "include_top": True,
            ## no pretrain 
            "weights": None,
            "input_shape": (224, 224, 3),
            "pooling": 'avg',
            "classes": class_count,
            "classifier_activation": 'softmax'
    }
    # =========================
    


    ## load chosen model
    # =========================
    if chosen_model == 'resnet50':
        model = tf.keras.applications.ResNet50(**comon_params)
    # elif chosen_model == 'resnet152':
    #   model = tf.keras.applications.ResNet152(**comon_params)
    elif chosen_model == 'vgg16':
        model = tf.keras.applications.VGG16(**comon_params)
    elif chosen_model == 'vgg19':
        model = tf.keras.applications.VGG19(**comon_params)
    elif chosen_model == 'mobilenet':
        model = tf.keras.applications.VGG19(**comon_params)
    elif chosen_model == 'mobilenet_v2':
        model = tf.keras.applications.VGG19(**comon_params)
    elif chosen_model == 'efficientnet':
        model = tf.keras.applications.VGG19(**comon_params)
    elif chosen_model == 'inception_v3':
        model = tf.keras.applications.VGG19(**comon_params)
    else:
        raise ValueError("Model name not recognized. Please choose a valid model.")
    # =========================

    train_df, test_df = train_test_split(images, train_size=0.8, shuffle=True, random_state=1, stratify=images['Label'])

    ## test  image
    # =========================
    test_generator = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=False)
    test_images = test_generator.flow_from_dataframe(
        dataframe=test_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=False,
        # 補preprocess_input的正規化
        rescale=1./255
    )
    # =========================

    ## train model, k fold cross validation
    # ==================================================
    fold_no = 1
    test_acc = []
    test_f1 = []
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)
    for train_index, val_index in kfold.split(train_df, train_df['Label']):
        print("-------Training {}, fold = {} -------".format(chosen_model, fold_no))
        k_fold_train = train_df.iloc[train_index]
        k_fold_val = train_df.iloc[val_index]
        # print("K-fold training set label distribution:\n", k_fold_train['Label'].value_counts(normalize=True))
        # print("K-fold validation set label distribution:\n", k_fold_val['Label'].value_counts(normalize=True))
        
        ## train, validation image
        # =========================
        # 關閉翻轉
        train_generator = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=False)
        val_generator = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=False)
        train_images = train_generator.flow_from_dataframe(
            dataframe=k_fold_train,
            x_col='Filepath',
            y_col='Label',
            target_size=(224, 224),
            color_mode='rgb',
            class_mode='categorical',
            batch_size=64,
            shuffle=True,
            rescale=1./255,
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
            rescale=1./255,
            seed=42
        )
        # =========================
        
        ## load model
        # =========================
        # print(model.summary())
        # =========================

        ## compile and evaluate
        # =========================
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

        ## early stop 
        # callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        
        ## no early stop
        model.fit(train_images, validation_data=val_images, epochs=30)

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
    acc_mean  = np.round(np.mean(test_acc), 2)
    acc_std = np.round(np.std(test_acc), 2)
    f1_mean  = np.round(np.mean(test_f1), 2)
    f1_std = np.round(np.std(test_f1), 2)
    # print(f"acc mean = {acc_mean}, std = {acc_std}")
    # print(test_acc)
    # print(f"f1  mean = {f1_mean}, std = {f1_std}")
    # print(test_f1)
    # =========================

    return {'acc_mean':acc_mean, 'acc_std':acc_std, 'test_acc':test_acc, 'f1_mean':f1_mean, 'f1_std':f1_std, 'test_f1':test_f1}


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
# path = "E://data_bone//4-a+b_swift_cut_正確//side"
path = "E://data_bone//4-a+b_swift_cut_正確//front"

# path = "E://data_bone//3-a+b_all_正確//side"
# path = "E://data_bone//3-a+b_all_正確//front"

# path = "E://data_bone//5-a+b_swift_cut_標準//front"
# path = "E://data_bone//5-a+b_swift_cut_標準//side"

# =========================

chosen_models = [
    'resnet50',
    # 'resnet152',
    'vgg16',
    'vgg19',
    'mobilenet',
    'mobilenet_v2',
    'efficientnet',
    'inception_v3'
]

results = []
for i in range(len(chosen_models)):
    tmp = train_original_cross(image_dir=path, class_count=3, chosen_model=chosen_models[i])
    results.append(tmp)

for i in range(len(chosen_models)):
    print(f'----------------------Model {chosen_models[i]}----------------------')
    print(f"acc mean = {results[i]['acc_mean']}, std = {results[i]['acc_std']}, test_acc = {results[i]['test_acc']}")
    print(f"f1  mean = {results[i]['f1_mean']}, std = {results[i]['f1_std']}, test_f1 = {results[i]['test_f1']}")


