import numpy as np
import tensorflow as tf
from keras.preprocessing import image
import matplotlib.cm as cm

# load the models when import "predictions.py"
model_elbow_frac = tf.keras.models.load_model("weights2/ResNet50_XR_ELBOW_frac.h5")
model_finger_frac = tf.keras.models.load_model("weights2/ResNet50_XR_FINGER_frac.h5")
model_forearm_frac = tf.keras.models.load_model("weights2/ResNet50_XR_FOREARM_frac.h5")
model_hand_frac = tf.keras.models.load_model("weights2/ResNet50_XR_Hand_frac.h5")
model_humerus_frac = tf.keras.models.load_model("weights2/ResNet50_XR_HUMERUS_frac.h5")
model_shoulder_frac = tf.keras.models.load_model("weights2/ResNet50_XR_Shoulder_frac.h5")
model_wrist_frac = tf.keras.models.load_model("weights2/ResNet50_XR_WRIST_frac.h5")
model_parts = tf.keras.models.load_model("weights2/ResNet50_BodyParts.h5")

# categories for each result by index

#   0-Elbow     1-Hand      2-Shoulder
categories_parts = ["XR_ELBOW", "XR_FINGER", "XR_FOREARM", "XR_HAND", "XR_HUMERUS", "XR_SHOULDER", "XR_WRIST"]


#   0-fractured     1-normal
categories_fracture = ['fractured', 'normal']

# Grad-CAM
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="conv5_block3_out", pred_index=None):
    # 建立一個模型，同時輸出最後一個卷積層和整個模型的預測結果
    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    # 計算對於輸入圖像的預測類別，相對於最後一個卷積層的梯度
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # 輸出分類神經元相對於最後一個卷積層的輸出特徵圖的梯度
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # 這是一個向量，其中每個數字都是特定特徵圖通道上的梯度的平均強度
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 將特徵圖乘以權重，等於該特徵圖中的某些區域對於該分類的重要性
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap) # 然後將所有通道相加以獲得熱圖

    # 為了視覺化，將熱圖正規化0~1之間
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()



def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    # 載入原始圖像
    img = tf.keras.utils.load_img(img_path)
    img = tf.keras.utils.img_to_array(img)

    # 將熱圖重新縮放到0-255的範圍
    heatmap = np.uint8(255 * heatmap)

    # 使用Jet色彩映射將熱圖上色
    jet = cm.get_cmap("jet")

    # 使用Jet色彩映射的RGB值
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # 創建帶有RGB色彩的熱圖圖像
    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)

    # 在原始圖像上疊加熱圖
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.utils.array_to_img(superimposed_img)

    # 儲存疊加後的圖像
    superimposed_img.save(cam_path)



# get image and model name, the default model is "Parts"
# Parts - bone type predict model of 3 classes
# otherwise - fracture predict for each part
def predict(img, model="Parts"):
    size = 224
    if model == 'XR_ELBOW':
        chosen_model = model_elbow_frac
    elif model == 'XR_FINGER':
        chosen_model = model_finger_frac
    elif model == 'XR_FOREARM':
        chosen_model = model_forearm_frac
    elif model == 'XR_HAND':
        chosen_model = model_hand_frac
    elif model == 'XR_HUMERUS':
        chosen_model = model_humerus_frac
    elif model == 'XR_SHOULDER':
        chosen_model = model_shoulder_frac
    elif model == 'XR_WRIST':
        chosen_model = model_wrist_frac


    # load image with 224px224p (the training model image size, rgb)
    temp_img = image.load_img(img, target_size=(size, size))
    x = image.img_to_array(temp_img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    prediction = np.argmax(chosen_model.predict(images), axis=1)

    # chose the category and get the string prediction
    if model == 'Parts':
        prediction_str = categories_parts[prediction.item()]
    else:
        prediction_str = categories_fracture[prediction.item()]
    
    # 存取cam
    heatmap = make_gradcam_heatmap(images, chosen_model)
    p_cam = img.replace("/test/", "/cam/", 1)
    save_and_display_gradcam(img, heatmap, cam_path=f"{p_cam}_cam.jpg")
    

    return prediction_str
