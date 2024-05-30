import numpy as np
import tensorflow as tf
from keras.preprocessing import image
import matplotlib.cm as cm

# load the models when import "predictions.py"
# front
front_model_hand_frac = tf.keras.models.load_model("weights3/HAND_front_frac.h5")
front_model_humerus_frac = tf.keras.models.load_model("weights3/HUMERUS_front_frac.h5")
front_model_wrist_frac = tf.keras.models.load_model("weights3/WRIST_front_frac.h5")
front_imagenet_frac = tf.keras.models.load_model("weights3/imagenet_front_frac.h5")
front_imagenet_3part_frac = tf.keras.models.load_model("weights3/imagenet_3part_front_frac.h5")
# side
side_model_hand_frac = tf.keras.models.load_model("weights3/HAND_side_frac.h5")
side_model_humerus_frac = tf.keras.models.load_model("weights3/HUMERUS_side_frac.h5")
side_model_wrist_frac = tf.keras.models.load_model("weights3/WRIST_side_frac.h5")
side_imagenet_frac = tf.keras.models.load_model("weights3/imagenet_side_frac.h5")
side_imagenet_3part_frac = tf.keras.models.load_model("weights3/imagenet_3part_side_frac.h5")

categories_parts = ["HAND_front", "HUMERUS_front", "WRIST_front", "imagenet_front", "imagenet_3part_front", 
                    "HAND_side", "HUMERUS_side", "WRIST_side", "imagenet_side", "imagenet_3part_side"
                    ]

# categories_fracture = ['fractured', 'normal']
categories_fracture = ['bi','tri', 'normal']

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

# Parts - bone type predict model of 3 classes
# otherwise - fracture predict for each part
def predict(img, model):
    size = 224
    # front
    if model == 'HAND_front':
        chosen_model = front_model_hand_frac
    elif model == 'HUMERUS_front':
        chosen_model = front_model_humerus_frac
    elif model == 'WRIST_front':
        chosen_model = front_model_wrist_frac
    elif model == 'imagenet_front':
        chosen_model = front_imagenet_frac
    elif model == 'imagenet_3part_front':
        chosen_model = front_imagenet_3part_frac
    # side
    elif model == 'HAND_side':
        chosen_model = side_model_hand_frac
    elif model == 'HUMERUS_side':
        chosen_model = side_model_humerus_frac
    elif model == 'WRIST_side':
        chosen_model = side_model_wrist_frac
    elif model == 'imagenet_side':
        chosen_model = side_imagenet_frac
    elif model == 'imagenet_3part_side':
        chosen_model = side_imagenet_3part_frac


    temp_img = image.load_img(img, target_size=(size, size))
    x = image.img_to_array(temp_img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    prediction = np.argmax(chosen_model.predict(images), axis=1)

    prediction_str = categories_fracture[prediction.item()]
    
    # 存取cam
    heatmap = make_gradcam_heatmap(images, chosen_model)
    p_cam = img.replace("/test2/", "/cam2/", 1)
    save_and_display_gradcam(img, heatmap, cam_path=f"{p_cam}_cam.jpg")
    

    return prediction_str
