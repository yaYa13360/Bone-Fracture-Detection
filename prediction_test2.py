import os
from colorama import Fore
from predictions2 import predict


# bone prediction with cam
def load_path(path):
    dataset = []
    for root, dirs, files in os.walk(path):
        for file in files:
            # label
            label = ""
            if "normal" in root:
                label = "normal"
            elif "tri" in root:
                label = "tri"
            elif "bi" in root:
                label = "bi"
            # else:
            #     label = "fractured"


            dataset.append(
                            {   
                                'uuid': root.split("_")[1],
                                'label': label,
                                'image_path': os.path.join(root, file)
                            }
                        )


    return dataset

def reportPredict(dataset, categories_part):
    status_count = 0

    print(Fore.YELLOW +
          '{0: <28}'.format('Name') +
          '{0: <20}'.format('Status') +
          '{0: <20}'.format('Predicted Status'))
    for img in dataset:
        # 不是用身體部位而改用指定模型
        fracture_predict = predict(img['image_path'], categories_part)
        if img['label'] == fracture_predict:
            status_count = status_count + 1
            color = Fore.GREEN
        else:
            color = Fore.RED
        print(color +
              '{0: <20}'.format((img['label'])) +
              '{0: <20}'.format(fracture_predict))

    print(Fore.BLUE + 'status acc: ' + str("%.2f" % (status_count / len(dataset) * 100)) + '%')
    return


# categories_parts = ["HAND_front", "HUMERUS_front", "WRIST_front", "imagenet_front", 
#                     "HAND_side", "HUMERUS_side", "WRIST_side", "imagenet_side"
#                     ]
categories_parts = ["imagenet_3part_front",
                    "imagenet_3part_side"
                    ]
for p in categories_parts:
    test_dir = 'D://reaserch//Bone-Fracture-Detection//test2//' + p + '//'
    print(f"----------------{p}----------------")
    reportPredict(load_path(test_dir), p)
