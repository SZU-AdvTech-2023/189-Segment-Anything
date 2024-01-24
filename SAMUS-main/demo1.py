import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
from models.segment_anything_samus import SamAutomaticMaskGenerator
from models.segment_anything import sam_model_registry
#automatic-mask-generator


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


image = cv2.imread('/data/sym/Echocardiography-CAMUS/dataset/camus320x320/img/patient0001-2CH/0.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis('off')
plt.show()

sys.path.append("..")

sam_checkpoint = "/data/sym/SAM-demo/sam_vit_l_0b3195.pth"
model_type = "vit_l"


sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(image)
print(len(masks))
print(masks[0].keys())
print(masks[0])
print(type(masks))

plt.figure(figsize=(10, 10))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show()
