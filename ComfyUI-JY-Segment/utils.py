from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
import cv2
import numpy as np
from torchvision.transforms.functional import to_pil_image
import torch
import json
import os

def extract_segmented_regions(image, anns):
    """
    提取分割区域的像素数据。

    参数:
        image: 原始图像。
        anns: 生成的掩码列表。

    返回:
        regions_info: 一个包含每个分割区域的像素数据及其边界框信息的列表。
    """
    regions_info = []
    for ann in anns:
        mask = ann['segmentation']
        y_indices, x_indices = np.where(mask)
        if len(y_indices) == 0 or len(x_indices) == 0:
            continue
        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()
        cropped_region = image[y_min:y_max+1, x_min:x_max+1]
        cropped_mask = mask[y_min:y_max+1, x_min:x_max+1]
        regions_info.append({
            'image_data': cropped_region,
            'mask_data': cropped_mask,
            'bbox': (x_min, y_min, x_max, y_max)
        })
    return regions_info
    
def extract_segmented_masks(image, anns):
    height, width = image.shape[:2]
    matrix = [[0 for _ in range(width)] for _ in range(height)]
    area_num = 1;
    for ann in anns:
        mask = ann['segmentation']
        y_indices, x_indices = np.where(mask)
        if len(y_indices) == 0 or len(x_indices) == 0:
            continue
        for i in range(len(y_indices)):
            matrix[y_indices[i]][x_indices[i]] = area_num
        area_num = area_num+1
    return matrix    

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

def get_results(input_image):
    input_image = input_image.squeeze(0)
    input_image = input_image.permute((2, 0, 1))
    #print(input_image.shape)
    image = np.array(to_pil_image(input_image))
    results = {"segments":[[]]}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_path = os.path.dirname(os.path.abspath(__file__))
    model_path = current_path+"\\..\\..\\models\\sams\\sam_vit_h_4b8939.pth"
    #print("当前路径:", model_path)
    sam = sam_model_registry["default"](checkpoint=model_path)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    #image = cv2.imread('D:\\liangyanan\\3.png')
    masks = mask_generator.generate(image)
    # 提取分割区域
    matrix = extract_segmented_masks(image, masks)
    results["segments"] = matrix
    the_str = json.dumps(results, ensure_ascii=False, indent=None,separators=(',', ':'))
    return the_str;

