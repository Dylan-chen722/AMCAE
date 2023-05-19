# mask methods for image
import random
import numpy as np

# random mask for image
# mask_size: size of mask, mask_ratio: ratio of mask area to image area
def apply_random_mask(image, img_size, mask_size, mask_ratio):
    
    mask_area = []
    x1, y1 = 0, 0
    mask_step = int(img_size/mask_size)

    for i in range(mask_step):
        x1 = i*mask_size
        x2 = x1 + mask_size
        for j in range(mask_step):
            y1 = j*mask_size
            y2 = y1 + mask_size
            mask_area.append([x1,x2,y1,y2])

    mask_corn = random.sample(mask_area, int(mask_ratio*np.power(mask_step, 2)))
    masked_img = image.clone()
    
    for mask in mask_corn:    
        x1, x2, y1, y2 = mask[0], mask[1], mask[2], mask[3]   
        masked_img[:, x1:x2, y1:y2] = 0

    return masked_img

# regional mask for image
# x1, x2, y1, y2: mask area
def apply_region_mask(image, x1, x2, y1, y2):

    masked_img = image.clone()
    masked_img[:, x1:x2, y1:y2] = 0

    return masked_img



