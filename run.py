import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil


from regularize import regularize_segmentations

mask = '/home/aoqiao/developer_dq/classlocation_oe/projectRegularization/images/mask_5996668.tif'
before = '/home/aoqiao/developer_dq/classlocation_oe/projectRegularization/images/before_5996668.tif'
after = '/home/aoqiao/developer_dq/classlocation_oe/projectRegularization/images/after_5996668.tif'
output = '/home/aoqiao/developer_dq/classlocation_oe/projectRegularization/output/'


os.makedirs(output, exist_ok=True)

# make a temporary folder, three subfolders: rgb, seg, reg_out
os.makedirs('temp', exist_ok=True)
os.makedirs('temp/rgb', exist_ok=True)
os.makedirs('temp/seg', exist_ok=True)
os.makedirs('temp/reg_out', exist_ok=True)

# preprocess the mask
# get the second channel of the mask and save it as a new tif in the seg folder
mask_img = cv2.imread(mask, cv2.IMREAD_UNCHANGED)
mask_after = mask_img[:,:,1]
cv2.imwrite('temp/seg/after.tif', mask_after)

mask_before = mask_img[:,:,2]
cv2.imwrite('temp/seg/before.tif', mask_before)

# copy the before and after images to the rgb folder
img_before = cv2.imread(before)
img_after = cv2.imread(after)
cv2.imwrite('temp/rgb/before.tif', img_before)
cv2.imwrite('temp/rgb/after.tif', img_after)

# regularize the segmentation
regularize_segmentations(img_folder='temp/rgb/*.tif', seg_folder='temp/seg/*.tif', out_folder='temp/reg_out/', in_mode="semantic", out_mode="instance", samples=None)

# postprocess the output
# save output 'before' as channel 2 with nonzero values=255, 'after' as channel 1 with nonzero values=255, for channel 0 all 0s
# save the output as a tif
output_img = np.zeros_like(mask_img)
output_img[:,:,0] = 0
output_img[:,:,1] = cv2.imread('temp/reg_out/after.tif', cv2.IMREAD_UNCHANGED)
output_img[:,:,2] = cv2.imread('temp/reg_out/before.tif', cv2.IMREAD_UNCHANGED)
output_img[output_img > 0] = 255
cv2.imwrite(output + 'output.tif', output_img)

# clear temp folders

shutil.rmtree('temp')