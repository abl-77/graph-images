import sys
import os

# Add the parent directory of FastSAM to the module search path
sys.path.append(os.path.abspath('FastSAM'))

from FastSAM.fastsam import FastSAM, FastSAMPrompt

model = FastSAM('FastSAM/weights/FastSAM-x.pt')
IMAGE_PATH = 'Synthetic faces/1.S.W.M.png'
DEVICE = 'cpu'
everything_results = model(IMAGE_PATH, device=DEVICE, retina_masks=True, imgsz=1024, conf=0.2, iou=0.7,)
prompt_process = FastSAMPrompt(IMAGE_PATH, everything_results, device=DEVICE)

# everything prompt
ann = prompt_process.everything_prompt()
#ann = prompt_process.text_prompt(text='the head')
# point prompt
# points default [[0,0]] [[x1,y1],[x2,y2]]
# point_label default [0] [1,0] 0:background, 1:foreground
#ann = prompt_process.point_prompt(points=[[512, 512]], pointlabel=[1])

prompt_process.plot(annotations=ann,output_path='output/1.S.W.M.png',)

import numpy as np

# Inside your loop, after ann = ...
ann_np = ann.cpu().numpy() if hasattr(ann, 'cpu') else ann  # Convert to numpy if it's a tensor

# Create .npy file path
basename = os.path.splitext('1.S.W.M.png')[0]  # removes extension like .png
npy_output_path = os.path.join('output', f"{basename}_masks.npy")

# Save the array
np.save(npy_output_path, ann_np)

mask_path = 'output/1.S.W.M_masks.npy'

# Load the array
masks = np.load(mask_path)
print(f"Loaded mask array shape: {masks.shape}")

import matplotlib.pyplot as plt

plt.imshow(masks[0], cmap='gray')
plt.title("First Mask")
plt.axis('off')
plt.show()
