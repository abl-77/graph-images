import sys
import os

# Add the parent directory of FastSAM to the module search path
sys.path.append(os.path.abspath('FastSAM'))

from FastSAM.fastsam import FastSAM, FastSAMPrompt

model = FastSAM('FastSAM/weights/FastSAM-x.pt')
IMAGE_PATH = 'Synthetic faces/1.S.W.M.png'
DEVICE = 'cpu'
everything_results = model(IMAGE_PATH, device=DEVICE, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9,)
prompt_process = FastSAMPrompt(IMAGE_PATH, everything_results, device=DEVICE)

# everything prompt
ann = prompt_process.everything_prompt()

prompt_process.plot(annotations=ann,output_path='output/1.S.W.M.png',)