import sys
import os

# Ensure FastSAM is in path
sys.path.append(os.path.abspath('FastSAM'))

from FastSAM.fastsam import FastSAM, FastSAMPrompt

# Load model
model = FastSAM('FastSAM/weights/FastSAM-x.pt')
DEVICE = 'cpu'

# Input and output directories
INPUT_DIR = 'Synthetic faces'
OUTPUT_DIR = 'output/synthetic'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Loop over all image files in the input directory
for filename in os.listdir(INPUT_DIR):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(INPUT_DIR, filename)
        print(f"Processing {image_path}")

        # Run FastSAM
        everything_results = model(
            image_path, device=DEVICE,
            retina_masks=True, imgsz=1024,
            conf=0.2, iou=0.7
        )
        prompt_process = FastSAMPrompt(image_path, everything_results, device=DEVICE)

        # Point prompt: adjust if needed
        ann = prompt_process.point_prompt(points=[[512, 512]], pointlabel=[1])

        # Output path
        output_path = os.path.join(OUTPUT_DIR, filename)
        prompt_process.plot(annotations=ann, output_path=output_path)

        print(f"Saved output to {output_path}")
