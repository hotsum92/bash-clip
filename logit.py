import os
import torch
import clip
from PIL import Image
import argparse
import sys
import numpy as np
import re

parser = argparse.ArgumentParser()
parser.add_argument('prompt', type=str)
args = parser.parse_args()

model_name = "ViT-B/32"
device = "cuda" if torch.cuda.is_available() else "cpu"
images_path = './images/'

prompt = args.prompt.split(',')

model, preprocess = clip.load(model_name, device=device)
text = clip.tokenize(prompt).to(device)

for root, dirs, files in os.walk(images_path):
    for file in files:

        try:
            image = Image.open(os.path.join(root, file))
        except:
            continue

        image = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)

            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        path = re.sub('^\./images/', '', root)
        path = os.path.join(path, file)

        print(",".join(np.append(path, probs.astype(str))))
