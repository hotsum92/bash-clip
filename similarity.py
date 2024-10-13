import os
import torch
import clip
from PIL import Image
import argparse
import sys
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('prompt', type=str)
args = parser.parse_args()

model_name = "ViT-B/32"
device = "cuda" if torch.cuda.is_available() else "cpu"
images_path = './images'

prompt = args.prompt

model, preprocess = clip.load(model_name, device=device)
text = clip.tokenize(prompt).to(device)

with torch.no_grad():
    text_features = model.encode_text(text)

text_features /= text_features.norm(dim=-1, keepdim=True)

for root, dirs, files in os.walk(images_path):
    for file in files:

        try:
            image = Image.open(os.path.join(root, file))
        except:
            continue

        image = torch.cat([ preprocess(image).unsqueeze(0) ]).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)

        image_features /= image_features.norm(dim=-1, keepdim=True)

        text_probs = torch.cosine_similarity(image_features, text_features)

        print(",".join(np.append(file, text_probs.cpu().numpy()[0].astype(str))))
