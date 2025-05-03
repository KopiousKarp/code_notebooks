#!/usr/bin/env python3
import os
import fnmatch
import json
import clip
import joblib
import torch
from PIL import Image
from tqdm import tqdm
import argparse

def find_images(directory, exts=('*.png','*.jpg','*.jpeg')):
    image_files = []
    for root, _, filenames in os.walk(directory):
        for pat in exts:
            for fn in fnmatch.filter(filenames, pat):
                image_files.append(os.path.join(root, fn))
    return sorted(image_files)

def predict_filepaths(
    image_paths,
    clip_model,
    preprocess,
    classifier,
    device,
    checkpoint_path,
    save_every=100
):
    # load or init
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)
    else:
        data = {}

    clip_model.eval()
    for idx, img_path in enumerate(tqdm(image_paths, desc="Images")):
        if img_path in data:
            continue

        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"[WARN] could not open {img_path}: {e}")
            data[img_path] = None
            continue

        # CLIP’s preprocess gives a tensor [3×224×224]
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feats = clip_model.encode_image(img_tensor)     # [1×512]
            feats = feats.cpu().numpy()                     # → (1,512)
            pred = classifier.predict(feats)[0]
            data[img_path] = int(pred)

        if (idx+1) % save_every == 0:
            with open(checkpoint_path, 'w') as f:
                json.dump(data, f, indent=2)
    # final save
    with open(checkpoint_path, 'w') as f:
        json.dump(data, f, indent=2)
    return data

def main():
    p = argparse.ArgumentParser(
        description="Batch‐label a folder of images via CLIP+SVM"
    )
    p.add_argument('--image-dir',      required=True,  help="root folder of images")
    p.add_argument('--svm-path',       required=True,  help="joblib .joblib/.pkl of trained SVM")
    p.add_argument('--output-json',    default="labels.json", help="where to write predictions")
    p.add_argument('--batch-size',     type=int, default=8, help="unused for CLIP—here for future")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] using device {device}")

    # 1) load CLIP
    model, preprocess = clip.load('ViT-B/32', device)

    # 2) load your SVM
    classifier = joblib.load(args.svm_path)

    # 3) find all images
    img_paths = find_images(args.image_dir)

    # 4) run predictions
    predict_filepaths(
        image_paths=img_paths,
        clip_model=model,
        preprocess=preprocess,
        classifier=classifier,
        device=device,
        checkpoint_path=args.output_json,
        save_every=100
    )

if __name__ == "__main__":
    main()
