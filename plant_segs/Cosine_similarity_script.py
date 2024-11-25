#Cosine similarity script 
import shutil
from sam2.build_sam import build_sam2
# from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
import cv2
import torch
import tempfile
import os
import glob
from tqdm import tqdm
import numpy as np
import pickle
import time
import socketserver

# Set up the device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

temp_dir = tempfile.mkdtemp()
sam2_checkpoint = "/opt/sam2/checkpoints/sam2.1_hiera_tiny.pt"
# model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
shutil.copy("/work/no_pos_sam2.1_hiera_t.yaml", "/opt/sam2/sam2/configs/sam2.1/no_pos_sam2.1_hiera_t.yaml")
model_cfg = "configs/sam2.1/no_pos_sam2.1_hiera_t.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
# predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)

#Load the files
image_path_list = glob.glob('/work/20230816_r_ios/*/*.jpg')
# image_list = [cv2.resize(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB), (512, 512))[150:406,150:406,:] for image_path in image_path_list]
image_list = [cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB) for image_path in image_path_list]
print(f"Loaded {len(image_list)} images")
good_frame_path = "/work/20230816_r_ios/20230816_192437/214.jpg"
bad_frame_path = "/work/20230816_r_ios/20230816_192437/218.jpg"
good_image = cv2.cvtColor(cv2.imread(good_frame_path), cv2.COLOR_BGR2RGB)
bad_image = cv2.cvtColor(cv2.imread(bad_frame_path), cv2.COLOR_BGR2RGB)
samples = []
for img in [good_image, bad_image]:
    predictor.set_image(img)
    embedding = predictor.get_image_embedding()
    flat = embedding.view(embedding.size(0), -1).cpu()
    emb_np = flat.numpy()
    samples.append(emb_np)

embeddings_file = 'embeddings.pkl'
if os.path.exists(embeddings_file):
    with open(embeddings_file, 'rb') as f:
        all_embeddings = pickle.load(f)
else:
    all_embeddings = []
    start = time.time()
    batch_size = 10
    for i in tqdm(range(0, len(image_list), batch_size), desc="Processing images"):
        image_batch = image_list[i:i + batch_size]
        # Ensure image_list is a list
        if not isinstance(image_batch, list):
            image_batch = [image_batch]
        predictor.set_image_batch(image_batch)
        list_of_embeddings = predictor.get_image_embedding()
        list_embedding_reshaped = list_of_embeddings[:].view(list_of_embeddings[:].size(0), -1)
        all_embeddings.append(list_embedding_reshaped.cpu())
    with open('embeddings.pkl', 'wb') as file:
        pickle.dump(all_embeddings, file)
    end = time.time()
    print(f"Generating and storing embeddings took: {(end-start):.2f} sec")

all_embeddings_flattened = [embedding.view(embedding.size(0), -1) for embedding in all_embeddings]
all_embeddings_np = np.vstack([embedding.numpy() for embedding in all_embeddings_flattened])

def normalize(vec: np.ndarray):
    if vec.ndim == 1:
        return vec / np.linalg.norm(vec)
    return vec / np.linalg.norm(vec, axis=1, keepdims=True)

# Free memory by deleting the embeddings
del all_embeddings
del all_embeddings_flattened
torch.cuda.empty_cache()
all_embeddings_np_norm = [normalize(emb) for emb in all_embeddings_np]

#manually select a good and bad sample from the list
print(f"all_embeddings_np_norm shape: {len(all_embeddings_np_norm)}")
good_sample = normalize(samples[0])
bad_sample = normalize(samples[1])

def extract_features(good,bad,big_list):
    ret = []
    for img_embedding in tqdm(big_list, desc="Calculating cosine similarities"):
        x = np.dot(good, img_embedding.T)
        y = np.dot(bad, img_embedding.T)
        ret.append((x, y))
    return ret

cos_similarity_space = extract_features(good_sample, bad_sample, all_embeddings_np_norm)
# Combine the indices with the similarity scores
indexed_cos_similarity_space = list(enumerate(cos_similarity_space))

# Sort by good sample similarity (descending) and bad sample similarity (ascending)
sorted_cos_similarity_space = sorted(indexed_cos_similarity_space, key=lambda x: (-x[1][0], x[1][1]))
# Filter sorted_cos_similarity_space for all entries where x[1][0] > x[1][1]
filtered_cos_sim_space = [x for x in sorted_cos_similarity_space if x[1][0] > x[1][1]]

filtered_image_indices = [index for index, _ in filtered_cos_sim_space]

for rank, idx in enumerate(filtered_image_indices):
    img = image_list[idx]
    # Apply sharpening filter to the image
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_img = cv2.filter2D(img, -1, kernel)
    img = cv2.rotate(sharpened_img, cv2.ROTATE_90_CLOCKWISE)
    img_path = os.path.join(temp_dir, f"filtered.image.{rank:04d}.{idx}.jpg")
    cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

print(f"Filtered images saved in: {temp_dir}")

import http.server

PORT = 8000

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=temp_dir, **kwargs)

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Serving HTTP on 0.0.0.0 port {PORT} (http://0.0.0.0:{PORT}/) ...")
    httpd.serve_forever()
