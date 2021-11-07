import torch
import clip
from PIL import Image
import os
import numpy as np
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

## Extraction function
def extract_clip_features(data_path, folder, batch_size, out_path):
    folder_clip_features = []
    image_filenames = sorted(os.listdir(os.path.join(data_path, folder)), key=lambda x: int(os.path.splitext(x)[0]))

    for start_idx in tqdm(range(0, len(image_filenames), batch_size)):
        end_idx = min(len(image_filenames), start_idx + batch_size)

        images = [preprocess(Image.open(os.path.join(data_path, folder, image_filename))) for image_filename in image_filenames[start_idx: end_idx]]
        images = torch.stack(images).to(device)

        with torch.no_grad():
            image_features = model.encode_image(images)
            image_features = image_features.cpu().detach().numpy()
            folder_clip_features.append(image_features)

    folder_clip_features = np.concatenate(folder_clip_features)
    np.save(os.path.join(out_path, f'{folder}.npy'), folder_clip_features)


## Define variables here
out_path = '/ssd_scratch/users/mounika.k/features/howto100m_clipfeature/'
os.makedirs(out_path, exist_ok=True)

data_path = '/ssd_scratch/users/mounika.k/Frames_folders/'
batch_size = 16

folders = os.listdir(data_path)

# Skip folders for which output already exists
skip_existing = True
if skip_existing:
    existing = [os.path.splitext(filename)[0] for filename in os.listdir(out_path)] 
    folders = set(folders).difference(set(existing))
    folders = list(folders)

for folder in folders:
    try:
        extract_clip_features(data_path, folder, batch_size, out_path)
    except Exception as e:
        print(f'Extraction failed for folder {folder}')
        print(e)
