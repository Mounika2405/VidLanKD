import os
import random
import json

path = '/ssd_scratch/users/mounika.k/'
filenames = os.listdir(path + 'HowTo100M_clip/')
filenames = [file.replace('.npy', '') for file in filenames]
random.seed(5)
random.shuffle(filenames)

train_split = 0.7
val_split = 0.2
train_keys = filenames[:int(len(filenames)*train_split)]
val_keys = filenames[len(train_keys): len(train_keys) + int(len(filenames)*val_split)]
test_keys = filenames[len(train_keys) + len(val_keys):]

# print('train_keys', train_keys)
# print('test_keys', test_keys)
# print('val_keys', val_keys)
with open(path + 'caption.json', 'r') as f:
    caption_data = json.load(f)

train_data = {}
val_data = {}
test_data = {}

for key in train_keys:
    if key in caption_data.keys():
        train_data[key] = caption_data[key]  

for key in val_keys:
    if key in caption_data.keys():
        val_data[key] = caption_data[key] 

for key in test_keys:
    if key in caption_data.keys():
        test_data[key] = caption_data[key] 

# print('train_data', train_data)
# print('test_data', test_data)
# print('val_data', val_data)

with open(path + 'train_3k_keys.json', 'w') as f:
    json.dump(train_keys, f)

with open(path + 'train_3k_data.json', 'w') as f:
    json.dump(train_data, f)

with open(path + 'val_3k_keys.json', 'w') as f:
    json.dump(val_keys, f)

with open(path + 'val_3k_data.json', 'w') as f:
    json.dump(val_data, f)

with open(path + 'test_3k_keys.json', 'w') as f:
    json.dump(test_keys, f)

with open(path + 'test_3k_data.json', 'w') as f:
    json.dump(test_data, f)