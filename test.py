import torch
import numpy as np
import os
from torchvision.datasets.folder import is_image_file, default_loader

data = np.array([1,2,3,4])
torch_data = torch.from_numpy(data)
print(
    '\ndata',data,
    '\ntorch_data',torch_data
)
part_file = open('./utils/parts_train.txt','r')
contents = part_file.readlines()
for i in range(len(contents)):
    contents[i].replace(' ','_').strip('\n')
print(contents)
images=[]
for root, _, fnames in sorted(os.walk('./datasets')):
    for fname in fnames:
        if is_image_file(fname):
            if fname in contents:
                path = os.path.join(root, fname)
                item = path
                images.append(item)
