import os
from PIL import Image
from model import *
import torch
import torch.nn as nn
import torchvision.transforms as tfs 
import torchvision.utils as vutils
import datetime

import warnings
warnings.filterwarnings("ignore")

torch.cuda.empty_cache()

abs = os.getcwd()+'/'
img_dir = './data/test/'
output_dir = abs + f'./data/pred_results/'
print("pred_dir: ",output_dir)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

model_dir='./pretrain_model/KTDN.pth'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckp = torch.load(model_dir,map_location=device)
net = Dehaze()
net = nn.DataParallel(net)
net.load_state_dict(ckp['model'])
net.eval()

all_run_time = 0

for im in os.listdir(img_dir):
    print(f'process {im} ...')
    haze = Image.open(img_dir+im)
    hazy = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])
    ])(haze)[None, ::]
    haze_no = tfs.ToTensor()(haze)[None, ::]

    with torch.no_grad():
        torch.cuda.synchronize()
        start_time = datetime.datetime.now()

        pred, _ = net(hazy)

        torch.cuda.synchronize()
        end_time = datetime.datetime.now()
        run_time = (end_time - start_time).microseconds
        all_run_time += run_time/1000000.0

    ts = torch.squeeze(pred.clamp(0,1).cpu())
    vutils.save_image(ts, output_dir+im.split('.')[0]+'.png')

print(f'runtime per image: {all_run_time/5.0:.1f}s')
