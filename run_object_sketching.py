import sys
import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import argparse
# import multiprocessing as mp
import os
import subprocess as sp
# from shutil import copyfile

# import numpy as np
import torch
# from IPython.display import Image as Image_colab
# from IPython.display import display, SVG, clear_output
# from ipywidgets import IntSlider, Output, IntProgress, Button
# import time
import pdb

parser = argparse.ArgumentParser()
parser.add_argument("--target_file", type=str,
                    help="target image file, located in <target_images>")
parser.add_argument("--num_strokes", type=int, default=16,
                    help="number of strokes used to generate the sketch, this defines the level of abstraction.")
parser.add_argument("--num_iter", type=int, default=2001, help="number of iterations") # 2001 xxxx8888
parser.add_argument("--num_sketches", type=int, default=3,
                    help="it is recommended to draw 3 sketches and automatically chose the best one")
parser.add_argument('-cpu', action='store_true')
# parser.add_argument('--gpunum', type=int, default=0)

args = parser.parse_args()


abs_path = os.path.abspath(os.getcwd())

target = f"{abs_path}/target_images/{args.target_file}"
assert os.path.isfile(target), f"{target} does not exists!"

# U2Net_/saved_models/u2net.pth
if not os.path.isfile(f"{abs_path}/U2Net_/saved_models/u2net.pth"):
    sp.run(["gdown", "https://drive.google.com/uc?id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ",
           "-O", "U2Net_/saved_models/"])

test_name = os.path.splitext(args.target_file)[0] # 'camel'
output_dir = f"{abs_path}/output_sketches/{test_name}/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

num_iter = args.num_iter # 2001
save_interval = 10
use_gpu = not args.cpu

if not torch.cuda.is_available():
    use_gpu = False
    print("CUDA is not configured with GPU, running with CPU instead.")
    print("Note that this will be very slow, it is recommended to use colab.")

# args.num_sketches -- 3
seeds = list(range(0, args.num_sketches * 1000, 1000))
# seeds -- [0, 1000, 2000]

exit_codes = []
# manager = mp.Manager()
# losses_all = manager.dict()

def run(seed, wandb_name):
    # seed = 0
    # wandb_name = 'camel_16strokes_seed0'
    exit_code = sp.run(["python", "painterly_rendering.py", target, # target_images/camel.png
                            "--num_paths", str(args.num_strokes), # 16
                            "--output_dir", output_dir,
                            "--wandb_name", wandb_name,
                            "--num_iter", str(num_iter),
                            "--save_interval", str(save_interval),
                            "--seed", str(seed),
                            "--use_gpu", str(int(use_gpu)),
                            ])
    if exit_code.returncode:
        sys.exit(1)

    # config = np.load(f"{output_dir}/{wandb_name}/config.npy",
    #                  allow_pickle=True)[()]
    # loss_eval = np.array(config['loss_eval'])
    # inds = np.argsort(loss_eval)
    # losses_all[wandb_name] = loss_eval[inds][0]
 
for seed in seeds:
    wandb_name = f"{test_name}_{args.num_strokes}strokes_seed{seed}"
    run(seed, wandb_name)
