# # sudo cog push r8.im/yael-vinker/clipasso

# # Prediction interface for Cog ⚙️
# # https://github.com/replicate/cog/blob/main/docs/python.md

# import warnings

# warnings.filterwarnings('ignore')
# warnings.simplefilter('ignore')

# from cog import BasePredictor, Input, Path
# import subprocess as sp
# import os
# import re

# import imageio
# import matplotlib.pyplot as plt
# import numpy as np
# import pydiffvg
# import torch
# from PIL import Image
# import multiprocessing as mp
# from shutil import copyfile

# import argparse
# import math
# import sys
# import time
# import traceback

# import PIL
# import torch.nn as nn
# import torch.nn.functional as F
# import wandb
# from torchvision import models, transforms
# from tqdm import tqdm

# import config
# import sketch_utils as utils
# from models.loss import Loss
# from models.painter_params import Painter, PainterOptimizer
# import pdb

# class Args():
#     def __init__(self, config):
#         for k in config.keys():
#             setattr(self, k, config[k])
        

# def load_renderer(args, target_im=None, mask=None):
#     renderer = Painter(num_strokes=args.num_paths, args=args,
#                        num_segments=args.num_segments,
#                        imsize=args.image_scale,
#                        device=args.device,
#                        target_im=target_im,
#                        mask=mask)
#     renderer = renderer.to(args.device)
#     return renderer


# def get_target(args):
#     target = Image.open(args.target)
#     if target.mode == "RGBA":
#         # Create a white rgba background
#         new_image = Image.new("RGBA", target.size, "WHITE")
#         # Paste the image on the background.
#         new_image.paste(target, (0, 0), target)
#         target = new_image
#     target = target.convert("RGB")
#     masked_im, mask = utils.get_mask_u2net(args, target)

#     transforms_ = []
#     if target.size[0] != target.size[1]:
#         transforms_.append(transforms.Resize(
#             (args.image_scale, args.image_scale), interpolation=PIL.Image.BICUBIC))
#     else:
#         transforms_.append(transforms.Resize(
#             args.image_scale, interpolation=PIL.Image.BICUBIC))
#         transforms_.append(transforms.CenterCrop(args.image_scale))
#     transforms_.append(transforms.ToTensor())
#     data_transforms = transforms.Compose(transforms_)
#     target_ = data_transforms(target).unsqueeze(0).to(args.device)
#     return target_, mask


# def main(args):
#     loss_func = Loss(args)
#     inputs, mask = get_target(args)
#     utils.log_input(0, inputs, args.output_dir)
#     renderer = load_renderer(args, inputs, mask)

#     optimizer = PainterOptimizer(args, renderer)
#     counter = 0
#     configs_to_save = {"loss_eval": []}
#     best_loss, best_fc_loss = 100, 100
#     best_iter, best_iter_fc = 0, 0
#     min_delta = 1e-5
#     terminate = False

#     renderer.set_random_noise(0)
#     img = renderer.init_image(stage=0)
#     optimizer.init_optimizers()

#     for epoch in tqdm(range(args.num_iter)):
#         renderer.set_random_noise(epoch)
#         if args.lr_scheduler:
#             optimizer.update_lr(counter)

#         start = time.time()
#         optimizer.zero_grad_()
#         sketches = renderer.get_image().to(args.device)
#         losses_dict = loss_func(sketches, inputs.detach(
#         ), renderer.get_color_parameters(), renderer, counter, optimizer)
#         loss = sum(list(losses_dict.values()))
#         loss.backward()
#         optimizer.step_()
#         if epoch % args.save_interval == 0:
#             utils.plot_batch(inputs, sketches, f"{args.output_dir}/jpg_logs", counter,
#                              title=f"iter{epoch}.jpg")
#             renderer.save_svg(
#                 f"{args.output_dir}/svg_logs", f"svg_iter{epoch}")
#             # if args.cog_display:
#             #     yield Path(f"{args.output_dir}/svg_logs/svg_iter{epoch}.svg")


#         if epoch % args.eval_interval == 0:
#             with torch.no_grad():
#                 losses_dict_eval = loss_func(sketches, inputs, renderer.get_color_parameters(
#                 ), renderer.get_points_parans(), counter, optimizer, mode="eval")
#                 loss_eval = sum(list(losses_dict_eval.values()))
#                 configs_to_save["loss_eval"].append(loss_eval.item())
#                 for k in losses_dict_eval.keys():
#                     if k not in configs_to_save.keys():
#                         configs_to_save[k] = []
#                     configs_to_save[k].append(losses_dict_eval[k].item())
#                 if args.clip_fc_loss_weight:
#                     if losses_dict_eval["fc"].item() < best_fc_loss:
#                         best_fc_loss = losses_dict_eval["fc"].item(
#                         ) / args.clip_fc_loss_weight
#                         best_iter_fc = epoch
#                 # print(
#                 #     f"eval iter[{epoch}/{args.num_iter}] loss[{loss.item()}] time[{time.time() - start}]")

#                 cur_delta = loss_eval.item() - best_loss
#                 if abs(cur_delta) > min_delta:
#                     if cur_delta < 0:
#                         best_loss = loss_eval.item()
#                         best_iter = epoch
#                         terminate = False
#                         utils.plot_batch(
#                             inputs, sketches, args.output_dir, counter, title="best_iter.jpg")
#                         renderer.save_svg(args.output_dir, "best_iter")

#                 if abs(cur_delta) <= min_delta:
#                     if terminate:
#                         break
#                     terminate = True

#         if counter == 0 and args.attention_init:
#             utils.plot_atten(renderer.get_attn(), renderer.get_thresh(), inputs, renderer.get_inds(),
#                              "{}/{}.jpg".format(
#                                  args.output_dir, "attention_map"),
#                              args.saliency_model)

#         counter += 1

#     renderer.save_svg(args.output_dir, "final_svg")
#     path_svg = os.path.join(args.output_dir, "best_iter.svg")
#     utils.log_sketch_summary_final(
#         path_svg, args.device, best_iter, best_loss, "best total")

#     return configs_to_save


# def read_svg(path_svg, multiply=False):
#     pdb.set_trace()

#     device = torch.device("cuda" if (
#         torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")
#     canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(
#         path_svg)
#     if multiply:
#         canvas_width *= 2
#         canvas_height *= 2
#         for path in shapes:
#             path.points *= 2
#             path.stroke_width *= 2
#     _render = pydiffvg.RenderFunction.apply
#     scene_args = pydiffvg.RenderFunction.serialize_scene(
#         canvas_width, canvas_height, shapes, shape_groups)
#     img = _render(canvas_width,  # width
#                   canvas_height,  # height
#                   2,   # num_samples_x
#                   2,   # num_samples_y
#                   0,   # seed
#                   None,
#                   *scene_args)
#     img = img[:, :, 3:4] * img[:, :, :3] + \
#         torch.ones(img.shape[0], img.shape[1], 3,
#                    device=device) * (1 - img[:, :, 3:4])
#     img = img[:, :, :3]
#     return img
