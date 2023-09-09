import os
import sys
import time
import traceback

import numpy as np
import PIL
import torch
import torch.nn as nn
import wandb
from PIL import Image
from torchvision import transforms

from tqdm.auto import tqdm

import config
import sketch_utils as utils
from models.loss import Loss
from models.painter_params import Painter, PainterOptimizer
import pdb

def load_renderer(args, target_im=None, mask=None):
    renderer = Painter(num_strokes=args.num_paths, 
                       args=args,
                       num_segments=args.num_segments,
                       imsize=args.image_scale, # 224
                       device=args.device,
                       target_im=target_im,
                       mask=mask)
    renderer = renderer.to(args.device)
    return renderer


def get_target(args):
    # args.image_scale -- 224
    target = Image.open(args.target).convert("RGB") # target_images/camel.png
    mask = utils.get_mask_u2net(args, target)

    transforms_ = []
    if target.size[0] != target.size[1]:
        transforms_.append(transforms.Resize((args.image_scale, args.image_scale), interpolation=PIL.Image.BICUBIC))
    else:
        transforms_.append(transforms.Resize(args.image_scale, interpolation=PIL.Image.BICUBIC))
        transforms_.append(transforms.CenterCrop(args.image_scale))
    transforms_.append(transforms.ToTensor())
    data_transforms = transforms.Compose(transforms_)
    target_ = data_transforms(target).unsqueeze(0).to(args.device)

    return target_, mask


def main(args):
    loss_func = Loss(args)
    inputs, mask = get_target(args)
    utils.log_input(inputs, args.output_dir)
    renderer = load_renderer(args, inputs, mask)

    optimizer = PainterOptimizer(args, renderer)
    counter = 0
    configs_to_save = {"loss_eval": []}
    best_loss, best_fc_loss = 100, 100
    best_iter, best_iter_fc = 0, 0
    min_delta = 1e-5
    terminate = False

    img = renderer.init_image(stage=0)
    optimizer.init_optimizers()

    epoch_range = tqdm(range(args.num_iter))

    for epoch in epoch_range:
        epoch_range.refresh()

        start = time.time()
        optimizer.zero_grad_()
        sketches = renderer.get_image().to(args.device)
        losses_dict = loss_func(sketches, inputs.detach(), counter)
        loss = sum(list(losses_dict.values()))
        loss.backward()

        optimizer.step_()
        if epoch % args.save_interval == 0:
            utils.plot_batch(inputs, sketches, f"{args.output_dir}/jpg_logs", counter, 
                title=f"iter{epoch}.jpg")
            renderer.save_svg(f"{args.output_dir}/svg_logs", f"svg_iter{epoch}")

        if epoch % args.eval_interval == 0:
            with torch.no_grad():
                losses_dict_eval = loss_func(sketches, inputs, counter, mode="eval")
                loss_eval = sum(list(losses_dict_eval.values()))
                configs_to_save["loss_eval"].append(loss_eval.item())
                for k in losses_dict_eval.keys():
                    if k not in configs_to_save.keys():
                        configs_to_save[k] = []
                    configs_to_save[k].append(losses_dict_eval[k].item())

                if args.clip_fc_loss_weight: # args.clip_fc_loss_weight -- 0.1
                    if losses_dict_eval["fc"].item() < best_fc_loss:
                        best_fc_loss = losses_dict_eval["fc"].item() / args.clip_fc_loss_weight
                        best_iter_fc = epoch

                cur_delta = loss_eval.item() - best_loss
                if abs(cur_delta) > min_delta:
                    if cur_delta < 0:
                        best_loss = loss_eval.item()
                        best_iter = epoch
                        terminate = False
                        utils.plot_batch(inputs, sketches, args.output_dir, counter, title="best_iter.jpg")
                        renderer.save_svg(args.output_dir, "best_iter")

                if abs(cur_delta) <= min_delta:
                    if terminate:
                        break
                    terminate = True

        if counter == 0:
            utils.plot_atten(renderer.get_attention_map(), 
                             renderer.get_thresh(), 
                             inputs, renderer.get_inds(),
                             "{}/{}.jpg".format(args.output_dir, "attention_map"))

        counter += 1

    renderer.save_svg(args.output_dir, "final_svg")
    path_svg = os.path.join(args.output_dir, "best_iter.svg")
    utils.log_sketch_summary_final(path_svg, args.device, best_iter, best_loss, "best total")

    return configs_to_save

if __name__ == "__main__":
    args = config.parse_arguments()
    final_config = vars(args)
    try:
        configs_to_save = main(args)
    except BaseException as err:
        print(f"Unexpected error occurred:\n {err}")
        print(traceback.format_exc())
        sys.exit(1)
