import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pydiffvg
import skimage
import skimage.io
import torch
import wandb
import PIL
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid
from skimage.transform import resize

from U2Net_.model import U2NET
import pdb

def imwrite(img, filename, gamma=2.2, normalize=False, wandb_name="", step=0, input_im=None):
    directory = os.path.dirname(filename)
    if directory != '' and not os.path.exists(directory):
        os.makedirs(directory)

    if not isinstance(img, np.ndarray):
        img = img.data.numpy()
    if normalize:
        img_rng = np.max(img) - np.min(img)
        if img_rng > 0:
            img = (img - np.min(img)) / img_rng
    img = np.clip(img, 0.0, 1.0)
    if img.ndim == 2:
        # repeat along the third dimension
        img = np.expand_dims(img, 2)
    img[:, :, :3] = np.power(img[:, :, :3], 1.0/gamma)
    img = (img * 255).astype(np.uint8)

    skimage.io.imsave(filename, img, check_contrast=False)
    images = [wandb.Image(Image.fromarray(img), caption="output")]
    if input_im is not None and step == 0:
        images.append(wandb.Image(input_im, caption="input"))


def plot_batch(inputs, outputs, output_dir, step, title):
    plt.figure()
    plt.subplot(2, 1, 1)
    grid = make_grid(inputs.clone().detach(), normalize=True, pad_value=2)
    npgrid = grid.cpu().numpy()
    plt.imshow(np.transpose(npgrid, (1, 2, 0)), interpolation='nearest')
    plt.axis("off")
    plt.title("inputs")

    plt.subplot(2, 1, 2)
    grid = make_grid(outputs, normalize=False, pad_value=2)
    npgrid = grid.detach().cpu().numpy()
    plt.imshow(np.transpose(npgrid, (1, 2, 0)), interpolation='nearest')
    plt.axis("off")
    plt.title("outputs")

    plt.tight_layout()
    plt.savefig("{}/{}".format(output_dir, title))
    plt.close()


def log_input(epoch, inputs, output_dir):
    grid = make_grid(inputs.clone().detach(), normalize=True, pad_value=2)
    npgrid = grid.cpu().numpy()
    plt.imshow(np.transpose(npgrid, (1, 2, 0)), interpolation='nearest')
    plt.axis("off")
    plt.tight_layout()
    plt.close()
    input_ = inputs[0].cpu().clone().detach().permute(1, 2, 0).numpy()
    input_ = (input_ - input_.min()) / (input_.max() - input_.min())
    input_ = (input_ * 255).astype(np.uint8)
    imageio.imwrite("{}/{}.png".format(output_dir, "input"), input_)


def log_sketch_summary_final(path_svg, device, epoch, loss, title):
    canvas_width, canvas_height, shapes, shape_groups = load_svg(path_svg)
    _render = pydiffvg.RenderFunction.apply
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_width, canvas_height, shapes, shape_groups)
    img = _render(canvas_width,  # width
                  canvas_height,  # height
                  2,   # num_samples_x
                  2,   # num_samples_y
                  0,   # seed
                  None,
                  *scene_args)

    img = img[:, :, 3:4] * img[:, :, :3] + \
        torch.ones(img.shape[0], img.shape[1], 3,
                   device=device) * (1 - img[:, :, 3:4])
    img = img[:, :, :3]
    plt.imshow(img.cpu().numpy())
    plt.axis("off")
    plt.title(f"{title} best res [{epoch}] [{loss}.]")
    plt.close()


def log_sketch_summary(sketch, title):
    plt.figure()
    grid = make_grid(sketch.clone().detach(), normalize=True, pad_value=2)
    npgrid = grid.cpu().numpy()
    plt.imshow(np.transpose(npgrid, (1, 2, 0)), interpolation='nearest')
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.close()


def load_svg(path_svg):
    svg = os.path.join(path_svg)
    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(svg)
    return canvas_width, canvas_height, shapes, shape_groups


def read_svg(path_svg, device, multiply=False):
    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(path_svg)
    pdb.set_trace()

    if multiply:
        canvas_width *= 2
        canvas_height *= 2
        for path in shapes:
            path.points *= 2
            path.stroke_width *= 2
    _render = pydiffvg.RenderFunction.apply
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_width, canvas_height, shapes, shape_groups)
    img = _render(canvas_width,  # width
                  canvas_height,  # height
                  2,   # num_samples_x
                  2,   # num_samples_y
                  0,   # seed
                  None,
                  *scene_args)
    img = img[:, :, 3:4] * img[:, :, :3] + \
        torch.ones(img.shape[0], img.shape[1], 3,
                   device=device) * (1 - img[:, :, 3:4])
    img = img[:, :, :3]
    return img


def plot_attn_clip(attn, threshold_map, inputs, inds, output_path):
    # output_path = 'output_sketches/camel/camel_16strokes_seed0/attention_map.jpg'

    # currently supports one image (and not a batch)
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    main_im = make_grid(inputs, normalize=True, pad_value=2)
    main_im = np.transpose(main_im.cpu().numpy(), (1, 2, 0))
    plt.imshow(main_im, interpolation='nearest')
    plt.scatter(inds[:, 1], inds[:, 0], s=10, c='red', marker='o')
    plt.title("input im")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(attn, interpolation='nearest', vmin=0, vmax=1)
    plt.title("atn map")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    threshold_map_ = (threshold_map - threshold_map.min()) / \
        (threshold_map.max() - threshold_map.min())
    plt.imshow(threshold_map_, interpolation='nearest', vmin=0, vmax=1)
    plt.title("prob softmax")
    plt.scatter(inds[:, 1], inds[:, 0], s=10, c='red', marker='o')
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_atten(attn, threshold_map, inputs, inds, output_path, saliency_model):
    plot_attn_clip(attn, threshold_map, inputs, inds, output_path)


def get_mask_u2net(args, pil_im):
    w, h = pil_im.size[0], pil_im.size[1]
    im_size = min(w, h)
    data_transforms = transforms.Compose([
        transforms.Resize(min(320, im_size), interpolation=PIL.Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(
            0.26862954, 0.26130258, 0.27577711)),
    ])

    input_im_trans = data_transforms(pil_im).unsqueeze(0).to(args.device)

    model_dir = os.path.join("./U2Net_/saved_models/u2net.pth")
    net = U2NET(3, 1)
    if torch.cuda.is_available() and args.use_gpu:
        net.load_state_dict(torch.load(model_dir))
        net.to(args.device)
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()
    with torch.no_grad():
        d1, d2_, d3_, d4_, d5_, d6_, d7_ = net(input_im_trans.detach())

    pred = d1[:, 0, :, :]
    pred = (pred - pred.min()) / (pred.max() - pred.min())
    predict = pred
    predict[predict < 0.5] = 0
    predict[predict >= 0.5] = 1
    mask = torch.cat([predict, predict, predict], axis=0).permute(1, 2, 0)
    mask = mask.cpu().numpy()
    mask = resize(mask, (h, w), anti_aliasing=False)
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    
    im = Image.fromarray((mask[:, :, 0]*255).astype(np.uint8)).convert('RGB')
    im.save(f"{args.output_dir}/mask.png")

    im_np = np.array(pil_im)
    im_np = im_np / im_np.max()
    im_np = mask * im_np
    im_np[mask == 0] = 1
    im_final = (im_np / im_np.max() * 255).astype(np.uint8)
    im_final = Image.fromarray(im_final)

    return im_final, predict
