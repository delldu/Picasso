import random
import CLIP_.clip as clip
import numpy as np
import pydiffvg
import sketch_utils as utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from torchvision import transforms
import pdb

class Painter(nn.Module):
    def __init__(self, args,
                num_strokes=4,
                num_segments=4,
                imsize=224,
                device=None,
                target_im=None,
                mask=None):
        super(Painter, self).__init__()
        # num_strokes = 16
        # num_segments = 1
        # imsize = 224

        # (Pdb) target_im.size(), target_im.min(), target_im.max()
        # ([1, 3, 224, 224], 0., 1.)

        # (Pdb) mask.size(), mask.min(), mask.max()
        # ([1, 248, 248], 0., 1.)

        self.args = args
        self.num_paths = num_strokes
        self.num_segments = num_segments
        self.width = args.width
        self.control_points_per_seg = args.control_points_per_seg # 4
        # self.opacity_optim = args.force_sparse # 0
        self.num_stages = args.num_stages # 1
        self.add_random_noise = "noise" in args.augemntations # False
        self.noise_thresh = args.noise_thresh # 0.5
        self.softmax_temp = args.softmax_temp # 0.3

        self.shapes = []
        self.shape_groups = []
        self.device = device
        self.canvas_width, self.canvas_height = imsize, imsize # 224, 224
        self.points_vars = []
        self.color_vars = []
        self.color_vars_threshold = args.color_vars_threshold # 0.0

        self.path_svg = args.path_svg # 'none'
        self.strokes_per_stage = self.num_paths # 16
        self.optimize_flag = []

        # attention related for strokes initialisation
        self.attention_init = args.attention_init # 1
        self.target_path = args.target # 'target_images/camel.png'
        self.saliency_model = args.saliency_model # 'clip'
        self.xdog_intersec = args.xdog_intersec # 1
        
        self.text_target = args.text_target # for clip gradients, 'none'
        self.saliency_clip_model = args.saliency_clip_model # 'ViT-B/32'
        self.define_attention_input(target_im)
        self.mask = mask
        self.attention_map = self.set_attention_map() # shape -- (224, 224)
        
        self.thresh = self.set_attention_threshold_map() # shape -- (224, 224)
        self.strokes_counter = 0 # counts the number of calls to "get_path"        
        self.epoch = 0
        self.final_epoch = args.num_iter - 1 # ==> 2000
        

    def init_image(self, stage=0):
        if stage > 0:
            pdb.set_trace()
            # if multi stages training than add new strokes on existing ones
            # don't optimize on previous strokes
            self.optimize_flag = [False for i in range(len(self.shapes))]
            for i in range(self.strokes_per_stage):
                stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0])
                path = self.get_path()
                self.shapes.append(path)
                path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(self.shapes) - 1]),
                                                    fill_color = None,
                                                    stroke_color = stroke_color)
                self.shape_groups.append(path_group)
                self.optimize_flag.append(True)

        else: # True
            num_paths_exists = 0
            if self.path_svg != "none": # False
                self.canvas_width, self.canvas_height, self.shapes, self.shape_groups = utils.load_svg(self.path_svg)
                # if you want to add more strokes to existing ones and optimize on all of them
                num_paths_exists = len(self.shapes)

            for i in range(num_paths_exists, self.num_paths):
                stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0])
                path = self.get_path()
                self.shapes.append(path)
                path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(self.shapes) - 1]),
                                                    fill_color = None,
                                                    stroke_color = stroke_color)
                self.shape_groups.append(path_group)        
            self.optimize_flag = [True for i in range(len(self.shapes))]
        
        img = self.render_warp()
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = self.device) * (1 - img[:, :, 3:4])
        img = img[:, :, :3]
        # Convert img from HWC to NCHW
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2).to(self.device) # NHWC -> NCHW

        return img

    def get_image(self):
        img = self.render_warp()
        opacity = img[:, :, 3:4]
        img = opacity * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = self.device) * (1 - opacity)
        img = img[:, :, :3]
        # Convert img from HWC to NCHW
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2).to(self.device) # NHWC -> NCHW
        return img

    def get_path(self):
        points = []
        self.num_control_points = torch.zeros(self.num_segments, dtype = torch.int32) + (self.control_points_per_seg - 2)
        # self.num_control_points -- tensor([2], dtype=torch.int32)

        p0 = self.inds_normalised[self.strokes_counter] if self.attention_init else (random.random(), random.random())
        points.append(p0)

        for j in range(self.num_segments): #  self.num_segments -- 1
            radius = 0.05
            for k in range(self.control_points_per_seg - 1):
                p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
                points.append(p1)
                p0 = p1
        points = torch.tensor(points).to(self.device)
        points[:, 0] *= self.canvas_width
        points[:, 1] *= self.canvas_height
        
        path = pydiffvg.Path(num_control_points = self.num_control_points,
                                points = points,
                                stroke_width = torch.tensor(self.width),
                                is_closed = False)
        self.strokes_counter += 1
        return path

    def render_warp(self):
        # if self.opacity_optim: # 0
        #     for group in self.shape_groups:
        #         group.stroke_color.data[:3].clamp_(0., 0.) # to force black stroke
        #         group.stroke_color.data[-1].clamp_(0., 1.) # opacity
        _render = pydiffvg.RenderFunction.apply

        # uncomment if you want to add random noise
        if self.add_random_noise: # False
            pdb.set_trace()

            if random.random() > self.noise_thresh:
                eps = 0.01 * min(self.canvas_width, self.canvas_height)
                for path in self.shapes:
                    path.points.data.add_(eps * torch.randn_like(path.points))
        scene_args = pydiffvg.RenderFunction.serialize_scene(\
            self.canvas_width, self.canvas_height, self.shapes, self.shape_groups)
        img = _render(self.canvas_width, # width
                    self.canvas_height, # height
                    2,   # num_samples_x
                    2,   # num_samples_y
                    0,   # seed
                    None,
                    *scene_args)
        return img
    
    def parameters(self):
        self.points_vars = []
        # storkes' location optimization
        for i, path in enumerate(self.shapes):
            if self.optimize_flag[i]:
                path.points.requires_grad = True
                self.points_vars.append(path.points)
        return self.points_vars
    
    def get_points_parans(self):
        return self.points_vars
    
    def get_color_parameters(self):
        return self.color_vars
        
    def save_svg(self, output_dir, name):
        pydiffvg.save_svg('{}/{}.svg'.format(output_dir, name), self.canvas_width, self.canvas_height, self.shapes, self.shape_groups)

    def define_attention_input(self, target_im):
        model, preprocess = clip.load(self.saliency_clip_model, device=self.device, jit=False)
        model.eval().to(self.device)
        data_transforms = transforms.Compose([preprocess.transforms[-1], ])
        self.image_input_attn_clip = data_transforms(target_im).to(self.device)
        

    def clip_attn(self):
        model, preprocess = clip.load(self.saliency_clip_model, device=self.device, jit=False)
        model.eval().to(self.device)
        text_input = clip.tokenize([self.text_target]).to(self.device)

        if "RN" in self.saliency_clip_model: # False
            saliency_layer = "layer4"
            attn_map = gradCAM(
                model.visual,
                self.image_input_attn_clip,
                model.encode_text(text_input).float(),
                getattr(model.visual, saliency_layer)
            )
            attn_map = attn_map.squeeze().detach().cpu().numpy()
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())

        else: # True
            attn_map = interpret(self.image_input_attn_clip, text_input, model, device=self.device)
        
        del model

        return attn_map

    def set_attention_map(self):
        assert self.saliency_model in ["dino", "clip"]
        return self.clip_attn()
        
    def softmax(self, x, tau=0.2):
        e_x = np.exp(x / tau)
        return e_x / e_x.sum() 

    def set_inds_clip(self):
        attn_map = (self.attention_map - self.attention_map.min()) / (self.attention_map.max() - self.attention_map.min())
        if self.xdog_intersec: # 1
            xdog = XDoG_()
            im_xdog = xdog(self.image_input_attn_clip[0].permute(1,2,0).cpu().numpy(), k=10)
            intersec_map = (1 - im_xdog) * attn_map
            attn_map = intersec_map
            
        attn_map_soft = np.copy(attn_map)
        attn_map_soft[attn_map > 0] = self.softmax(attn_map[attn_map > 0], 
            tau=self.softmax_temp)
        
        k = self.num_stages * self.num_paths
        self.inds = np.random.choice(range(attn_map.flatten().shape[0]), 
            size=k, replace=False, p=attn_map_soft.flatten())
        self.inds = np.array(np.unravel_index(self.inds, attn_map.shape)).T
    
        self.inds_normalised = np.zeros(self.inds.shape)
        self.inds_normalised[:, 0] = self.inds[:, 1] / self.canvas_width
        self.inds_normalised[:, 1] = self.inds[:, 0] / self.canvas_height
        self.inds_normalised = self.inds_normalised.tolist()
        return attn_map_soft

    def set_attention_threshold_map(self):
        assert self.saliency_model in ["dino", "clip"]
        return self.set_inds_clip()

    def get_attn(self):
        return self.attention_map
    
    def get_thresh(self):
        return self.thresh

    def get_inds(self):
        return self.inds
    
    def get_mask(self):
        return self.mask

    def set_random_noise(self, epoch):
        if epoch % self.args.save_interval == 0:
            self.add_random_noise = False
        else:
            self.add_random_noise = "noise" in self.args.augemntations # False

class PainterOptimizer:
    def __init__(self, args, renderer):
        self.renderer = renderer
        self.points_lr = args.lr
        # self.color_lr = args.color_lr
        self.args = args
        # self.optim_color = args.force_sparse

    def init_optimizers(self):
        self.points_optim = torch.optim.Adam(self.renderer.parameters(), lr=self.points_lr)

    def update_lr(self, counter):
        new_lr = utils.get_epoch_lr(counter, self.args)
        for param_group in self.points_optim.param_groups:
            param_group["lr"] = new_lr
    
    def zero_grad_(self):
        self.points_optim.zero_grad()
    
    def step_(self):
        self.points_optim.step()
    
    def get_lr(self):
        return self.points_optim.param_groups[0]['lr']


# class Hook:
#     """Attaches to a module and records its activations and gradients."""

#     def __init__(self, module: nn.Module):
#         self.data = None
#         self.hook = module.register_forward_hook(self.save_grad)
        
#     def save_grad(self, module, input, output):
#         self.data = output
#         output.requires_grad_(True)
#         output.retain_grad()
        
#     def __enter__(self):
#         return self
    
#     def __exit__(self, exc_type, exc_value, exc_traceback):
#         self.hook.remove()
        
#     @property
#     def activation(self) -> torch.Tensor:
#         return self.data
    
#     @property
#     def gradient(self) -> torch.Tensor:
#         return self.data.grad


def interpret(image, texts, model, device):
    # image.size() -- [1, 3, 224, 224]
    # texts.size() -- [1, 77]
    # texts = tensor([[49406,  8906, 49407,     0,     0,     0,     0,     0,     0,     0,
    #              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
    #              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
    #              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
    #              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
    #              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
    #              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
    #              0,     0,     0,     0,     0,     0,     0]], device='cuda:0')

    images = image.repeat(1, 1, 1, 1)
    res = model.encode_image(images)
    model.zero_grad()

    image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())
    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
    R = R.unsqueeze(0).expand(1, num_tokens, num_tokens)
    cams = [] # there are 12 attention blocks
    for i, blk in enumerate(image_attn_blocks):
        cam = blk.attn_probs.detach() #attn_probs shape is 12, 50, 50
        # each patch is 7x7 so we have 49 pixels + 1 for positional encoding
        cam = cam.reshape(1, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0)
        cam = cam.clamp(min=0).mean(dim=1) # mean of the 12 something
        cams.append(cam)  
        R = R + torch.bmm(cam, R)
              
    cams_avg = torch.cat(cams) # 12, 50, 50
    cams_avg = cams_avg[:, 0, 1:] # 12, 1, 49
    image_relevance = cams_avg.mean(dim=0).unsqueeze(0)
    image_relevance = image_relevance.reshape(1, 1, 7, 7)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bicubic')
    image_relevance = image_relevance.reshape(224, 224).data.cpu().numpy().astype(np.float32)
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())

    # image_relevance.shape -- (224, 224)
    return image_relevance


# # Reference: https://arxiv.org/abs/1610.02391
# def gradCAM(
#     model: nn.Module,
#     input: torch.Tensor,
#     target: torch.Tensor,
#     layer: nn.Module
# ) -> torch.Tensor:
#     pdb.set_trace()

#     # Zero out any gradients at the input.
#     if input.grad is not None:
#         input.grad.data.zero_()
        
#     # Disable gradient settings.
#     requires_grad = {}
#     for name, param in model.named_parameters():
#         requires_grad[name] = param.requires_grad
#         param.requires_grad_(False)
        
#     # Attach a hook to the model at the desired layer.
#     assert isinstance(layer, nn.Module)
#     pdb.set_trace()

#     with Hook(layer) as hook:        
#         # Do a forward and backward pass.
#         output = model(input)
#         output.backward(target)

#         grad = hook.gradient.float()
#         act = hook.activation.float()
    
#         # Global average pool gradient across spatial dimension
#         # to obtain importance weights.
#         alpha = grad.mean(dim=(2, 3), keepdim=True)
#         # Weighted combination of activation maps over channel
#         # dimension.
#         gradcam = torch.sum(act * alpha, dim=1, keepdim=True)
#         # We only want neurons with positive influence so we
#         # clamp any negative ones.
#         gradcam = torch.clamp(gradcam, min=0)

#     # Resize gradcam to input resolution.
#     gradcam = F.interpolate(
#         gradcam,
#         input.shape[2:],
#         mode='bicubic',
#         align_corners=False)
    
#     # Restore gradient settings.
#     for name, param in model.named_parameters():
#         param.requires_grad_(requires_grad[name])
        
#     return gradcam    


class XDoG_(object):
    def __init__(self):
        super(XDoG_, self).__init__()
        self.gamma=0.98
        self.phi=200
        self.eps=-0.1
        self.sigma=0.8
        self.binarize=True
        
    def __call__(self, im, k=10):
        if im.shape[2] == 3:
            im = rgb2gray(im)
        imf1 = gaussian_filter(im, self.sigma)
        imf2 = gaussian_filter(im, self.sigma * k)
        imdiff = imf1 - self.gamma * imf2
        imdiff = (imdiff < self.eps) * 1.0  + (imdiff >= self.eps) * (1.0 + np.tanh(self.phi * imdiff))
        imdiff -= imdiff.min()
        imdiff /= imdiff.max()
        if self.binarize:
            th = threshold_otsu(imdiff)
            imdiff = imdiff >= th
        imdiff = imdiff.astype('float32')
        return imdiff
