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
                num_strokes=16,
                num_segments=1,
                imsize=224,
                device=None,
                target_im=None,
                mask=None):
        super(Painter, self).__init__()
        # num_strokes = 16
        # num_segments = 1
        # imsize = 224
        # device = device(type='cuda')

        # (Pdb) target_im.size(), target_im.min(), target_im.max()
        # ([1, 3, 224, 224], 0., 1.)

        # (Pdb) mask.size(), mask.min(), mask.max()
        # ([1, 248, 248], 0., 1.)

        self.num_paths = num_strokes # 16
        self.num_segments = num_segments # 1
        self.stroke_width = args.stroke_width
        self.control_points_per_seg = args.control_points_per_seg # 4
        self.num_stages = args.num_stages # 1
        self.softmax_temp = args.softmax_temp # 0.3

        self.shapes = []
        self.shape_groups = []
        self.device = device
        self.canvas_width, self.canvas_height = imsize, imsize # 224, 224
        self.points_vars = []

        self.strokes_per_stage = self.num_paths # 16
        # self.xdog_intersec = args.xdog_intersec # 1
        
        self.clip_model = args.clip_model # 'ViT-B/32'

        # pdb.set_trace()
        self.define_attention_input(target_im)
        self.mask = mask

        # pdb.set_trace()
        self.attention_map = self.clip_attn() # shape -- (224, 224)
        
        # pdb.set_trace()
        self.thresh = self.set_inds_clip() # shape -- (224, 224)
        self.strokes_counter = 0 # counts the number of calls to "get_path"        
        

    def init_image(self, stage=0):
        for i in range(0, self.num_paths): # self.num_paths -- 16
            stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0])
            path = self.get_path()
            self.shapes.append(path)
            path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(self.shapes) - 1]),
                                             fill_color = None,
                                             stroke_color = stroke_color)
            self.shape_groups.append(path_group)        

        img = self.render_warp()
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, \
            device = self.device) * (1 - img[:, :, 3:4])
        img = img[:, :, :3]

        # Convert img from HWC to NCHW
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2).to(self.device) # NHWC -> NCHW

        # img.size() -- [1, 3, 224, 224]
        return img

    def get_image(self):
        img = self.render_warp() # img.size() -- [224, 224, 4]

        opacity = img[:, :, 3:4]
        img = opacity * img[:, :, :3] + \
            torch.ones(img.shape[0], img.shape[1], 3, device = self.device) * (1 - opacity)
        img = img[:, :, :3]

        # Convert img from HWC to NCHW
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2).to(self.device) # NHWC -> NCHW
        # img.size() -- [1, 3, 224, 224]
        return img

    def get_path(self):
        points = []
        # self.num_segments -- 1
        self.num_control_points = torch.zeros(self.num_segments, dtype = torch.int32) \
            + (self.control_points_per_seg - 2)
        # ==> self.num_control_points -- tensor([2], dtype=torch.int32)

        p0 = self.inds_normalised[self.strokes_counter]
        points.append(p0)

        for j in range(self.num_segments): #  self.num_segments -- 1
            radius = 0.05
            for k in range(self.control_points_per_seg - 1):
                p1 = (p0[0] + radius * (random.random() - 0.5), 
                      p0[1] + radius * (random.random() - 0.5))
                points.append(p1)
                p0 = p1
        points = torch.tensor(points).to(self.device)
        points[:, 0] *= self.canvas_width
        points[:, 1] *= self.canvas_height
        
        path = pydiffvg.Path(num_control_points = self.num_control_points,
                                points = points,
                                stroke_width = torch.tensor(self.stroke_width),
                                is_closed = False)
        self.strokes_counter += 1
        return path

    def render_warp(self):
        _render = pydiffvg.RenderFunction.apply

        scene_args = pydiffvg.RenderFunction.serialize_scene(\
            self.canvas_width, self.canvas_height, self.shapes, self.shape_groups)
        # len(self.shapes) -- 16
        # (Pdb) self.shapes[0]
        # Path : num_control_points=tensor([2], dtype=torch.int32), 
        #     points=[[125.0, 83.0], 
        #             [128.85752868652344, 85.88909149169922], 
        #             [127.96792602539062, 83.18895721435547], 
        #             [128.09420776367188, 82.12421417236328]], 
        #     is_closed=False, stroke_width=1.5, use_distance_approx=False

        # (Pdb) len(self.shape_groups) -- 16
        # (Pdb) self.shape_groups[0]
        # ShapeGroup : shape_ids=[0], use_even_odd_rule=True, 
        #     stroke_color=[0.0, 0.0, 0.0, 1.0], 
        #     shape_to_canvas=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        # ---------------------------------------------------------------------------
        # (Pdb) self.shape_groups[15]
        # ShapeGroup : shape_ids=[15], use_even_odd_rule=True, 
        #     stroke_color=[0.0, 0.0, 0.0, 1.0], 
        #     shape_to_canvas=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

        img = _render(
                    self.canvas_width, # width
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
            path.points.requires_grad = True
            self.points_vars.append(path.points)

        # len(self.points_vars) -- 16
        # self.points_vars[x].size() -- [4, 2]
        return self.points_vars
    

    def save_svg(self, output_dir, name):
        pydiffvg.save_svg('{}/{}.svg'.format(output_dir, name), 
            self.canvas_width, self.canvas_height, self.shapes, self.shape_groups)

    def define_attention_input(self, target_im):
        model, preprocess = clip.load(self.clip_model, device=self.device, jit=False)
        model.eval().to(self.device)
        data_transforms = transforms.Compose([preprocess.transforms[-1], ])
        self.image_input_attn_clip = data_transforms(target_im).to(self.device)
        # target_im.size() -- [1, 3, 224, 224]
        # self.image_input_attn_clip.size() -- [1, 3, 224, 224]
        del model
        # pdb.set_trace()
        

    def clip_attn(self):
        model, preprocess = clip.load(self.clip_model, device=self.device, jit=False)
        model.eval().to(self.device)
        attn_map = interpret(self.image_input_attn_clip, model, device=self.device)
        del model

        # (Pdb) type(attn_map) -- <class 'numpy.ndarray'>, attn_map.shape -- (224, 224)
        # pdb.set_trace()
        return attn_map

        
    def softmax(self, x, tau=0.2):
        e_x = np.exp(x / tau)
        return e_x / e_x.sum() 

    def set_inds_clip(self):
        attn_map = (self.attention_map - self.attention_map.min()) \
            / (self.attention_map.max() - self.attention_map.min())

        # if self.xdog_intersec: # 1
        #     xdog = XDoG_()
        #     im_xdog = xdog(self.image_input_attn_clip[0].permute(1,2,0).cpu().numpy(), k=10)
        #     intersec_map = (1 - im_xdog) * attn_map
        #     attn_map = intersec_map
            
        attn_map_soft = np.copy(attn_map)
        attn_map_soft[attn_map > 0] = self.softmax(attn_map[attn_map > 0], 
            tau=self.softmax_temp)
        
        k = self.num_stages * self.num_paths # self.num_stages ===1, ==> 16

        self.inds = np.random.choice(range(attn_map.flatten().shape[0]), 
            size=k, replace=False, p=attn_map_soft.flatten())
        self.inds = np.array(np.unravel_index(self.inds, attn_map.shape)).T
    
        self.inds_normalised = np.zeros(self.inds.shape)
        self.inds_normalised[:, 0] = self.inds[:, 1] / self.canvas_width
        self.inds_normalised[:, 1] = self.inds[:, 0] / self.canvas_height
        self.inds_normalised = self.inds_normalised.tolist()

        # pdb.set_trace()
        return attn_map_soft

    def get_attention_map(self):
        return self.attention_map
    
    def get_thresh(self):
        return self.thresh

    def get_inds(self):
        return self.inds
    
    def get_mask(self):
        return self.mask


class PainterOptimizer:
    def __init__(self, args, renderer):
        self.renderer = renderer
        self.points_lr = args.lr # 1.0
        self.args = args

    def init_optimizers(self):
        self.points_optim = torch.optim.Adam(self.renderer.parameters(), lr=self.points_lr)
        # (Pdb) for l in self.renderer.parameters(): print(l.size())
        # torch.Size([4, 2])
        # torch.Size([4, 2])
        # torch.Size([4, 2])
        # torch.Size([4, 2])
        # torch.Size([4, 2])
        # torch.Size([4, 2])
        # torch.Size([4, 2])
        # torch.Size([4, 2])
        # torch.Size([4, 2])
        # torch.Size([4, 2])
        # torch.Size([4, 2])
        # torch.Size([4, 2])
        # torch.Size([4, 2])
        # torch.Size([4, 2])
        # torch.Size([4, 2])
        # torch.Size([4, 2])

    def update_lr(self, counter):
        pdb.set_trace()
        new_lr = utils.get_epoch_lr(counter, self.args)
        print("---- update_lr: ", counter, new_lr)
        for param_group in self.points_optim.param_groups:
            param_group["lr"] = new_lr
    
    def zero_grad_(self):
        self.points_optim.zero_grad()
    
    def step_(self):
        self.points_optim.step()
    
    def get_lr(self):
        pdb.set_trace()
        return self.points_optim.param_groups[0]['lr']

def interpret(image, model, device):
    # image.size() -- [1, 3, 224, 224]
    # images = image.repeat(1, 1, 1, 1)
    # res = model.encode_image(images)

    res = model.encode_image(image)
    model.zero_grad()

    image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())

    num_tokens = image_attn_blocks[0].attn_probs.shape[-1] # ==> 50
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
    image_relevance = F.interpolate(image_relevance, size=224, mode='bicubic', antialias=True)
    image_relevance = image_relevance.reshape(224, 224).data.cpu().numpy().astype(np.float32)
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())

    # image_relevance.shape -- (224, 224)
    return image_relevance


class XDoG_(object):
    def __init__(self):
        super(XDoG_, self).__init__()
        self.gamma = 0.98
        self.phi = 200
        self.eps = -0.1
        self.sigma=0.8
        
    def __call__(self, im, k=10):
        # (Pdb) im.shape, im.min(), im.max()
        # ((224, 224, 3), -1.7630657, 2.145897)
        # k = 10
        if im.shape[2] == 3: # True
            im = rgb2gray(im)
            # ==> im.shape -- (224, 224)

        imf1 = gaussian_filter(im, self.sigma)
        imf2 = gaussian_filter(im, self.sigma * k)

        imdiff = imf1 - self.gamma * imf2
        imdiff = (imdiff < self.eps) * 1.0  + \
            (imdiff >= self.eps) * (1.0 + np.tanh(self.phi * imdiff))
        imdiff -= imdiff.min()
        imdiff /= imdiff.max()
        # imdiff.shape -- (224, 224)

        th = threshold_otsu(imdiff) # 0.701171875

        imdiff = imdiff >= th
        imdiff = imdiff.astype('float32')

        # (Pdb) imdiff.shape, imdiff.min(), imdiff.max()
        # ((224, 224), 0.0, 1.0)
        return imdiff
