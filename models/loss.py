
import CLIP_.clip as clip
import torch
import torch.nn as nn
from torchvision import models, transforms
import pdb

class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()
        self.args = args
        self.clip_conv_loss = args.clip_conv_loss # 1
        self.clip_fc_loss_weight = args.clip_fc_loss_weight # 0.1
        self.losses_to_apply = self.get_losses_to_apply() # ['clip_conv_loss']
        # pdb.set_trace()

        self.loss_mapper = \
            {
                "clip": CLIPLoss(args),
                "clip_conv_loss": CLIPConvLoss(args)
            }


    def get_losses_to_apply(self):
        losses_to_apply = []
        losses_to_apply.append("clip_conv_loss")
        return losses_to_apply


    def forward(self, sketches, targets, epoch, mode="train"):
        loss = 0

        losses_dict = dict.fromkeys(self.losses_to_apply, torch.tensor([0.0]).to(self.args.device))
        loss_coeffs = dict.fromkeys(self.losses_to_apply, 1.0)

        for loss_name in self.losses_to_apply:
            if loss_name in ["clip_conv_loss"]:
                # pdb.set_trace()
                conv_loss = self.loss_mapper[loss_name](sketches, targets, mode)

                for layer in conv_loss.keys():
                    losses_dict[layer] = conv_loss[layer]
            elif loss_name == "l2":
                losses_dict[loss_name] = self.loss_mapper[loss_name](sketches, targets).mean()
            else:
                losses_dict[loss_name] = self.loss_mapper[loss_name](sketches, targets, mode).mean()

        for key in self.losses_to_apply:
            losses_dict[key] = losses_dict[key] * loss_coeffs[key]
        return losses_dict


class CLIPLoss(nn.Module):
    def __init__(self, args):
        super(CLIPLoss, self).__init__()

        self.args = args
        self.model, clip_preprocess = clip.load('ViT-B/32', args.device, jit=False)
        self.model.eval()
        self.preprocess = transforms.Compose([clip_preprocess.transforms[-1]])  # clip normalisation
        self.device = args.device
        self.NUM_AUGS = args.num_aug_clip
        augemntations = []
        if "affine" in args.augemntations: # True
            augemntations.append(transforms.RandomPerspective(
                fill=0, p=1.0, distortion_scale=0.5))
            augemntations.append(transforms.RandomResizedCrop(
                224, scale=(0.8, 0.8), ratio=(1.0, 1.0)))
        augemntations.append(
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)))
        self.augment_trans = transforms.Compose(augemntations)
        # pdb.set_trace()

        self.calc_target = True
        self.counter = 0

    def forward(self, sketches, targets, mode="train"):
        if self.calc_target:
            targets_ = self.preprocess(targets).to(self.device)
            self.targets_features = self.model.encode_image(targets_).detach()
            self.calc_target = False
            pdb.set_trace()

        if mode == "eval":
            # pdb.set_trace()
            # for regular clip distance, no augmentations
            with torch.no_grad():
                sketches = self.preprocess(sketches).to(self.device)
                sketches_features = self.model.encode_image(sketches)
                return 1. - torch.cosine_similarity(sketches_features, self.targets_features)

        loss_clip = 0
        sketch_augs = []
        img_augs = []
        for n in range(self.NUM_AUGS):
            augmented_pair = self.augment_trans(torch.cat([sketches, targets]))
            sketch_augs.append(augmented_pair[0].unsqueeze(0))

        sketch_batch = torch.cat(sketch_augs)
        sketch_features = self.model.encode_image(sketch_batch)

        for n in range(self.NUM_AUGS):
            loss_clip += (1. - torch.cosine_similarity(
                sketch_features[n:n+1], self.targets_features, dim=1))
        self.counter += 1
        return loss_clip


def l2_layers(xs_conv_features, ys_conv_features, clip_model_name):
    return [torch.square(x_conv - y_conv).mean() for x_conv, y_conv in
            zip(xs_conv_features, ys_conv_features)]


def l1_layers(xs_conv_features, ys_conv_features, clip_model_name):
    return [torch.abs(x_conv - y_conv).mean() for x_conv, y_conv in
            zip(xs_conv_features, ys_conv_features)]


def cos_layers(xs_conv_features, ys_conv_features, clip_model_name):
    if "RN" in clip_model_name:
        return [torch.square(x_conv, y_conv, dim=1).mean() for x_conv, y_conv in
                zip(xs_conv_features, ys_conv_features)]
    return [(1 - torch.cosine_similarity(x_conv, y_conv, dim=1)).mean() for x_conv, y_conv in
            zip(xs_conv_features, ys_conv_features)]


class CLIPConvLoss(nn.Module):
    def __init__(self, args):
        super(CLIPConvLoss, self).__init__()
        self.clip_model_name = args.clip_model_name
        assert self.clip_model_name in [
            "RN50",
            "RN101", # True
            "RN50x4",
            "RN50x16",
            "ViT-B/32",
            "ViT-B/16",
        ]

        self.clip_fc_loss_type = "Cos"  # args.clip_fc_loss_type
        self.distance_metrics = {
                "L2": l2_layers,
                "L1": l1_layers,
                "Cos": cos_layers
            }

        self.model, clip_preprocess = clip.load(self.clip_model_name, args.device, jit=False)

        self.visual_model = self.model.visual
        layers = list(self.model.visual.children())
        init_layers = torch.nn.Sequential(*layers)[:8]
        self.layer1 = layers[8]
        self.layer2 = layers[9]
        self.layer3 = layers[10]
        self.layer4 = layers[11]
        self.att_pool2d = layers[12]

        self.args = args

        # self.img_size = clip_preprocess.transforms[1].size
        self.model.eval()
        self.target_transform = transforms.Compose([transforms.ToTensor()])  # clip normalisation
        self.normalize_transform = transforms.Compose([
            clip_preprocess.transforms[0],  # Resize
            clip_preprocess.transforms[1],  # CenterCrop
            clip_preprocess.transforms[-1],  # Normalize
        ])

        self.model.eval()
        self.device = args.device
        self.num_augs = self.args.num_aug_clip

        augemntations = []
        if "affine" in args.augemntations: # True
            augemntations.append(transforms.RandomPerspective(fill=0, p=1.0, distortion_scale=0.5))
            augemntations.append(transforms.RandomResizedCrop(224, scale=(0.8, 0.8), ratio=(1.0, 1.0)))
        augemntations.append(
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)))
        self.augment_trans = transforms.Compose(augemntations)
        # pdb.set_trace()

        self.clip_fc_loss_weight = args.clip_fc_loss_weight
        self.counter = 0


    def forward(self, sketch, target, mode="train"):
        """
        Parameters
        ----------
        sketch: Torch Tensor [1, C, H, W]
        target: Torch Tensor [1, C, H, W]
        """
        # print("----- CLIPConvLoss forward mode: ", mode) # ==> 'train'
        conv_loss_dict = {}
        x = sketch.to(self.device)
        y = target.to(self.device)
        sketch_augs, img_augs = [self.normalize_transform(x)], [self.normalize_transform(y)]
        if mode == "train": # True
            for n in range(self.num_augs):
                augmented_pair = self.augment_trans(torch.cat([x, y]))
                sketch_augs.append(augmented_pair[0].unsqueeze(0))
                img_augs.append(augmented_pair[1].unsqueeze(0))

        xs = torch.cat(sketch_augs, dim=0).to(self.device)
        ys = torch.cat(img_augs, dim=0).to(self.device)

        xs_fc_features, xs_conv_features = self.forward_inspection_clip_resnet(xs.contiguous())
        ys_fc_features, ys_conv_features = self.forward_inspection_clip_resnet(ys.detach())

        conv_loss = self.distance_metrics['L2'](xs_conv_features, ys_conv_features, \
            self.clip_model_name)

        # self.args.clip_conv_layer_weights -- [0.0, 0.0, 1.0, 1.0, 0.0]
        for layer, w in enumerate(self.args.clip_conv_layer_weights):
            if w: # w == 0.0, 1.0 ...
                conv_loss_dict[f"clip_conv_loss_layer{layer}"] = conv_loss[layer] * w

        if self.clip_fc_loss_weight:
            # fc distance is always cos
            fc_loss = (1 - torch.cosine_similarity(xs_fc_features, ys_fc_features, dim=1)).mean()
            conv_loss_dict["fc"] = fc_loss * self.clip_fc_loss_weight

        self.counter += 1
        return conv_loss_dict

    def forward_inspection_clip_resnet(self, x):
        def stem(m, x):
            for conv, bn in [(m.conv1, m.bn1), (m.conv2, m.bn2), (m.conv3, m.bn3)]:
                x = m.relu(bn(conv(x)))
            x = m.avgpool(x)
            return x

        x = x.type(self.visual_model.conv1.weight.dtype)
        x = stem(self.visual_model, x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        y = self.att_pool2d(x4)
        return y, [x, x1, x2, x3, x4]
        