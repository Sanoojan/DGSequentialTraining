import colorsys
import os
import sys

import numpy as np
import torch
import torchvision
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Polygon
from skimage.measure import find_contours
from torch.nn.functional import interpolate
from tqdm import tqdm

from domainbed.lib.utils import get_voc_dataset, get_model, parse_args
from domainbed.lib.visiontransformer import VisionTransformer
import domainbed.algorithms as algorithms
from domainbed.lib import misc


def get_attention_masks(args, image, model):
    # make the image divisible by the patch size
    w, h = image.shape[2] - image.shape[2] % args.patch_size, image.shape[3] - image.shape[3] % args.patch_size
    image = image[:, :w, :h]
    w_featmap = image.shape[-2] // args.patch_size
    h_featmap = image.shape[-1] // args.patch_size
    

    # attentions=model.get_last_selfattention(image.cuda())
    attentions = model.forward_selfattention(image.cuda())
    nh = attentions.shape[1]

    # we keep only the output patch attention
    if args.is_dist:
        if args.use_shape:
            attentions = attentions[0, :, 1, 2:].reshape(nh, -1)  # use distillation token attention
        else:
            attentions = attentions[0, :, 0, 2:].reshape(nh, -1)  # use class token attention
    else:
        attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

    # we keep only a certain percentage of the mass
    val, idx = torch.sort(attentions)
    val /= torch.sum(val, dim=1, keepdim=True)
    cum_val = torch.cumsum(val, dim=1)
    th_attn = cum_val > (1 - args.threshold)
    idx2 = torch.argsort(idx)
    for head in range(nh):
        th_attn[head] = th_attn[head][idx2[head]]
    th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
    # interpolate
    th_attn = interpolate(th_attn.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0]

    return th_attn

def get_all_attention_masks(args, image, model):
    # make the image divisible by the patch size
    w, h = image.shape[2] - image.shape[2] % args.patch_size, image.shape[3] - image.shape[3] % args.patch_size
    image = image[:, :w, :h]
    w_featmap = image.shape[-2] // args.patch_size
    h_featmap = image.shape[-1] // args.patch_size
    th_attn_all=[]

    # attentions=model.get_last_selfattention(image.cuda())
    model.train()
    _ , attentions_all = model.forward(image.cuda(),return_attention=True) # change here
    for attentions in attentions_all:
        nh = attentions.shape[1]

        # we keep only the output patch attention
        # print("attentions.shape:",attentions.shape)
        if args.is_dist:
            if args.use_shape:
                attentions = attentions[0, :, 1, 2:].reshape(nh, -1)  # use distillation token attention
            else:
                attentions = attentions[0, :, 0, 2:].reshape(nh, -1)  # use class token attention
        else:
            attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

        # we keep only a certain percentage of the mass
        val, idx = torch.sort(attentions)
        val /= torch.sum(val, dim=1, keepdim=True)
        cum_val = torch.cumsum(val, dim=1)
        th_attn = cum_val > (1 - args.threshold)
        idx2 = torch.argsort(idx)
        for head in range(nh):
            th_attn[head] = th_attn[head][idx2[head]]
        th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
        # interpolate
        th_attn = interpolate(th_attn.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0]
        th_attn_all.append(th_attn)
    return th_attn_all


def get_per_sample_jaccard(pred,target):
    jac = 0
    object_count = 0
    for mask_idx in torch.unique(target):
        if mask_idx in [0, 255]:  # ignore index
            continue
        cur_mask = target == mask_idx
        intersection = (cur_mask * pred) * (cur_mask != 255)  # handle void labels
        intersection = torch.sum(intersection, dim=[1, 2])  # handle void labels
        union = ((cur_mask + pred) > 0) * (cur_mask != 255)
        union = torch.sum(union, dim=[1, 2])
        jac_all = intersection / union
        jac += jac_all.max().item()
        object_count += 1
    return jac / object_count


def run_eval(args, data_loader, model, device):
    model.to(device)
    model.eval()
    total_jac = 0
    image_count = 0
    for idx, (sample, target) in tqdm(enumerate(data_loader), total=len(data_loader)):
        sample, target = sample.to(device), target.to(device)
        attention_mask = get_attention_masks(args, sample, model)
        jac_val = get_per_sample_jaccard(attention_mask, target)
        total_jac += jac_val
        image_count += 1
    return total_jac / image_count

def run_eval_self(args, model, device):
    model.to(device)
    model.eval()

    samples = []
    for fol_name in tqdm(os.listdir(args.test_dir)):
        cnt=0
        for im_name in tqdm(os.listdir(args.test_dir+"/"+fol_name)):
            # if(cnt>20):
            #     break
            # cnt+=1
            im_path = f"{args.test_dir}/{fol_name}/{im_name}"
            img = Image.open(f"{im_path}").resize((224, 224))
            img = torchvision.transforms.functional.to_tensor(img)
            if img.shape[0] == 1:
                img = torch.cat([img, img, img], dim=0)
            samples.append(img)
    samples = torch.stack(samples, 0).to(device)

    total_jac = [0]*11
    image_count = 0
    jac_values=[]
    for sample in samples:
       
        all_attention_mask = get_all_attention_masks(args, sample.unsqueeze(0), model)
        # print(len(all_attention_mask))
        for att_i in range(len(all_attention_mask)-1):

            jac_val = get_per_sample_jaccard(all_attention_mask[att_i], all_attention_mask[-1])
            jac_values.append(jac_val)
            # print(total_jac)
            # print(att_i)
            total_jac[att_i]=total_jac[att_i]+jac_val
        image_count+=1
    return [total_jac_i / image_count for total_jac_i in total_jac]  
        # attention_masks.append(get_attention_masks(args, sample.unsqueeze(0), model))


    
    for idx, (sample, target) in tqdm(enumerate(data_loader), total=len(data_loader)):
        sample, target = sample.to(device), target.to(device)
        attention_mask = get_attention_masks(args, sample, model)
        jac_val = get_per_sample_jaccard(attention_mask, target)
        total_jac += jac_val
        image_count += 1
    return total_jac / image_count


def apply_mask_last(image, mask, color=(0.0, 0.0, 1.0), alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image


def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5,batch=False):
    image = image.permute(1, 2, 0).cpu().numpy()
    mask = mask.cpu().numpy()

    plt.ioff()
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]

    # Generate random colors

    def random_colors(N, bright=True):
        """
        Generate random colors.
        """
        brightness = 1.0 if bright else 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        return colors

    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = (image * 255).astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            pass
            # _mask = cv2.blur(_mask, (10, 10))
        # Mask
        masked_image = apply_mask_last(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    if(batch):
        plt.close(fig)
        return masked_image
    else:
        ax.imshow(masked_image.astype(np.uint8), aspect='auto')
        fig.savefig(fname)
        plt.close(fig)
    


def generate_images_per_model(args, model, device):

    model.to(device)
    model.eval()

    samples = []
    for fol_name in tqdm(os.listdir(args.test_dir)):
        cnt=0
        for im_name in tqdm(os.listdir(args.test_dir+"/"+fol_name)):
            if(cnt>15):
                break
            cnt+=1
            im_path = f"{args.test_dir}/{fol_name}/{im_name}"
            img = Image.open(f"{im_path}").resize((224, 224))
            img = torchvision.transforms.functional.to_tensor(img)
            if img.shape[0] == 1:
                img = torch.cat([img, img, img], dim=0)
            samples.append(img)
    samples = torch.stack(samples, 0).to(device)

    attention_masks = []
    for sample in samples:
        attention_masks.append(get_attention_masks(args, sample.unsqueeze(0), model))
    
    os.makedirs(f"{args.save_path}", exist_ok=True)
    os.makedirs(f"{args.save_path}/{args.model_name}_{args.threshold}", exist_ok=True)
    for idx, (sample, mask) in enumerate(zip(samples, attention_masks)):
        for head_idx, mask_h in enumerate(mask):
            f_name = f"{args.save_path}/{args.model_name}_{args.threshold}/im_{idx:03d}_{head_idx}.png"
            display_instances(sample, mask_h, fname=f_name)


def generate_images_per_model_per_block(args, model, device):

    model.to(device)
    model.eval()

    samples = []
    for fol_name in tqdm(os.listdir(args.test_dir)):
        cnt=0
        for im_name in tqdm(os.listdir(args.test_dir+"/"+fol_name)):
            if(cnt>15):
                break
            cnt+=1
            im_path = f"{args.test_dir}/{fol_name}/{im_name}"
            img = Image.open(f"{im_path}").resize((224, 224))
            img = torchvision.transforms.functional.to_tensor(img)
            if img.shape[0] == 1:
                img = torch.cat([img, img, img], dim=0)
            samples.append(img)
    samples = torch.stack(samples, 0).to(device)

    attention_masks = []
    for sample in samples:
        attention_masks.append(get_all_attention_masks(args, sample.unsqueeze(0), model))
    
    for i in range(len(attention_masks[0])):
        os.makedirs(f"{args.save_path}", exist_ok=True)
        os.makedirs(f"{args.save_path}/{args.model_name}_{args.threshold}", exist_ok=True)
        for idx, (sample, mask) in enumerate(zip(samples, attention_masks)):
            for head_idx, mask_h in enumerate(mask[i]):
                f_name = f"{args.save_path}/{args.model_name}_{args.threshold}/im_{idx:03d}_{head_idx}_blk{i}.png"
                display_instances(sample, mask_h, fname=f_name)

def generate_images_per_model_full(args, model, device):

    model.to(device)
    model.eval()

    samples = []
    for fol_name in tqdm(os.listdir(args.test_dir)):
        cnt=0
        for im_name in tqdm(os.listdir(args.test_dir+"/"+fol_name)):
            if(cnt>15):
                break
            cnt+=1
            im_path = f"{args.test_dir}/{fol_name}/{im_name}"
            img = Image.open(f"{im_path}").resize((224, 224))
            img = torchvision.transforms.functional.to_tensor(img)
            if img.shape[0] == 1:
                img = torch.cat([img, img, img], dim=0)
            samples.append(img)
    samples = torch.stack(samples, 0).to(device)

    attention_masks = []
    for sample in samples:
        attention_masks.append(get_all_attention_masks(args, sample.unsqueeze(0), model))

    
    
    os.makedirs(f"{args.save_path}", exist_ok=True)
    os.makedirs(f"{args.save_path}/batched/{args.model_name}_{args.threshold}", exist_ok=True)
    for idx, (sample, mask) in enumerate(zip(samples, attention_masks)):
        
        f_name = f"{args.save_path}/batched/{args.model_name}_{args.threshold}/im_{idx:03d}all.png"
        figsize=(5, 5)
        # plt.ioff()
        # plt.figure()
        plt.figure(figsize=(224,224),frameon=False)
        gs1 = gridspec.GridSpec(len(mask[0]),len(attention_masks[0]))
        gs1.update(wspace=0, hspace=0) # set the spacing between axes. 

        for i in range(len(attention_masks[0])):

            
            for head_idx, mask_h in enumerate(mask[i]):

                img = display_instances(sample, mask_h, fname=f_name,batch=True)
                pos=i+head_idx*12
                # axs[i,head_idx].set_axis_off()
                # fig.add_subplot(rows, columns, pos)
                plt.subplot(gs1[pos])
                plt.imshow(img.astype(np.uint8),aspect="equal")
                
                # img.close(fig)
        # plt.subplots_adjust(wspace=0, hspace=0)
        plt.tight_layout()
        plt.savefig(f_name)
        plt.close()
    
    # fig = plt.figure(figsize=(8, 8))
    # columns = len(attention_masks[0])
    # rows = len(attention_masks[0][0])
    # for i in range(1, columns*rows +1):
    #     img = np.random.randint(10, size=(h,w))
    #     fig.add_subplot(rows, columns, i)
    #     plt.imshow(img)
    # plt.show()

if __name__ == '__main__':
    opt = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    # test_dataset, test_data_loader = get_voc_dataset()
    test_data_loader=None
    # data_transform = transforms.Compose([
    #     transforms.Resize((512, 512)),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    # ])

    # def load_target(image):
    #     image = np.array(image)
    #     image = torch.from_numpy(image)
    #     return image

    # target_transform = transforms.Compose([
    #     transforms.Resize((512, 512)),
    #     transforms.Lambda(load_target),
    # ])

    # transform = transforms.Compose([
    #         transforms.Resize((224,224)),
    #         transforms.ToTensor(),
    #         transforms.Normalize(
    #             mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #     ])

    # dataset = torchvision.datasets.VOCSegmentation(root=opt.test_dir, image_set="val", transform=data_transform,
    #                                                target_transform=target_transform)
    # test_data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, drop_last=False)
    environments = [f.name for f in os.scandir(opt.test_dir) if f.is_dir()]
    environments = sorted(environments)
    domain=environments[int(opt.domain)]
    if opt.domain is not None:
        opt.save_path=opt.save_path+"/"+domain
        opt.test_dir=opt.test_dir+"/"+domain
    
    print("test_dir:",opt.test_dir)
    opt.is_dist = "Dist" in opt.model_name
    if not (opt.is_dist):
        opt.use_shape=False
    if opt.use_shape:
        assert opt.is_dist, "shape token only present in distilled models"

    if opt.rand_init:
        model, mean, std = get_model(opt, pretrained=False)
    else:
        model, mean, std = get_model(opt)
        if opt.pretrained_weights.startswith("https://"):
            state_dict = torch.hub.load_state_dict_from_url(url=opt.pretrained_weights, map_location="cpu")
        # else:
        #     # state_dict = torch.load(opt.pretrained_weights, map_location="cpu")
        # # msg = model.load_state_dict(state_dict["model"], strict=False)
        # # print(msg)
    


    if opt.generate_images:
        generate_images_per_model(opt, model, device)
    elif opt.generate_images_blockwise:
        generate_images_per_model_per_block(opt, model, device)
    elif opt.generate_images_block_asbatch:
        os.makedirs(opt.save_path, exist_ok=True)
        generate_images_per_model_full(opt, model, device)
    else:
        model_accuracy = run_eval_self(opt, model, device)
        os.makedirs(opt.jacard_out, exist_ok=True)
    
        sys.stdout = misc.Tee(os.path.join(opt.jacard_out, 'out.txt'))
        print(f"Jaccard index for {opt.model_name}:{opt.domain}: {model_accuracy}")