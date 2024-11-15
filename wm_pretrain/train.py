import sys
sys.path.append("../")
import os
import random
from pathlib import Path
import glob
import PIL
from PIL import Image
from tqdm.auto import tqdm
import numpy as np
import argparse
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torch.utils.checkpoint
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import transformers
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    UNet2DConditionModel,
)
import lpips
from utils.models import *
from utils.misc import torch_to_pil
from utils.noise_layers.noiser import Noiser
import torchsummary


def Channel_Loss(img1, img2, k):
    diff = torch.abs(img1 - img2)
    diff_combined = torch.mean(diff, dim=1, keepdim=False)  
    diff_combined_flat = diff_combined.view(-1)
    top_k_diff, _ = torch.topk(diff_combined_flat, k)
    loss = torch.sum(top_k_diff)
    
    return loss
def base_augment(image):
    if random.random() > 0.5:
        image = torch.flip(image, dims=[-1])
    image = torch.rot90(image, k=random.randint(0, 3), dims=[-2, -1])
    return image

class traindataset(Dataset):
    def __init__(self, root, random_aug=True):
        self.root = root
        self.image_files = glob.glob(root + "/*.png") + glob.glob(root + "/*.jpg")
        self.random_aug = random_aug
    def __len__(self):
        return len(self.image_files)
    def process(self, image):
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = image.resize((512, 512), resample=PIL.Image.Resampling.BICUBIC)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = torch.from_numpy(image).permute(2, 0, 1)
        if self.random_aug and random.random() > 0.5:
            image = base_augment(image)
        return image
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image = Image.open(img_name)
        image = self.process(image)
        return image

def generate_random_mask(batch_size, height=512, width=512):
    mask = torch.zeros(batch_size, 3, height, width)
    total_area = height * width
    for b in range(batch_size):
        area_ratio = random.uniform(0.3, 0.6)
        target_area = int(total_area * area_ratio)
        found = False
        while not found:
            mask_height = random.randint(1, height)
            mask_width = target_area // mask_height
            if mask_width <= width and mask_height <= height:
                found = True
        start_x = random.randint(0, width - mask_width)
        start_y = random.randint(0, height - mask_height)
        mask[b, :, start_y:start_y + mask_height, start_x:start_x + mask_width] = 1
    return mask

argparser = argparse.ArgumentParser()
argparser.add_argument('--pretrained_model_name_or_path', type=str, default='../../stable-diffusion-v1-4/')
argparser.add_argument('--epochs', type=int, default=40)
argparser.add_argument('--batch_size', type=int, default=2)
argparser.add_argument('--bit_num', type=int, default=32)
argparser.add_argument('--resume_from_ckpt', type=str, default=None)
argparser.add_argument('--dataset', type=str,default='coco/ground_truth/')
argparser.add_argument('--output_dir', default='checkpoints')
argparser.add_argument('--warmup', default=True)
argparser.add_argument('--init',default=True)
argparser.add_argument('--random_aug',default=True)
argparser.add_argument('--lr', type=float, default=0.0005)
args =argparser.parse_known_args()[0]

if os.path.exists(args.output_dir) is False:
    os.makedirs(args.output_dir)
    os.makedirs(args.output_dir + '/logs')
    os.makedirs(f'{args.output_dir}/checkpoints')
    os.makedirs(f'{args.output_dir}/log_images')
    
train_loader = torch.utils.data.DataLoader(
        traindataset(args.dataset, args.random_aug),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
vae = vae.cuda()
# Freeze vae and unet
vae.requires_grad_(False)
def decode_latents(latents):
    # latents = 1 / vae.config.scaling_factor * latents
    image = vae.decode(latents).sample
    return image
sec_encoder = SecretEncoder(args.bit_num).cuda()
sec_decoder = SecretDecoder(output_size=args.bit_num).cuda()
# get params size
torchsummary.summary(sec_decoder, (3, 512, 512))
loss_fn_alex = lpips.LPIPS(net='alex').cuda()
loss_fn_alex.requires_grad_(False)
noise_config = ['Identity','Jpeg','CropandResize','GaussianBlur','GaussianNoise','ColorJitter']
posibilities = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
noiser = Noiser(noise_config, posibilities, device='cuda')
current_epoch = 0
if args.resume_from_ckpt is not None:
    models = torch.load(args.resume_from_ckpt)
    sec_encoder.load_state_dict(models['sec_encoder'])
    sec_decoder.load_state_dict(models['sec_decoder'])
    current_epoch = int(args.resume_from_ckpt.split('_')[-1].split('.')[0])
optimizer = optim.AdamW([
    {'params': sec_encoder.parameters()},
    {'params': sec_decoder.parameters()}
], lr=args.lr, weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=2, gamma=0.8)
writer = SummaryWriter(args.output_dir + '/logs')

def gen_combined_latents(latents, wm_latent, scale=1.0):
    cornerfy_aug = random.choice([True, False, False, False]) # 1/4 chance to cornerfy_aug
    height, width = wm_latent.shape[2], wm_latent.shape[3]
    height_scale, width_scale = (random.uniform(1.0, 2.0), random.uniform(1.0, 2.0)) if cornerfy_aug else (1.0, 1.0)
    if cornerfy_aug:
        wm_template = F.interpolate(torch.zeros_like(latents), scale_factor=(height_scale, width_scale), mode='bilinear') if cornerfy_aug else latents
        wm_template[:,:,:height//2, :width//2] =  wm_latent[:,:,:height//2, :width//2]
        wm_template[:,:,:height//2, -width//2:] = wm_latent[:,:,:height//2, -width//2:]
        wm_template[:,:,-height//2:,:width//2] =  wm_latent[:,:,-height//2:,:width//2]
        wm_template[:,:,-height//2:,-width//2:] = wm_latent[:,:,-height//2:,-width//2:]
        wm_template = F.interpolate(wm_template, size=(height, width), mode='bilinear')
    else:
        wm_template = wm_latent
    watermarked_latents = latents + wm_template * scale
    return watermarked_latents

all_iter = args.epochs * len(train_loader)
pbar = tqdm(total=all_iter)
iterations = 0
warmup = args.warmup
fixinit= args.init
zero_batch = torch.zeros(args.batch_size, 3, 512, 512).cuda()
for epoch in range(current_epoch, current_epoch + args.epochs):
    sec_encoder.train()
    sec_decoder.train()
    msgloss_10buffer = []
    for batch_idx, oimage in enumerate(train_loader):
        if fixinit:
            # use zero image when start training
            oimage =zero_batch
        optimizer.zero_grad()
        oimage = oimage.cuda()
        mask_tensor = generate_random_mask(args.batch_size).cuda()
        latents = vae.encode(oimage*mask_tensor).latent_dist.sample().detach()
        msg = torch.randint(0, 2, (args.batch_size, args.bit_num)).cuda()
        _, wm_latent = sec_encoder(latents, msg.float())
        if warmup:
            watermarked_latents = gen_combined_latents(latents, wm_latent, scale=0.03)
        else:
            watermarked_latents = gen_combined_latents(latents, wm_latent)
        clean_image = decode_latents(latents).detach()
        watermarked_image = decode_latents(watermarked_latents)
        lpips_loss = loss_fn_alex(clean_image, watermarked_image).mean()
        channel_loss = Channel_Loss(clean_image, watermarked_image)
        if epoch - current_epoch > 12 or args.resume_from_ckpt is not None:
            watermarked_image = noiser([watermarked_image,None],[0.4, 0.1, 0.2, 0.05, 0.1, 0.15])[0]
        else:
            watermarked_image = noiser([watermarked_image,None],[0.6, 0., 0.4, 0., 0., 0.])[0]
        mask_watermarked_image=watermarked_image*mask_tensor
        reveal_output = sec_decoder(mask_watermarked_image)
        # Create labels tensor
        labels = F.one_hot(msg, num_classes=2).float()
        # change to BCE loss
        msgloss = F.binary_cross_entropy_with_logits(reveal_output, labels.cuda())
        
        if len(msgloss_10buffer) == 10:
            msgloss_10buffer.pop(0)
        msgloss_10buffer.append(msgloss.item())
        # when the message loss is less than 0.1 for 10 consecutive batches, we consider the model is warmed up
        if len(msgloss_10buffer) == 10 and sum(msgloss_10buffer) / 10 < 0.1:
            warmup = False
            fixinit = False
        if warmup:
            loss = msgloss
        else:
            if epoch> 5:
                loss = lpips_loss * 6 + msgloss * 2 + channel_loss * 3
            else:
                loss = msgloss
        loss.backward()
        optimizer.step()
        pbar.update(1)
        iterations += 1
        print('lpips_loss: %.6f, channel_loss:%.4f, msgloss: %.4f, loss: %.4f' % (lpips_loss, channel_loss,msgloss, loss))
        writer.add_scalar('Loss/train', loss, iterations)
        writer.add_scalar('Loss/lpips_loss', lpips_loss, iterations)
        writer.add_scalar('Loss/channel_loss', channel_loss, iterations)
        writer.add_scalar('Loss/msgloss', msgloss, iterations)
        
    watermarked_image_pil = torch_to_pil(watermarked_image)[0]
    watermarked_image_pil.save(f"{args.output_dir}/log_images/watermarked_{epoch}_{batch_idx}.png")

    sec_encoder.eval()
    sec_decoder.eval()
    with torch.no_grad():
        msg_val = torch.randint(0, 2, (args.batch_size, args.bit_num)).cuda()
        watermarked_latent, _ = sec_encoder(latents, msg_val.float())
        watermarked = decode_latents(watermarked_latent)
        decoded_msg = sec_decoder(watermarked*mask_tensor)
        decoded_msg = torch.argmax(decoded_msg, dim=2)
        acc = 1 - torch.abs(decoded_msg - msg_val).sum().float() / (args.bit_num * args.batch_size)
        print(f"Epoch {epoch}: acc {acc}")
        writer.add_scalar('Accuracy/train', acc, epoch)
    print(f"Epoch {epoch}: loss {loss}, lpips_loss {lpips_loss}, prvl_loss {channel_loss}, msgloss {msgloss}")
    scheduler.step()
    torch.save({
        'sec_decoder': sec_decoder.state_dict(),
        'sec_encoder': sec_encoder.state_dict(),
    }, f"{args.output_dir}/checkpoints/state_dict_{epoch}.pth")
    torch.cuda.empty_cache()
writer.close()

