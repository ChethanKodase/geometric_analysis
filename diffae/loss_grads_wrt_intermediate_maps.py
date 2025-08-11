


#%load_ext autoreload
#%autoreload 2

'''

cd geometric_analysis
conda activate dt2
python diffae/loss_grads_wrt_intermediate_maps.py --desired_norm_l_inf 0.27 --which_gpu 1 --diffae_checkpoint ../diffae/checkpoints --ffhq_images_directory ../diffae/imgs_align_uni_ad --chosen_space_ind 14

'''


from templates import *
import matplotlib.pyplot as plt
import torch.optim as optim

from torch.nn import DataParallel
import torch.nn.functional as F

from torch.utils.data import DataLoader
from conditioning import get_layer_pert_recon

import torch.autograd.function
print(torch.autograd.function.__file__)

#seeding code begins
import torch
import random
import numpy as np
import os

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#seeding code ends


import argparse

parser = argparse.ArgumentParser(description='DiffAE celebA training')

parser.add_argument('--which_gpu', type=int, default=0, help='Index of the GPU to use (0-N)')
parser.add_argument('--attck_type', type=str, default=5, help='Type of attack')
parser.add_argument('--desired_norm_l_inf', type=float, default=0.08, help='Type of attack')
parser.add_argument('--diffae_checkpoint', type=str, default=5, help='Type of attack')
parser.add_argument('--ffhq_images_directory', type=str, default=5, help='images directory')

parser.add_argument('--chosen_space_ind', type=int, default=5, help='images directory')


args = parser.parse_args()

which_gpu = args.which_gpu
attck_type = args.attck_type
desired_norm_l_inf = args.desired_norm_l_inf
diffae_checkpoint = args.diffae_checkpoint
ffhq_images_directory = args.ffhq_images_directory
chosen_space_ind = args.chosen_space_ind

device = 'cuda:'+str(which_gpu)+''


conf = ffhq256_autoenc()

#conf = ffhq256_autoenc_latent()
print(conf.name)
model = LitModel(conf)
print("diffae_checkpoint", diffae_checkpoint)
#state = torch.load(f'../diffae/checkpoints/{conf.name}/last.ckpt', map_location='cpu')
state = torch.load(f"{diffae_checkpoint}/{conf.name}/last.ckpt", map_location='cpu')
model.load_state_dict(state['state_dict'], strict=False)
#model.ema_model.eval()
model.ema_model.to(device);


total_params = sum(p.numel() for p in model.ema_model.parameters())
trainable_params = sum(p.numel() for p in model.ema_model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")

data = ImageDataset(ffhq_images_directory, image_size=conf.img_size, exts=['jpg', 'JPG', 'png'], do_augment=False)

print("{len(data)}", len(data))

batch_size = 25
train_loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

batch_list = []
for source_im in train_loader:
    batch = source_im['img'].to(device)
    batch_list.append(batch)  # Store batch in a list
    print("len(batch_list)", len(batch_list))
    if(len(batch_list)==3):
        break
big_tensor = torch.stack(batch_list)  # This we do to put all the images into the GPU so that there is no latency due to communication between CPU and GPU during optimization
print("big_tensor.shape", big_tensor.shape)
del batch_list
del train_loader


source_im = data[0]['img'][None].to(device)


import matplotlib.pyplot as plt
import os




noise_addition = torch.rand(source_im.shape).to(device)
print("no file, so random initialization")

noise_addition = noise_addition * (source_im.max() - source_im.min()) + source_im.min()  
noise_addition = noise_addition.clone().detach().requires_grad_(True)




#noise_addition.requires_grad = True
optimizer = optim.Adam([noise_addition], lr=0.0001)
source_im.requires_grad = True

adv_alpha = 0.5

criterion = nn.MSELoss()

num_steps = 1000000
from geomloss import SamplesLoss


with torch.no_grad():
    cond_nums_normalized = get_layer_pert_recon(model)
    print("cond_nums_normalized", cond_nums_normalized)
#################################################################################################################

def cos(a, b):
    a = a.view(-1)
    b = b.view(-1)
    a = F.normalize(a, dim=0)
    b = F.normalize(b, dim=0)
    return (a * b).sum()


def run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, normalized_attacked, optimized_noise, adv_gen):
    print(f"Step {step}, Loss: {total_loss.item()}, distortion L-2: {l2_distortion}, distortion L-inf: {l_inf_distortion}, deviation: {deviation}")
    print()
    print("attack type", attck_type)    
    adv_div_list.append(deviation.item())
    with torch.no_grad():
        fig, ax = plt.subplots(1, 3, figsize=(10, 10))
        ax[0].imshow(((normalized_attacked[0]+1)/2).permute(1, 2, 0).cpu().numpy())
        ax[0].set_title('Attacked Image')
        ax[0].axis('off')

        ax[1].imshow(optimized_noise[0].cpu().detach().permute(1, 2, 0).cpu().numpy())
        ax[1].set_title('Noise')
        ax[1].axis('off')

        ax[2].imshow(adv_gen[0].cpu().detach().permute(1, 2, 0).cpu().numpy())
        ax[2].set_title('Attack reconstruction')
        ax[2].axis('off')
        plt.show()
        plt.savefig("diffae/runtime_plots/DiffAE_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+"_.png")
    torch.save(optimized_noise, "diffae/noise_storage/DiffAE_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+"_.pt")
    #np.save("../diffae/attack_run_time_univ/adv_div_convergence/DiffAE_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+"_.npy", adv_div_list)





def get_combined_cosine_loss_gcr_simp2(normal_x, source_x):

    retained_outputs = []

    # Input blocks

    for i, block in enumerate(model.ema_model.encoder.input_blocks):
        
        source_x = block(source_x)
        normal_x = block(normal_x)

        normal_x.retain_grad()
        retained_outputs.append(normal_x)
        #retained_outputs.append(source_x)

    for i, block in enumerate(model.ema_model.encoder.middle_block):
        source_x = block(source_x)
        normal_x = block(normal_x)

        normal_x.retain_grad()
        retained_outputs.append(normal_x)
        #retained_outputs.append(source_x)

    for i, block in enumerate(model.ema_model.encoder.out):
        source_x = block(source_x)
        normal_x = block(normal_x)

        normal_x.retain_grad()
        retained_outputs.append(normal_x)
        #retained_outputs.append(source_x)

    final_loss = -1 * ((F.cosine_similarity(source_x, normal_x, dim=1) - 1) ** 2).mean()

    return final_loss, retained_outputs



adv_div_list = []
epoch_list = []
loss_list = []
sel_layer_loss_list = []
for step in range(155):
    batch_ali = []
    loss_ali = []
    layer_step_loss = []
    for source_im in big_tensor:
        print("noise_addition.shape", noise_addition.shape)
        print("source_im.shape", source_im.shape)
        normalized_attacked = (source_im + (noise_addition * (desired_norm_l_inf / (torch.norm(noise_addition, p=float('inf')))) ))
        normalized_attacked = ( source_im.max() - source_im.min() ) * ((normalized_attacked-normalized_attacked.min())/(normalized_attacked.max()-normalized_attacked.min()))  + source_im.min() 
        final_loss, retained_outputs = get_combined_cosine_loss_gcr_simp2(normalized_attacked, source_im)

        final_loss.backward()
        print("noise_addition.grad.shape", noise_addition.grad.shape)
        for i, block in enumerate(model.ema_model.encoder.input_blocks):

            for name, param in block.named_parameters():
                print("name", name)
                print("param.shape", param.shape)
                print("param.grad", param.grad)

        optimizer.step()
        optimizer.zero_grad()

        layer_ind = 0
        for nx in retained_outputs:
            print(f"Gradient for layer :{layer_ind}, of shape {nx.grad.shape} - mean grad: {nx.grad.abs().mean().item():.6f}")
            print("l2 norm : ", torch.norm(nx, 2))
            print("retained output shape : ", nx.shape)
            print()
            layer_ind +=1
        
    break

    print("step", step)
    if(step%50==0 and not step==0):
        with torch.no_grad():
            attacked_embed = model.encode(normalized_attacked.to(device))
            xT_ad = model.encode_stochastic(normalized_attacked.to(device), attacked_embed, T=250)
            adv_gen = model.render(xT_ad, attacked_embed, T=20)
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            deviation = torch.norm(adv_gen - source_im, p=2)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, normalized_attacked, scaled_noise, adv_gen)


