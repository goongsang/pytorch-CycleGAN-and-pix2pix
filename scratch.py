import os
import random
from shutil import copyfile

# dataRoot = "/media/jseo/BigData/DI4D_SideLA/transferred_cache/yay/deltas/"
# pix2pixRoot = "/home/jseo/nv/pix2pix-yay/datasets/yay/"

dataRoot = "F:/DI4D_SideLA/transferred_cache/yay/deltas512/"
pix2pixRoot = "F:/DI4D_SideLA/transferred_cache/yay/deltas512_dataset/"

dataFiles = sorted(os.listdir(dataRoot))
random.seed(1711)
random.shuffle(dataFiles)

n = len(dataFiles)
nTests = int(n*0.12)
nVals = int(n*0.12)

testFiles = dataFiles[:nTests]
for i,f in enumerate(testFiles):
    print(i,f)
    # os.system("ln -s %s %s%05d%s"%(dataRoot+f, pix2pixRoot+"test/", i, ".npy"))
    copyfile(dataRoot+f, "%s%05d%s"%(pix2pixRoot+"test/", i, ".npy"))

valFiles = dataFiles[nTests:nTests+nVals]
for i,f in enumerate(valFiles):
    print(i,f)
    # os.system("ln -s %s %s%05d%s"%(dataRoot+f, pix2pixRoot+"val/", i, ".npy"))
    copyfile(dataRoot+f, "%s%05d%s"%(pix2pixRoot+"val/", i, ".npy"))

trainFiles = dataFiles[nTests+nVals:]
for i,f in enumerate(trainFiles):
    print(i,f)
    # os.system("ln -s %s %s%05d%s"%(dataRoot+f, pix2pixRoot+"train/", i, ".npy"))
    copyfile(dataRoot+f, "%s%05d%s"%(pix2pixRoot+"train/", i, ".npy"))


# python train.py --dataroot ./datasets/yay/ --name yay --model pix2pix --dataset_mode aligned_raw --netG unet_512 --num_threads 8 --preprocess none --batch_size 5 --continue_train
# python train.py --dataroot F:/DI4D_SideLA/transferred_cache/yay/deltas512_dataset/ --name yay512 --model pix2pix --dataset_mode aligned_raw --netG unet_512 --num_threads 8 --preprocess none --batch_size 5 --continue_train
# cd .\nvidia\yay\pix2pix-yay\;conda activate pytorch-pix2pix;
# python train.py --dataroot F:/DI4D_SideLA/transferred_cache/yay/deltas512_dataset/ --name yay512 --model pix2pix --dataset_mode aligned_raw --netG unet_512 --num_threads 8 --preprocess none --batch_size 2 --continue_train --lr 0.0004 --save_epoch_freq 1

# ----------------- Options ---------------
#                batch_size: 1                             
#                     beta1: 0.5                           
#           checkpoints_dir: ./checkpoints                 
#            continue_train: False                         
#                 crop_size: 256                           
#                  dataroot: ./datasets/night2day                 [default: None]
#              dataset_mode: aligned                       
#                 direction: AtoB                          
#               display_env: main                          
#              display_freq: 400                           
#                display_id: 1                             
#             display_ncols: 4                             
#              display_port: 8097                          
#            display_server: http://localhost              
#           display_winsize: 256                           
#                     epoch: latest                        
#               epoch_count: 1                             
#                  gan_mode: vanilla                       
#                   gpu_ids: 0                             
#                 init_gain: 0.02                          
#                 init_type: normal                        
#                  input_nc: 3                             
#                   isTrain: True                                 [default: None]
#                 lambda_L1: 100.0                         
#                 load_iter: 0                                    [default: 0]
#                 load_size: 286                           
#                        lr: 0.0002                        
#            lr_decay_iters: 50                            
#                 lr_policy: linear                        
#          max_dataset_size: inf                           
#                     model: pix2pix                              [default: cycle_gan]
#                  n_epochs: 100                           
#            n_epochs_decay: 100                           
#                n_layers_D: 3                             
#                      name: night2day_pix2pix                    [default: experiment_name]
#                       ndf: 64                            
#                      netD: basic                         
#                      netG: unet_256                      
#                       ngf: 64                            
#                no_dropout: False                         
#                   no_flip: False                         
#                   no_html: False                         
#                      norm: batch                         
#               num_threads: 4                             
#                 output_nc: 3                             
#                     phase: train                         
#                 pool_size: 0                             
#                preprocess: resize_and_crop               
#                print_freq: 100                           
#              save_by_iter: False                         
#           save_epoch_freq: 5                             
#          save_latest_freq: 5000                          
#            serial_batches: False                         
#                    suffix:                               
#          update_html_freq: 1000                          
#                   verbose: False                         
# ----------------- End -------------------