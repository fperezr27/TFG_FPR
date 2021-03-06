#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 17:12:08 2019

@author: aneesh
"""
import argparse
import itertools
from distutils.log import error

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
from helpers.utils import parse_args
from networks.resnet6 import ResnetGenerator
from networks.segnet import segnet, segnetm
from networks.unet import unet, unetm
from networks.mobileone_pytorch import mobileone_s4
import torch
import os
import os.path as osp
import numpy as np

from skimage import io
from PIL import Image
import matplotlib.pyplot as plt

import scipy.io as sio

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'AeroRIT baseline evalutions')    
    
    ### 0. Config file?
    parser.add_argument('--config-file', default = None, help = 'Path to configuration file')
    
    ### 1. Data Loading
    parser.add_argument('--bands', default = 51, help = 'Which bands category to load \
                        - 3: RGB, 4: RGB + 1 Infrared, 6: RGB + 3 Infrared, 31: Visible, 51: All', type = int)
    parser.add_argument('--hsi_c', default = 'rad', help = 'Load HSI Radiance or Reflectance data?')
    
    ### 2. Network selections
    ### a. Which network?
    parser.add_argument('--network_arch', default = 'unet', help = 'Network architecture?')
    parser.add_argument('--use_mini', action = 'store_true', help = 'Use mini version of network?')
    
    ### b. ResNet config
    parser.add_argument('--resnet_blocks', default = 6, help = 'How many blocks if ResNet architecture?', type = int)
    
    ### c. UNet configs
    parser.add_argument('--use_SE', action = 'store_true', help = 'Network uses SE Layer?')
    parser.add_argument('--use_preluSE', action = 'store_true', help = 'SE layer uses ReLU or PReLU activation?')
    
    ### Load weights post network config
    parser.add_argument('--network_weights_path', default = './savedmodels/network.pt', help = 'Path to Saved Network weights')
    
    ### Use GPU or not
    parser.add_argument('--use_cuda', action = 'store_true', help = 'use GPUs?')
    
    args = parse_args(parser)
    print(args)
    
    size_chips = 64
    
    args.use_mini = True
    # args.use_SE = True
    # args.use_preluSE = True
    #args.network_weights_path = 'savedmodels/unetm.pt'
    
    if args.use_cuda and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    

    folder_dir = osp.join('Aerial Data', 'Collection') #path to full files
    
    image_rgb = io.imread(osp.join(folder_dir, 'image_rgb.tif'))[53:,7:,:]
    
    image_hsi_rad = io.imread(osp.join(folder_dir, 'image_hsi_radiance.tif'))
    image_hsi_rad = np.transpose(image_hsi_rad, [1,2,0])[53:,7:,:]
    
    image_hsi_ref = io.imread(osp.join(folder_dir, 'image_hsi_reflectance.tif'))
    image_hsi_ref = np.transpose(image_hsi_ref, [1,2,0])[53:,7:,:]
    
    image_labels = io.imread(osp.join(folder_dir, 'image_labels.tif'))[53:,7:,:]
    


    # if args.network_arch == 'resnet':
    #     net = ResnetGenerator(args.bands, 6, n_blocks=args.resnet_blocks)
    # elif args.network_arch == 'segnet':
    #     if args.mini == True:
    #         net = segnetm(args.bands, 6)
    #     else:
    #         net = segnet(args.bands, 6)
    # elif args.network_arch == 'unet':
    #     if args.use_mini == True:
    #         net = unetm(args.bands, 6, use_SE = args.use_SE, use_PReLU = args.use_preluSE)
    #     else:
    #         net = unet(args.bands, 6)
    # else:
    #     raise NotImplementedError('required parameter not found in dictionary')

    net = mobileone_s4()

    net.load_state_dict(torch.load(args.network_weights_path))
    net.eval()
    net.to(device)
    
    print('Completed loading pretrained network weights...')
    
    print('Calculating prediction accuracy...')
    
    labels_gt = []
    labels_pred = []

    patches_pred = []
    
    clmap_color = np.zeros(((1920, 3968, 3)))
    clmap_labels = np.zeros(((1920,3968)))
    clmap_labels_color = np.zeros(((1920, 3968, 3)))
    clmap_dif = np.zeros(((1920, 3968, 3)))
    errors = {"building":0, "ground":0, "road":0, "water":0,"car":0}
    class_num = {"building":0, "ground":0, "road":0, "water":0,"car":0}
    precisions = {"building":0, "ground":0, "road":0, "water":0,"car":0}
    keys = list(errors.keys())

    rgb = image_rgb[:,:,:]
    rad = image_hsi_rad[:,:,:]
    ref = image_hsi_ref[:,:,:]
    labels = image_labels[:,:,:]


    x_arr, y_arr, _ = rgb.shape
        
    salir=False
    for xx in range(0, x_arr - size_chips//2, size_chips//2):
        for yy in range(0, y_arr - size_chips//2, size_chips//2):

            name = 'image_{}_{}'.format(xx,yy)
                
            rgb_temp = rgb[xx:xx + size_chips, yy:yy + size_chips,:]
            rgb_temp = Image.fromarray(rgb_temp)
                
            hsi_rad_temp = rad[xx:xx + size_chips, yy:yy + size_chips,:]
            hsi_ref_temp = ref[xx:xx + size_chips, yy:yy + size_chips,:]
                
            labels_temp = labels[xx:xx + size_chips, yy:yy + size_chips,:]
            shapeim = labels_temp.shape
                
                
            if shapeim[0] == shapeim[1] and shapeim[0] == size_chips:                    
                hsi_rad_temp = np.clip(hsi_rad_temp, 0, 2**14)/2**14
                hsi_rad_temp = np.transpose(hsi_rad_temp, (2, 0, 1)).astype("float32")
                hsi_rad_temp = torch.from_numpy(hsi_rad_temp)
                label_pred = net(hsi_rad_temp.unsqueeze(0).to(device))
                label_pred = label_pred.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()
                labels_temp = Image.fromarray(labels_temp)


                if clmap_color[xx:xx+size_chips,yy:yy+size_chips].shape[0] == \
                    clmap_color[xx:xx+size_chips,yy:yy+size_chips].shape[1] and \
                    clmap_color[xx:xx+size_chips,yy:yy+size_chips].shape[0] == size_chips and \
                    clmap_color[xx:xx+size_chips,yy:yy+size_chips].shape[1] == size_chips:
                        clmap_color[xx:xx+size_chips,yy:yy+size_chips, :] = labels_temp
                        
  
                

                if clmap_labels[xx:xx+size_chips,yy:yy+size_chips].shape[0] == \
                    clmap_labels[xx:xx+size_chips,yy:yy+size_chips].shape[1] and \
                    clmap_labels[xx:xx+size_chips,yy:yy+size_chips].shape[0] == size_chips and \
                    clmap_labels[xx:xx+size_chips,yy:yy+size_chips].shape[1] == size_chips:
                        clmap_labels[xx:xx+size_chips,yy:yy+size_chips] = label_pred
                # if xx > 300:
                #     salir=True
                #     break
        print(xx)
#            if salir: break
#        break

    for i in range(1920):
        for j in range(3968):
            if clmap_labels[i,j] == 1:
                #print(clmap_color[i,j])
                clmap_labels_color[i,j,1] = 255
                class_num[keys[int(clmap_labels[i,j])]] +=1

            elif clmap_labels[i,j] == 2:
                #print(clmap_color[i,j])
                clmap_labels_color[i,j,2] = 255
                class_num[keys[int(clmap_labels[i,j])]] +=1
            elif clmap_labels[i,j] == 3:
                #print(clmap_color[i,j])
                clmap_labels_color[i,j,1] = 255
                clmap_labels_color[i,j,2] = 255
                class_num[keys[int(clmap_labels[i,j])]] +=1
            elif clmap_labels[i,j] == 4:
                # print(clmap_color[i,j])
                clmap_labels_color[i,j,0] = 255
                clmap_labels_color[i,j,1] = 127
                clmap_labels_color[i,j,2] = 80
                class_num[keys[int(clmap_labels[i,j])]] +=1
            elif clmap_labels[i,j] == 0:
                # print(clmap_color[i,j])
                clmap_labels_color[i,j,0] = 255
                class_num[keys[int(clmap_labels[i,j])]] +=1
                
            if clmap_color[i,j,0] != clmap_labels_color[i,j,0] or \
                clmap_color[i,j,1] != clmap_labels_color[i,j,1] or \
                clmap_color[i,j,2] != clmap_labels_color[i,j,2]:

                clmap_dif[i,j] = [255,255,255]
                errors[keys[int(clmap_labels[i,j])]] +=1

    for index in  range(5):
        precisions[keys[index]] = (class_num[keys[index]]-errors[keys[index]])/(class_num[keys[index]])

    #Creating confussion matrix
    
    confusionMatrix = np.zeros(((5,5)))

    for i in range(1920):
        for j in range(3968):
            if clmap_labels[i,j] == 0:
                if clmap_color[i,j,0] == 255 and clmap_color[i,j,1] == 0 and clmap_color[i,j,2] == 0:
                    confusionMatrix[0,0] += 1
                elif clmap_color[i,j,1] == 255 and clmap_color[i,j,0] == 0 and clmap_color[i,j,2] == 0:
                    confusionMatrix[0,1] += 1
                elif clmap_color[i,j,2] == 255 and clmap_color[i,j,0] == 0 and clmap_color[i,j,1] == 0:
                    confusionMatrix[0,2] += 1
                elif clmap_color[i,j,1] == 255 and clmap_color[i,j,2] == 255 and clmap_color[i,j,0] == 0:
                    confusionMatrix[0,3] += 1
                elif clmap_color[i,j,2] == 80 and clmap_color[i,j,0] == 255 and clmap_color[i,j,1] == 127:
                    confusionMatrix[0,4] += 1
            elif clmap_labels[i,j] == 1:
                if clmap_color[i,j,0] == 255 and clmap_color[i,j,1] == 0 and clmap_color[i,j,2] == 0:
                    confusionMatrix[1,0] += 1
                elif clmap_color[i,j,1] == 255 and clmap_color[i,j,0] == 0 and clmap_color[i,j,2] == 0:
                    confusionMatrix[1,1] += 1
                elif clmap_color[i,j,2] == 255 and clmap_color[i,j,0] == 0 and clmap_color[i,j,1] == 0:
                    confusionMatrix[1,2] += 1
                elif clmap_color[i,j,1] == 255 and clmap_color[i,j,2] == 255 and clmap_color[i,j,0] == 0:
                    confusionMatrix[1,3] += 1
                elif clmap_color[i,j,2] == 80 and clmap_color[i,j,0] == 255 and clmap_color[i,j,1] == 127:
                    confusionMatrix[1,4] += 1
            elif clmap_labels[i,j] == 2:
                if clmap_color[i,j,0] == 255 and clmap_color[i,j,1] == 0 and clmap_color[i,j,2] == 0:
                    confusionMatrix[2,0] += 1
                elif clmap_color[i,j,1] == 255 and clmap_color[i,j,0] == 0 and clmap_color[i,j,2] == 0:
                    confusionMatrix[2,1] += 1
                elif clmap_color[i,j,2] == 255 and clmap_color[i,j,0] == 0 and clmap_color[i,j,1] == 0:
                    confusionMatrix[2,2] += 1
                elif clmap_color[i,j,1] == 255 and clmap_color[i,j,2] == 255 and clmap_color[i,j,0] == 0:
                    confusionMatrix[2,3] += 1
                elif clmap_color[i,j,2] == 80 and clmap_color[i,j,0] == 255 and clmap_color[i,j,1] == 127:
                    confusionMatrix[2,4] += 1
            elif clmap_labels[i,j] == 3:
                if clmap_color[i,j,0] == 255 and clmap_color[i,j,1] == 0 and clmap_color[i,j,2] == 0:
                    confusionMatrix[3,0] += 1
                elif clmap_color[i,j,1] == 255 and clmap_color[i,j,0] == 0 and clmap_color[i,j,2] == 0:
                    confusionMatrix[3,1] += 1
                elif clmap_color[i,j,2] == 255 and clmap_color[i,j,0] == 0 and clmap_color[i,j,1] == 0:
                    confusionMatrix[3,2] += 1
                elif clmap_color[i,j,1] == 255 and clmap_color[i,j,2] == 255 and clmap_color[i,j,0] == 0:
                    confusionMatrix[3,3] += 1
                elif clmap_color[i,j,2] == 80 and clmap_color[i,j,0] == 255 and clmap_color[i,j,1] == 127:
                    confusionMatrix[3,4] += 1
            elif clmap_labels[i,j] == 4:
                if clmap_color[i,j,0] == 255 and clmap_color[i,j,1] == 0 and clmap_color[i,j,2] == 0:
                    confusionMatrix[4,0] += 1
                elif clmap_color[i,j,1] == 255 and clmap_color[i,j,0] == 0 and clmap_color[i,j,2] == 0:
                    confusionMatrix[4,1] += 1
                elif clmap_color[i,j,2] == 255 and clmap_color[i,j,0] == 0 and clmap_color[i,j,1] == 0:
                    confusionMatrix[4,2] += 1
                elif clmap_color[i,j,1] == 255 and clmap_color[i,j,2] == 255 and clmap_color[i,j,0] == 0:
                    confusionMatrix[4,3] += 1
                elif clmap_color[i,j,2] == 80 and clmap_color[i,j,0] == 255 and clmap_color[i,j,1] == 127:
                    confusionMatrix[4,4] += 1
    

    confusionMatrix = normalize(confusionMatrix, axis=1, norm='l1')
    confusionMatrix = np.round(confusionMatrix, decimals=4)
    plt.imshow(confusionMatrix,interpolation = 'nearest', cmap = 'plasma')
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(keys))
    plt.xticks(tick_marks, keys, rotation = 45)
    plt.yticks(tick_marks, keys)

    thres = 0.
    confusionMatrix = normalize(confusionMatrix, axis=1, norm='l1')
    confusionMatrix = np.round(confusionMatrix, decimals=4)
    for i, j in itertools.product(range(confusionMatrix.shape[0]), range(confusionMatrix.shape[1])):
        plt.text(j, i, confusionMatrix[i, j], horizontalalignment="center", color="white" if confusionMatrix[i, j] < thres else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
       
                
    sio.savemat("clmap.mat", {"clmap":clmap_color})
    plt.imshow(clmap_color,vmin=0)
    plt.show()
    sio.savemat("clmap_pred.mat", {"clmap_pred":clmap_labels})
    plt.imshow(clmap_labels,vmin=0)
    plt.show()
    sio.savemat("clmap_comp.mat", {"clmap_comp":clmap_labels_color})
    plt.imshow(clmap_labels_color,vmin=0)
    plt.show()
    sio.savemat("clmap_diff.mat", {"clmap_diff":clmap_dif})
    plt.imshow(clmap_dif,vmin=0)
    plt.show()
    print(errors)
    print(precisions)
    print(net)

    # sio.savemat("clmap_all.mat", {"clmap_diff":clmap_dif,"clmap_comp":clmap_labels_color, "clmap":clmap_color})
    
    

                
                #rgb_temp.save(osp.join(loc, 'RGB', name + '.tif'))
                #labels_temp.save(osp.join(loc, 'Labels', name + '.tif'))
                #np.save(osp.join(loc, 'HSI-rad', name), hsi_rad_temp)
                #np.save(osp.join(loc, 'HSI-ref', name), hsi_ref_temp)
                
                #trainfile.write("%s\n" % name)
                
                #if (xx%size_chips == 0 and yy%size_chips == 0):
                    #testfile.write("%s\n" % name)

        #trainfile.close()
        #testfile.close()
        
        #print('Stopping chip making now')
