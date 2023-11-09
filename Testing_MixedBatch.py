from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torchvision import models, transforms
from torch.utils.data import Dataset

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
import copy
from PIL import Image

# from sklearn.metrics import roc_auc_score, roc_curve

import warnings
warnings.filterwarnings('ignore')

cudnn.benchmark = True
plt.ion()   # interactive mode



# In[2] Dataset Class
class NSCLC_Dataset(Dataset):
    ''' Dataset class for NSCLC dataset
    
    Parameters:
    root_dir (string) : Directory of the dataset
    data_type (string): 'train' or 'val' for training and validation/testing
    transform (torchvision.transforms) : Data transform
    validation_slide_split (int array) : Slide numbers of the reserved balanced slides for validation/testing
    '''
    
    def __init__(self, B1Path, B2Path, transform, slides, labels, tile_per_slide):
        self.B1Path = B1Path
        self.B2Path = B2Path
        self.transform = transform
        self.slides = slides
        self.tile_per_slide = tile_per_slide

        self.filename_list = []

        count = 0
        for slide in self.slides:
            label = labels[count]
            count += 1
            # print(slide)

            if slide[0] == 'D':
                slide_path = os.path.join(self.B1Path, 'Data', slide)
            else:
                slide_path = os.path.join(self.B2Path, 'Data', slide)

            num = len(os.listdir(slide_path))
            np.random.seed(10)
            idxs = np.random.choice(num, self.tile_per_slide, replace=False)
            list_temp = os.listdir(slide_path)
            # Append slide_path with filename
            for idx in idxs:
                self.filename_list.append((os.path.join(slide_path, list_temp[idx]), label))

        # np.random.shuffle(self.filename_list)
            

    def __len__(self):
        return len(self.slides) * self.tile_per_slide
    
    def __getitem__(self, idx):
        image = Image.open(self.filename_list[idx][0])
        label = self.filename_list[idx][1]
        image = self.transform(image)

        slide_name = os.path.basename(os.path.dirname(self.filename_list[idx][0]))
        return image, label, slide_name


    
# In[3] Main
if __name__ == '__main__':
    # Paths to where the dataset folder is stored 
    ''' CHANGE THIS '''
    Cpath = 'E:/NSCLC_Datasets'
    B1Name = 'NSCLC_3rd_TumorOnly_mag_20_color_Yes_for_shuffle_batch1_Full_ColorNorm'
    B2Name = 'NSCLC_3rd_TumorOnly_mag_20_color_Yes_for_shuffle_batch2_Full_ColorNorm'
    
    Result_dir = 'results'
    if not os.path.exists(Result_dir):
        os.makedirs(Result_dir)    
    save_dir = os.path.join(Result_dir)
    

    # Hyper Parameters and Paths
    model_abbr = 'Resenet18_' # Model Identifier
    magnif = '20'             # 20x Magnification Images
    num_epochs_list = [20,20,20] # Number of epochs for each train-test experiment

    tile_per_slide = 500 #
    
    # Train-test experiments
    nfold = 3    # Number of train-test experiments
    num_val = 20 # 20*4
    
    # Model Training Parameters
    batch_size = 200    # Batch size
    num_workers = 8     # Number of workers for data loading
    lr = 1e-3           # Learning rate
    weight_decay = 0.1  # Weight decay


    B1Path = os.path.join(Cpath, B1Name)
    B2Path = os.path.join(Cpath, B2Name)


    # In[] Prepare index files
    index1path = os.path.join(B1Path, 'Index')
    index2path = os.path.join(B2Path, 'Index')
    # read index csv files
    iminfo1 = pd.read_csv(os.path.join(index1path, 'summary_info.csv'))
    iminfo2 = pd.read_csv(os.path.join(index2path, 'summary_info.csv'))

    # get slide info and gt (label) from index files
    slide1 = iminfo1['slide']
    gt1 = iminfo1['gt']
    slide2 = iminfo2['slide']
    gt2 = iminfo2['gt']

    # bm: brain metastatisis, gt == 1; co: control, gt == 0
    slides_bm1 = []
    slides_co1 = []
    for i in range(len(slide1)):
        if gt1[i] == 1 and len(os.listdir(os.path.join(B1Path,'Data', slide1[i]))) >= tile_per_slide:
            slides_bm1.append(slide1[i])
        elif gt1[i] == 0 and len(os.listdir(os.path.join(B1Path,'Data', slide1[i]))) >= tile_per_slide:
            slides_co1.append(slide1[i])

    slides_bm2 = []
    slides_co2 = []
    for i in range(len(slide2)):
        if os.path.exists(os.path.join(B2Path,'Data', slide2[i])):
            if gt2[i] == 1  and len(os.listdir(os.path.join(B2Path,'Data', slide2[i]))) >= tile_per_slide:
                slides_bm2.append(slide2[i])
            elif gt2[i] == 0 and len(os.listdir(os.path.join(B2Path,'Data', slide2[i]))) >= tile_per_slide:
                slides_co2.append(slide2[i])
        
    # fix random seed

    for fold in range(3,8): # nfold
        print('Fold: ',fold)
        np.random.seed(fold)

        # Generate 0 to N-1 index for each slide
        idx_bm1 = np.arange(len(slides_bm1))
        idx_co1 = np.arange(len(slides_co1))
        idx_bm2 = np.arange(len(slides_bm2))
        idx_co2 = np.arange(len(slides_co2))

        # shuffle the index
        np.random.shuffle(idx_bm1)
        np.random.shuffle(idx_co1)
        np.random.shuffle(idx_bm2)
        np.random.shuffle(idx_co2)

        # Take num_val slides in each category for testing
        test_slides = []
        test_slides_bm1 = [slides_bm1[i] for i in idx_bm1[:num_val].tolist()]
        test_slides_co1 = [slides_co1[i] for i in idx_co1[:num_val].tolist()]
        test_slides_bm2 = [slides_bm2[i] for i in idx_bm2[:num_val].tolist()]
        test_slides_co2 = [slides_co2[i] for i in idx_co2[:num_val].tolist()]
        
        test_slides.extend(test_slides_bm1 + test_slides_co1 + test_slides_bm2 + test_slides_co2)


        # Labels one-to-one mapping to gt1 and gt2
        test_labels = []
        test_labels.extend(list(np.ones(num_val)) + 
                   list(np.zeros(num_val)) + 
                   list(np.ones(num_val)) + 
                   list(np.zeros(num_val)))

        # Take the rest slides in each category for training
        train_slides = []
        train_slides_bm1 = [slides_bm1[i] for i in idx_bm1[num_val:].tolist()]
        train_slides_co1 = [slides_co1[i] for i in idx_co1[num_val:].tolist()]
        train_slides_bm2 = [slides_bm2[i] for i in idx_bm2[num_val:].tolist()]
        train_slides_co2 = [slides_co2[i] for i in idx_co2[num_val:].tolist()]
        
        train_slides.extend(train_slides_bm1 + train_slides_co1 + train_slides_bm2 + train_slides_co2)

        # Labels one-to-one mapping to gt1 and gt2
        train_labels = []
        train_labels.extend(list(np.ones(len(slides_bm1)-num_val)) + 
                   list(np.zeros(len(slides_co1)-num_val)) + 
                   list(np.ones(len(slides_bm2)-num_val)) + 
                   list(np.zeros(len(slides_co2)-num_val)))
        
        # In[] Train and Test
        device = torch.device('cuda:0')
        # print(device)
    
        mean_r = 0.70395398               
        mean_g = 0.53016896
        mean_b = 0.69610734
        std_r = 0.15090187
        std_g = 0.16195891
        std_b = 0.12082885
        # print(mean_r,mean_g,mean_b,std_r,std_g,std_b)
        
        # Data transform
        # For training, random cropping, rotations and flippings are added as data augmentation
        # For testing, only center cropping will be used
        data_transforms = {
            'train': transforms.Compose([
                                    transforms.RandomCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip(),
                                    # transforms.RandomRotation(90),
                                    transforms.ToTensor(),
                                    transforms.Normalize([mean_r,mean_g,mean_b], [std_r,std_g,std_b])
                                    ]),
            'test': transforms.Compose([
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([mean_r,mean_g,mean_b], [std_r,std_g,std_b])
                                    ]),
            }
        

                
        # # Initialize pretrained model
        # if model_abbr == 'Resenet18_': 
        #     model_ft = models.resnet18(pretrained=True)
        #     num_ftrs = model_ft.fc.in_features
        #     model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 2))  


        # Load model weights
        model_ft = (torch.load(os.path.join(save_dir, 'model_' + str(fold) + '_whole_model.pt')))
        model_ft = model_ft.to(device)
        model_ft.eval()

        t = time.time()
        
        # Customized Dataset and corresponding Dataloader for current train-test experiment
        test_dataset = NSCLC_Dataset(B1Path, B2Path, data_transforms['test'], test_slides, test_labels, tile_per_slide)
        testloaders = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        testdataset_sizes =  len(test_dataset)
            

        # create a pandas dataframe to store the prediction results
        df = pd.DataFrame(columns=['slide', 'label', 'pred_score', 'pred_class'])


        running_corrects = 0.0
        # Iterate over data.
    
        model_ft.eval() # 5min
        for inputs, labels, slide_name in testloaders:
            # since = time.time()
            inputs = inputs.float().to(device)
            labels = labels.long().to(device)
            # print(slide_name)

            with torch.no_grad():
                outputs = model_ft(inputs)
                preds_score, preds_class = torch.max(outputs,1)

            # Calculate Loss and Accuracy
            running_corrects += torch.sum(preds_class == labels.data)

            # Append the prediction results to the dataframe
            for i in range(len(slide_name)):
                df = df.append({'slide': slide_name[i], 'label': labels[i].cpu().numpy(), 'pred_score': preds_score[i].cpu().numpy(), 'pred_class': preds_class[i].cpu().numpy()}, ignore_index=True)

            # elapsed = time.time() - since
            # print('Testing Time per batch: ',int(elapsed))
        
        epoch_acc = (running_corrects / testdataset_sizes).cpu()
        
        elapsed = time.time() - t
        print('Testing Time per epoch: ',int(elapsed))
        print('{} Acc: {:.4f} '.format('Test', epoch_acc))

        # Save the prediction results to csv file
        df.to_csv(os.path.join(save_dir, 'model_' + str(fold) + '_test_results.csv'), index=False)

        # In[] Calculate Slide Accuracy
    for fold in range(8):
        df = pd.read_csv(os.path.join(save_dir, 'model_' + str(fold) + '_test_results.csv'))
        slides = df['slide'].unique()
        slide_acc = []
        for slide in slides:
            df_slide = df[df['slide'] == slide]
            slide_acc.append(np.sum(df_slide['label'] == df_slide['pred_class']) / len(df_slide))  

        print('Fold: ',fold)
        print('Slide Acc: ', np.sum(np.sign(np.array(slide_acc)-0.5) /2+0.5) /len(slides) )


        