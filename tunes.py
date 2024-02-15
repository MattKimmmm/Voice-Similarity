# Hyperparameter Tunning
from train_siamese import train_loop, train_loop_agg
from test_siamese import test_loop, test_loop_agg
from siamese import ContrastiveLoss
import torch
import numpy as np
from torch.utils.data import DataLoader

from draw import plot_roc, plot_roc_agg

def margin_threshold_siamese(margins, network, dataloader_train, dataloader_test, optimizer, epochs, rcs, sr, 
                             threshold_cv, num_tubes, vowels, offset, device):

    for margin in margins:
        print(f"For margin {margin}")

        # Need new training
        train_loop(network, dataloader_train, ContrastiveLoss(margin=margin), optimizer, epochs, rcs, sr, threshold_cv, num_tubes, vowels, offset, 
                   device, margin)
        # print("After training loop")
        
        # Use pretrained model
        # state_dict = torch.load(f"models/full/siamese_margin_{margin}.pth")
        # network.load_state_dict(state_dict)
        

        fpr_l, tpr_l, thresholds, roc_auc = test_loop(network, dataloader_test, ContrastiveLoss(margin=margin), epochs, rcs, sr, 
                                          threshold_cv, num_tubes, vowels, offset, device, 1)
        
        path = './figures/roc/full'
        plot_roc(fpr_l, tpr_l, roc_auc, margin, path)
        youdens = []

        # Find the best threshold value
        for i in range(len(fpr_l)):
            fpr = fpr_l[i]
            tpr = tpr_l[i]
            threshold = thresholds[i]

            specificity = 1 - fpr
            youden = tpr + specificity - 1
            youdens.append(youden)

        max_index = np.argmax(youdens)
        best_threshold = thresholds[max_index]
        best_youden = youdens[max_index]

        print(f"Threshold {best_threshold} was selected with Youden's J of {best_youden}")

        test_loop(network, dataloader_test, ContrastiveLoss(margin=margin), epochs, rcs, sr, threshold_cv, num_tubes, 
                  vowels, offset, device, best_threshold)

        print("")

def margin_threshold_siamese_agg(margins, network, dataloader_train, dataloader_test, optimizer, epochs, rcs, sr, 
                             threshold_cv, num_tubes, vowels, offset, device, agg_num):

    for margin in margins:
        print(f"For margin {margin}")

        # Need new training
        train_loop_agg(network, dataloader_train, ContrastiveLoss(margin=margin), optimizer, epochs, rcs, sr, threshold_cv, num_tubes, vowels, offset, 
                   device, margin, agg_num)
        # print("After training loop")
        
        # Use pretrained model
        # state_dict = torch.load(f"models/agg/agg_{agg_num}_siamese_margin_{margin}.pth")
        # network.load_state_dict(state_dict)
        

        fpr_l, tpr_l, thresholds, roc_auc = test_loop_agg(network, dataloader_test, ContrastiveLoss(margin=margin), epochs, rcs, sr, 
                                          threshold_cv, num_tubes, vowels, offset, device, 1)
        
        path = './figures/roc/agg'
        plot_roc_agg(fpr_l, tpr_l, roc_auc, margin, path, agg_num)
        youdens = []

        # Find the best threshold value
        for i in range(len(fpr_l)):
            fpr = fpr_l[i]
            tpr = tpr_l[i]
            threshold = thresholds[i]

            specificity = 1 - fpr
            youden = tpr + specificity - 1
            youdens.append(youden)

        max_index = np.argmax(youdens)
        best_threshold = thresholds[max_index]
        best_youden = youdens[max_index]

        print(f"Threshold {best_threshold} was selected with Youden's J of {best_youden}")

        test_loop_agg(network, dataloader_test, ContrastiveLoss(margin=margin), epochs, rcs, sr, threshold_cv, num_tubes, 
                  vowels, offset, device, best_threshold)

        print("")

# Call margin_threshold_siames_agg for each agg_num
def margin_threshold_multiple(margins, network, optimizer, epochs, rcs, sr, threshold_cv, num_tubes, vowels, offset, 
                              device, train_tests_paired, batch_size):
    
    rcs_org = rcs
    
    for train_test_paired in train_tests_paired:
        agg_num, train, test = train_test_paired
        print(f"For agg {agg_num}")
        print(f"train data length: {len(train)}")
        print(f"test data length: {len(test)}")

        dataloader_train = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8)
        dataloader_test = DataLoader(test, batch_size=batch_size, shuffle=True, num_workers=8)

        margin_threshold_siamese_agg(margins, network, dataloader_train, dataloader_test, optimizer, epochs, rcs, sr, 
                             threshold_cv, num_tubes, vowels, offset, device, agg_num)
        # Keep the same starting point
        rcs = rcs_org

        

