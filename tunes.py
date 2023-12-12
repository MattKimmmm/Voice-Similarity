# Hyperparameter Tunning
from train_siamese import train_loop, train_loop_agg
from test_siamese import test_loop, test_loop_agg
from siamese import ContrastiveLoss
import torch
import numpy as np

from draw import plot_roc

def margin_threshold_siamese(margins, network, dataloader_train, dataloader_test, optimizer, epochs, rcs, sr, 
                             threshold_cv, num_tubes, vowels, offset, device):

    for margin in margins:
        print(f"For margin {margin}")

        # Need new training
        train_loop(network, dataloader_train, ContrastiveLoss(margin=margin), optimizer, epochs, rcs, sr, threshold_cv, num_tubes, vowels, offset, 
                   device, margin)
        
        # Use pretrained model

        # state_dict = torch.load(f"models/dropout_hidden_margin_{margin}.pth")
        # network.load_state_dict(state_dict)
        

        fpr_l, tpr_l, thresholds, roc_auc = test_loop(network, dataloader_test, ContrastiveLoss(margin=margin), epochs, rcs, sr, 
                                          threshold_cv, num_tubes, vowels, offset, device, 1)
        
        path = './figures/roc'
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
                             threshold_cv, num_tubes, vowels, offset, device):

    for margin in margins:
        print(f"For margin {margin}")

        # Need new training
        train_loop(network, dataloader_train, ContrastiveLoss(margin=margin), optimizer, epochs, rcs, sr, threshold_cv, num_tubes, vowels, offset, 
                   device, margin)
        # print("After training loop")
        
        # Use pretrained model

        # state_dict = torch.load(f"models/b_siamese_margin_{margin}.pth")
        # network.load_state_dict(state_dict)
        

        fpr_l, tpr_l, thresholds, roc_auc = test_loop(network, dataloader_test, ContrastiveLoss(margin=margin), epochs, rcs, sr, 
                                          threshold_cv, num_tubes, vowels, offset, device, 1)
        
        path = './figures/roc'
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



