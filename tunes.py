# Hyperparameter Tunning
from train_siamese import train_loop, train_loop_agg, train_loop_agg_model, train_loop_agg_margin
from test_siamese import test_loop, test_loop_agg
from siamese import ContrastiveLoss
import torch
import numpy as np

from torch.utils.data import DataLoader
from torch import optim
from sklearn.model_selection import KFold, StratifiedKFold

from draw import plot_roc, plot_roc_agg, plot_roc_agg_model

def margin_threshold_siamese(margins, network, dataloader_train, dataloader_test, optimizer, epochs, rcs, sr, 
                             threshold_cv, num_tubes, vowels, offset, device):

    for margin in margins:
        print(f"For margin {margin}")

        # Need new training
        train_loop(network, dataloader_train, ContrastiveLoss(margin=margin), optimizer, epochs, rcs, sr, threshold_cv, num_tubes, vowels, offset, 
                   device, margin)
        # print("After training loop")
        
        # Use pretrained model
        # state_dict = torch.load(f"models/siamese_margin_{margin}.pth")
        # network.load_state_dict(state_dict)
        

        fpr_l, tpr_l, thresholds, roc_auc, _, _, _, _ = test_loop(network, dataloader_test, ContrastiveLoss(margin=margin), epochs, rcs, sr, 
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
        

        fpr_l, tpr_l, thresholds, roc_auc, _, _, _, _ = test_loop_agg(network, dataloader_test, ContrastiveLoss(margin=margin), epochs, rcs, sr, 
                                          threshold_cv, num_tubes, vowels, offset, device, 1)
        
        path = './figures/roc/final'
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
        print(f"Given AUC: {roc_auc}")

        test_loop_agg(network, dataloader_test, ContrastiveLoss(margin=margin), epochs, rcs, sr, threshold_cv, num_tubes, 
                  vowels, offset, device, best_threshold)

        print("")

# margin_threshold_siamese_agg for model
def margin_threshold_siamese_agg_model(margins, network, dataloader_train, dataloader_test, optimizer, epochs, rcs, sr, 
                             threshold_cv, num_tubes, vowels, offset, device, agg_num, model_name):

    for margin in margins:
        print(f"For margin {margin}")

        # Need new training
        train_loop_agg_model(network, dataloader_train, ContrastiveLoss(margin=margin), optimizer, epochs, rcs, sr, threshold_cv, num_tubes, vowels, offset, 
                   device, margin, agg_num, model_name)
        # print("After training loop")
        
        # Use pretrained model
        # state_dict = torch.load(f"models/agg/agg_{agg_num}_siamese_margin_{margin}.pth")
        # network.load_state_dict(state_dict)
        

        fpr_l, tpr_l, thresholds, roc_auc, _, _, _, _ = test_loop_agg(network, dataloader_test, ContrastiveLoss(margin=margin), epochs, rcs, sr, 
                                          threshold_cv, num_tubes, vowels, offset, device, 1)
        
        path = './figures/roc/ablation'
        plot_roc_agg_model(fpr_l, tpr_l, roc_auc, margin, path, agg_num, model_name)
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
        print(f"Given AUC: {roc_auc}")

        _, _, _, _, precision, recall, accuracy, f_score = test_loop_agg(network, dataloader_test, ContrastiveLoss(margin=margin), epochs, rcs, sr, threshold_cv, num_tubes, 
                  vowels, offset, device, best_threshold)

        print("")

        return roc_auc, precision, recall, accuracy, f_score
    
# single margin
def siamese_agg_model(margin, network, dataloader_train, dataloader_test, optimizer, epochs, rcs, sr, 
                             threshold_cv, num_tubes, vowels, offset, device, agg_num, model_name):

    # Need new training
    train_loop_agg_model(network, dataloader_train, ContrastiveLoss(margin=margin), optimizer, epochs, rcs, sr, threshold_cv, num_tubes, vowels, offset, 
                device, margin, agg_num, model_name)
    # print("After training loop")
    
    # Use pretrained model
    # state_dict = torch.load(f"models/agg/agg_{agg_num}_siamese_margin_{margin}.pth")
    # network.load_state_dict(state_dict)
    

    fpr_l, tpr_l, thresholds, roc_auc, precision, recall, accuracy, f_score = test_loop_agg(network, dataloader_test, ContrastiveLoss(margin=margin), epochs, rcs, sr, 
                                        threshold_cv, num_tubes, vowels, offset, device, 1)

    path = './figures/final'
    plot_roc_agg_model(fpr_l, tpr_l, roc_auc, margin, path, agg_num, model_name)
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
    print(f"Given AUC: {roc_auc}")

    fpr_l, tpr_l, thresholds, roc_auc, precision, recall, accuracy, f_score = test_loop_agg(network, dataloader_test, ContrastiveLoss(margin=margin), epochs, rcs, sr, threshold_cv, num_tubes, 
                vowels, offset, device, best_threshold)

    print("")

    return roc_auc, precision, recall, accuracy, f_score

# single margin, non-aggregated
def siamese_model_train(margin, network, dataloader_train, dataloader_test, optimizer, epochs, rcs, sr, 
                             threshold_cv, num_tubes, vowels, offset, device, agg_num, model_name):

    # Need new training
    train_loop(network, dataloader_train, ContrastiveLoss(margin=margin), optimizer, epochs, rcs, sr, threshold_cv, num_tubes, vowels, offset, 
                device, margin)
    # print("After training loop")
    
    # Use pretrained model
    # state_dict = torch.load(f"models/final/siamese_margin_{margin}_balanced.pth")
    # network.load_state_dict(state_dict)
    

    fpr_l, tpr_l, thresholds, roc_auc, precision, recall, accuracy, f_score = test_loop(network, dataloader_test, ContrastiveLoss(margin=margin), epochs, rcs, sr, 
                                        threshold_cv, num_tubes, vowels, offset, device, 1)

    path = './figures/final'
    plot_roc_agg_model(fpr_l, tpr_l, roc_auc, margin, path, agg_num, model_name)
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
    print(f"Given AUC: {roc_auc}")

    fpr_l, tpr_l, thresholds, roc_auc, precision, recall, accuracy, f_score = test_loop(network, dataloader_test, ContrastiveLoss(margin=margin), epochs, rcs, sr, threshold_cv, num_tubes, 
                vowels, offset, device, best_threshold)

    print("")

    return roc_auc, precision, recall, accuracy, f_score

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

# single margin for different num_aggs
def margin_threshold_multiple_single_margin(margin, network, optimizer, epochs, rcs, sr, threshold_cv, num_tubes, vowels, offset, 
                              device, train_tests_paired, batch_size, model_name):
    
    rcs_org = rcs
    
    for train_test_paired in train_tests_paired:
        agg_num, train, test = train_test_paired

        print(f"For agg {agg_num}")
        print(f"train data length: {len(train)}")
        print(f"test data length: {len(test)}")

        dataloader_train = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8)
        dataloader_test = DataLoader(test, batch_size=batch_size, shuffle=True, num_workers=8)

        siamese_agg_model(margin, network, dataloader_train, dataloader_test, optimizer, epochs, rcs, sr, 
                            threshold_cv, num_tubes, vowels, offset, device, agg_num, model_name)
        # Keep the same starting point
        rcs = rcs_org

# Call margin_threshold_siamese_agg for different model architectures
def margin_threshold_multiple_models(margins, models, model_names, epochs, rcs, sr, threshold_cv, num_tubes, vowels, offset, 
                              device, batch_size, agg_num, train_b, test_b):
    
    dataloader_train = DataLoader(train_b, batch_size=batch_size, shuffle=True, num_workers=8)
    dataloader_test = DataLoader(test_b, batch_size=batch_size, shuffle=True, num_workers=8)

    for i in range(len(models)):
        model = models[i]
        model_name = model_names[i]
        print(f"For model {model_name}")

        margin_threshold_siamese_agg_model(margins, model, dataloader_train, dataloader_test, optim.Adam(model.parameters(), lr=0.0005), epochs, rcs, sr, 
                            threshold_cv, num_tubes, vowels, offset, device, agg_num, model_name)  
        
# run cross validation on various model architectures
def margin_threshold_multiple_models_cv(margin, models, model_names, epochs, rcs, sr, threshold_cv, num_tubes, vowels, offset, 
                              device, batch_size, agg_num, train_b):
    
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    labels = [datapoint[4] for datapoint in train_b]
    # print(labels)

    fold_auc = {model_name: [] for model_name in model_names}
    fold_precision = {model_name: [] for model_name in model_names}
    fold_recall = {model_name: [] for model_name in model_names}
    fold_accuracy = {model_name: [] for model_name in model_names}
    fold_f_score = {model_name: [] for model_name in model_names}

    for i in range(len(models)):
        
        model = models[i]
        model_name = model_names[i]
        print(f"For model {model_name}")

        # cross validation
        for train_index, test_index in kf.split(train_b, labels):
            train_fold = [train_b[i] for i in train_index]
            test_fold = [train_b[i] for i in test_index]

            # print(train_fold[:5])
            # print(test_fold[:5])

            dataloader_train = DataLoader(train_fold, batch_size=batch_size, shuffle=True, num_workers=8)
            dataloader_test = DataLoader(test_fold, batch_size=batch_size, shuffle=True, num_workers=8)

            auc, precision, recall, accuracy, f_score = siamese_agg_model(margin, model, dataloader_train, dataloader_test, optim.Adam(model.parameters(), lr=0.0005), epochs, rcs, sr, 
                                threshold_cv, num_tubes, vowels, offset, device, agg_num, model_name)
            
            fold_auc[model_name].append(auc)
            fold_precision[model_name].append(precision)
            fold_recall[model_name].append(recall)
            fold_accuracy[model_name].append(accuracy)
            fold_f_score[model_name].append(f_score)

    for model_name in model_names:
        print(f"For model {model_name}")
        auc_avg = np.mean(fold_auc[model_name])
        precision_avg = np.mean(fold_precision[model_name])
        recall_avg = np.mean(fold_recall[model_name])
        accuracy_avg = np.mean(fold_accuracy[model_name])
        f_score_avg = np.mean(fold_f_score[model_name])

        print(f"aucs: {fold_auc[model_name]}")
        print(f"precisions: {fold_precision[model_name]}")
        print(f"recalls: {fold_recall[model_name]}")
        print(f"accuracies: {fold_accuracy[model_name]}")
        print(f"f-scores: {fold_f_score[model_name]}")

        print(f"Average auc: {auc_avg}")
        print(f"Average precision: {precision_avg}")
        print(f"Average recall: {recall_avg}")
        print(f"Average accuracy: {accuracy_avg}")
        print(f"Average f-score: {f_score_avg}")
        
# run cross validation on various model architectures
def margin_threshold_multiple_models_cv_margins(margins, model, model_name, epochs, rcs, sr, threshold_cv, num_tubes, vowels, offset, 
                              device, batch_size, agg_num, train_b):
    
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    labels = [datapoint[4] for datapoint in train_b]
    # print(labels)
    # print(train_b[0])
    # print(len(train_b[0]))

    fold_auc = {margin: [] for margin in margins}
    fold_precision = {margin: [] for margin in margins}
    fold_recall = {margin: [] for margin in margins}
    fold_accuracy = {margin: [] for margin in margins}
    fold_f_score = {margin: [] for margin in margins}

    for i in range(len(margins)):
        
        margin = margins[i]

        print(f"For margin {margin}")

        # cross validation
        for train_index, test_index in kf.split(train_b, labels):
            train_fold = [train_b[i] for i in train_index]
            test_fold = [train_b[i] for i in test_index]

            # print(train_fold[:5])
            # print(test_fold[:5])

            dataloader_train = DataLoader(train_fold, batch_size=batch_size, shuffle=True, num_workers=8)
            dataloader_test = DataLoader(test_fold, batch_size=batch_size, shuffle=True, num_workers=8)

            auc, precision, recall, accuracy, f_score = siamese_agg_model(margin, model, dataloader_train, dataloader_test, optim.Adam(model.parameters(), lr=0.0005), epochs, rcs, sr, 
                                threshold_cv, num_tubes, vowels, offset, device, agg_num, model_name)
            
            fold_auc[margin].append(auc)
            fold_precision[margin].append(precision)
            fold_recall[margin].append(recall)
            fold_accuracy[margin].append(accuracy)
            fold_f_score[margin].append(f_score)

    for margin in margins:
        print(f"For margin {margin}")
        auc_avg = np.mean(fold_auc[margin])
        precision_avg = np.mean(fold_precision[margin])
        recall_avg = np.mean(fold_recall[margin])
        accuracy_avg = np.mean(fold_accuracy[margin])
        f_score_avg = np.mean(fold_f_score[margin])

        print(f"aucs: {fold_auc[margin]}")
        print(f"precisions: {fold_precision[margin]}")
        print(f"recalls: {fold_recall[margin]}")
        print(f"accuracies: {fold_accuracy[margin]}")
        print(f"f-scores: {fold_f_score[margin]}")

        print(f"Average auc: {auc_avg}")
        print(f"Average precision: {precision_avg}")
        print(f"Average recall: {recall_avg}")
        print(f"Average accuracy: {accuracy_avg}")
        print(f"Average f-score: {f_score_avg}")