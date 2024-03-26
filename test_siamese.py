from process_audio import audio_single
import torch
import time
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc

def test_loop(network, dataloader, criterion, epochs, rcs, sr, threshold_vc, num_tubes, vowels, offset, device, pred_threshold):
    
    losses = []
    scores_0 = []
    scores_1 = []
    losses_0 = []
    losses_1 = []
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    labels = []
    euclidean_distances = []
    fpr_l = []
    tpr_l = []
    roc_auc = 0

    network.eval()
    network.to(device)
    criterion.to(device)
    
    with torch.no_grad():
        # batches
        for i, (audio1, phoneme1, text1, speaker1, rcs1, audio2, phoneme2, text2, speaker2, rcs2, label) in enumerate(dataloader):
            since = time.time()
            # print(f"phoneme1: {phoneme1[0]}")
            # print(f"phoneme2: {phoneme2}")
            # print(f"audio1: {audio1}")
            # print(f"audio2: {audio2}")
            # print(f"Speaker1: {speaker1}")
            # print(f"Speaker2: {speaker2}")
            # print("for texts:")
            # print(f"text1: {text1}")
            # print(f"text2: {text2}")
            # print(f"label: {label}")

            # layer_1 = audio_single(rcs, epochs, sr, threshold_vc, num_tubes, audio1, phoneme1, vowels, offset)
            # layer_2 = audio_single(rcs, epochs, sr, threshold_vc, num_tubes, audio2, phoneme2, vowels, offset)
            # # print(f"layer_1 shape: {layer_1.shape}")
            # # print(f"layer_1: {layer_1}")

            # layer_1_tensor = torch.from_numpy(layer_1).float().to(device)  # Convert to tensor and ensure dtype is float
            # layer_2_tensor = torch.from_numpy(layer_2).float().to(device)  # Convert to tensor
            # label = torch.from_numpy(np.array(label)).float().to(device)

            # # Initialize two (1, 1, 320) tensors
            # # layer_1_tensor = torch.randn((1, 1, 320))
            # # layer_2_tensor = torch.randn((1, 1, 320))

            # # Make it 3D if the tensor is 1D (Batch size 1)
            # if len(layer_1_tensor.shape) == 1:
            #     layer_1_tensor = layer_1_tensor.unsqueeze(0).unsqueeze(0)
            #     layer_2_tensor = layer_2_tensor.unsqueeze(0).unsqueeze(0)

            # Convert rcs1 and rcs 2 to tensors
            rcs1 = rcs1.float().to(device)
            rcs2 = rcs2.float().to(device)
            label = label.to(device)

            # print("rcs1: ", rcs1)
            # print("rcs2: ", rcs2)
            
            rcs1 = rcs1.unsqueeze(1)
            rcs2 = rcs2.unsqueeze(1)

            # Pass in the two layers
            output1, output2 = network(rcs1, rcs2)
            output1 = output1.to(device)
            output2 = output2.to(device)

            # Loss
            loss = criterion(output1, output2, label)
            losses.append(loss.item())

            euclidean_distance = F.pairwise_distance(output1, output2)
            # print(f"Prediction: {euclidean_distance}")
            # print(f"label: {label}")

            # print(f"Batch {i} loss: {loss.item()}")
            # print(f"Time elapsed: {time.time() - since}s")

            # print(loss.item())

            for l in label:
                labels.append(l.cpu().numpy())
            for e in euclidean_distance:
                euclidean_distances.append(e.cpu().numpy())

            for i in range(len(label)):
                if label[i] == 0:
                    scores_0.append(euclidean_distance[i])
                    losses_0.append(loss.item())
                    if euclidean_distance[i] < pred_threshold:
                        tp += 1
                    else:
                        fn += 1
                else:
                    scores_1.append(euclidean_distance[i])
                    losses_1.append(loss.item())
                    if euclidean_distance[i] < pred_threshold:
                        fp += 1
                    else:
                        tn += 1

    # Avoid division by 0
    if tp == 0:
        tp = 1
    if fp == 0:
        fp = 1
    if tn == 0:
        tn = 1
    if fn == 0:
        fn = 1

    # Analytics
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f = 2 * (precision * recall) / (precision + recall)
    fpr = fp / (fp + tn)

    print(f"For {len(scores_0)} samples with label 0")
    print(f"Average score: {sum(scores_0) / len(scores_0)}")
    print(f"Average loss: {sum(losses_0) / len(scores_0)}")
    print(f"True Positive: {tp}")
    print(f"False Positive: {fp}")

    print(f"For {len(scores_1)} samples with label 1")
    print(f"Average score: {sum(scores_1) / len(scores_1)}")
    print(f"Average loss: {sum(losses_1) / len(scores_1)}")
    print(f"True Negative: {tn}")
    print(f"False Negative: {fn}")

    print("--------------")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"FPR: {fpr}")
    print(f"Accuracy: {accuracy}")
    print(f"F-score: {f}")

    fpr_l, tpr_l, thresholds = roc_curve(labels, euclidean_distances)
    roc_auc = auc(fpr_l, tpr_l)

    print("Test Done")

    return fpr_l, tpr_l, thresholds, roc_auc, precision, recall, accuracy, f

def test_loop_agg(network, dataloader, criterion, epochs, rcs, sr, threshold_vc, num_tubes, vowels, offset, device, pred_threshold):
    
    losses = []
    scores_0 = []
    scores_1 = []
    losses_0 = []
    losses_1 = []
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    labels = []
    euclidean_distances = []
    fpr_l = []
    tpr_l = []
    roc_auc = 0

    network.eval()
    network.to(device)
    criterion.to(device)
    
    with torch.no_grad():
        # batches
        for i, (speaker1, rcs1, speaker2, rcs2, label) in enumerate(dataloader):
            since = time.time()
            # print(f"phoneme1: {phoneme1[0]}")
            # print(f"phoneme2: {phoneme2}")
            # print(f"audio1: {audio1}")
            # print(f"audio2: {audio2}")
            # print(f"Speaker1: {speaker1}")
            # print(f"Speaker2: {speaker2}")
            # print(f"rcs1: {rcs1}")
            # print(f"rcs2: {rcs2}")
            # print(f"label: {label}")
            # print("for texts:")
            # print(f"text1: {text1}")
            # print(f"text2: {text2}")

            # layer_1 = audio_single(rcs, epochs, sr, threshold_vc, num_tubes, audio1, phoneme1, vowels, offset)
            # layer_2 = audio_single(rcs, epochs, sr, threshold_vc, num_tubes, audio2, phoneme2, vowels, offset)
            # # print(f"layer_1 shape: {layer_1.shape}")
            # # print(f"layer_1: {layer_1}")

            # layer_1_tensor = torch.from_numpy(layer_1).float().to(device)  # Convert to tensor and ensure dtype is float
            # layer_2_tensor = torch.from_numpy(layer_2).float().to(device)  # Convert to tensor
            # label = torch.from_numpy(np.array(label)).float().to(device)

            # # Initialize two (1, 1, 320) tensors
            # # layer_1_tensor = torch.randn((1, 1, 320))
            # # layer_2_tensor = torch.randn((1, 1, 320))

            # # Make it 3D if the tensor is 1D (Batch size 1)
            # if len(layer_1_tensor.shape) == 1:
            #     layer_1_tensor = layer_1_tensor.unsqueeze(0).unsqueeze(0)
            #     layer_2_tensor = layer_2_tensor.unsqueeze(0).unsqueeze(0)

            # Convert rcs1 and rcs 2 to tensors
            rcs1 = rcs1.float().to(device)
            rcs2 = rcs2.float().to(device)
            label = label.to(device)

            # print("rcs1: ", rcs1)
            # print("rcs2: ", rcs2)
            
            rcs1 = rcs1.unsqueeze(1)
            rcs2 = rcs2.unsqueeze(1)

            # Pass in the two layers
            output1, output2 = network(rcs1, rcs2)
            output1 = output1.to(device)
            output2 = output2.to(device)

            # Loss
            loss = criterion(output1, output2, label)
            losses.append(loss.item())

            euclidean_distance = F.pairwise_distance(output1, output2)
            # print(f"Prediction: {euclidean_distance}")
            # print(f"label: {label}")

            # print(f"Batch {i} loss: {loss.item()}")
            # print(f"Time elapsed: {time.time() - since}s")

            # print(loss.item())

            for l in label:
                labels.append(l.cpu().numpy())
            for e in euclidean_distance:
                euclidean_distances.append(e.cpu().numpy())

            for i in range(len(label)):
                if label[i] == 0:
                    scores_0.append(euclidean_distance[i])
                    losses_0.append(loss.item())
                    if euclidean_distance[i] < pred_threshold:
                        tp += 1
                    else:
                        fn += 1
                else:
                    scores_1.append(euclidean_distance[i])
                    losses_1.append(loss.item())
                    if euclidean_distance[i] < pred_threshold:
                        fp += 1
                    else:
                        tn += 1

    # Avoid division by 0
    if tp == 0:
        tp = 1
    if fp == 0:
        fp = 1
    if tn == 0:
        tn = 1
    if fn == 0:
        fn = 1

    # Analytics
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f = 2 * (precision * recall) / (precision + recall)
    fpr = fp / (fp + tn)

    print(f"For {len(scores_0)} samples with label 0")
    print(f"Average score: {sum(scores_0) / len(scores_0)}")
    print(f"Average loss: {sum(losses_0) / len(scores_0)}")
    print(f"True Positive: {tp}")
    print(f"False Positive: {fp}")

    print(f"For {len(scores_1)} samples with label 1")
    print(f"Average score: {sum(scores_1) / len(scores_1)}")
    print(f"Average loss: {sum(losses_1) / len(scores_1)}")
    print(f"True Negative: {tn}")
    print(f"False Negative: {fn}")

    print("--------------")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"FPR: {fpr}")
    print(f"Accuracy: {accuracy}")
    print(f"F-score: {f}")

    fpr_l, tpr_l, thresholds = roc_curve(labels, euclidean_distances)
    roc_auc = auc(fpr_l, tpr_l)

    print("Test Done")

    return fpr_l, tpr_l, thresholds, roc_auc, precision, recall, accuracy, f
    

