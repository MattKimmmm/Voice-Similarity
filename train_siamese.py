from process_audio import audio_single
from utils import reset_model_params, reinitialize_model
import torch
import time
import numpy as np

def train_loop(network, dataloader, criterion, optimizer, epochs, rcs, sr, threshold_vc, num_tubes, vowels, offset, device, margin):
    loss_prev = np.inf
    
    # reset_model_params(network)
    reinitialize_model(network)

    network.to(device)
    criterion.to(device)

    for epoch in range(epochs):
        since = time.time()

        print(f"Epoch {epoch + 1}\n-------------------------------")
        # batches
        for i, (audio1, phoneme1, text1, speaker1, rcs1, audio2, phoneme2, text2, speaker2, rcs2, label) in enumerate(dataloader):
            # Initialize two (1, 1, 320) tensors
            # layer_1_tensor = torch.randn((1, 1, 320))
            # layer_2_tensor = torch.randn((1, 1, 320))

            # Convert rcs1 and rcs 2 to tensors
            rcs1 = rcs1.float().to(device)
            rcs2 = rcs2.float().to(device)
            label = label.to(device)

            # print("rcs1: ", rcs1)
            # print("rcs2: ", rcs2)

            rcs1 = rcs1.unsqueeze(1)
            rcs2 = rcs2.unsqueeze(1)

            # print(f"rcs1 shape after : {rcs1.shape}")
            # print(f"rcs2 shape after : {rcs2.shape}")
            # print(f"phoneme1: {phoneme1}")
            # print(f"phoneme2: {phoneme2}")
            # print(f"audio1: {audio1}")
            # print(f"audio2: {audio2}")
            # print(f"Speaker1: {speaker1}")
            # print(f"Speaker2: {speaker2}")
            # print("for texts:")
            # print(f"text1: {text1}")
            # print(f"text2: {text2}")

            # Forward pass
            optimizer.zero_grad()

            # Pass in the two layers
            output1, output2 = network(rcs1, rcs2)
            output1 = output1.to(device)
            output2 = output2.to(device)

            # Loss
            loss = criterion(output1, output2, label)

            # Backpropagate
            loss.backward()
            optimizer.step()

            # current_it = i + 1
            # print(f"Current iteration {current_it} / {iterations}")
            # if i % 1000 == 0:
            #     print(f"Current iteration {i + 1}")
            #     print(f"1000 Iterations took {time.time() - since}")

        print("loss_prev: ", loss_prev)
        print("loss: ", loss.item())
        # If loss improvement is less than threshold, stop training
        if abs(loss_prev - loss.item()) < threshold_vc:
            print(f"Loss improvement less than threshold, stop training at epoch {epoch}, batch {i}")
            break

        loss_prev = loss.item()
        print(f"Epoch {epoch} loss: {loss.item()}")
        print(f"Time elapsed: {time.time() - since}s")
    
    # save the model
    torch.save(network.state_dict(), f'models/final/siamese_margin_{margin}_balanced.pth')
    print("Training Done")

def train_loop_agg(network, dataloader, criterion, optimizer, epochs, rcs, sr, threshold_vc, num_tubes, vowels, offset, 
                   device, margin, agg_num):
    loss_prev = np.inf

    # reset_model_params(network)
    reinitialize_model(network)

    network.to(device)
    criterion.to(device)

    for epoch in range(epochs):
        since = time.time()

        print(f"Epoch {epoch + 1}\n-------------------------------")
        # batches
        for i, (speaker1, rcs1, speaker2, rcs2, label) in enumerate(dataloader):
            # print("For")

            # Initialize two (1, 1, 320) tensors
            # layer_1_tensor = torch.randn((1, 1, 320))
            # layer_2_tensor = torch.randn((1, 1, 320))

            # Convert rcs1 and rcs 2 to tensors
            # print("rcs1 type:", type(rcs1))
            # print("rcs1 content:", rcs1)
            # print(f"rcs1 shape: {rcs1.shape}")
            rcs1 = rcs1.float().to(device)
            rcs2 = rcs2.float().to(device)
            label = label.to(device)

            # print("rcs1: ", rcs1)
            # print("rcs2: ", rcs2)

            # print(f"rcs1 before unsqueeze: {rcs1.shape}")
            # print(f"rcs2 before unsqueeze: {rcs2.shape}")
            rcs1 = rcs1.unsqueeze(1)
            rcs2 = rcs2.unsqueeze(1)

            # print(f"rcs1 after shape: {rcs1.shape}")
            # print(f"rcs2 after shape: {rcs2.shape}")
            # print(f"phoneme1: {phoneme1}")
            # print(f"phoneme2: {phoneme2}")
            # print(f"audio1: {audio1}")
            # print(f"audio2: {audio2}")
            # print(f"Speaker1: {speaker1}")
            # print(f"Speaker2: {speaker2}")
            # print("for texts:")
            # print(f"text1: {text1}")
            # print(f"text2: {text2}")

            # Forward pass
            optimizer.zero_grad()

            # Pass in the two layers
            output1, output2 = network(rcs1, rcs2)
            output1 = output1.to(device)
            output2 = output2.to(device)

            # Loss
            loss = criterion(output1, output2, label)

            # Backpropagate
            loss.backward()
            optimizer.step()

        print("loss_prev: ", loss_prev)
        print("loss: ", loss.item())
        # If loss improvement is less than threshold, stop training
        if abs(loss_prev - loss.item()) < threshold_vc:
            print(f"Loss improvement less than threshold, stop training at epoch {epoch}, batch {i}")
            break

        loss_prev = loss.item()
        print(f"Epoch {epoch} loss: {loss.item()}")
        print(f"Time elapsed: {time.time() - since}s")
    
    # save the model
    torch.save(network.state_dict(), f'models/final/agg_{agg_num}_siamese_margin_{margin}.pth')
    print("Training Done")
    
# train_loop_agg + model name
def train_loop_agg_model(network, dataloader, criterion, optimizer, epochs, rcs, sr, threshold_vc, num_tubes, vowels, offset, 
                   device, margin, agg_num, model_name):
    loss_prev = np.inf

    # reset_model_params(network)
    reinitialize_model(network)

    network.to(device)
    criterion.to(device)
    count = 0

    for epoch in range(epochs):
        since = time.time()

        print(f"Epoch {epoch + 1}\n-------------------------------")
        # batches
        for i, (speaker1, rcs1, speaker2, rcs2, label) in enumerate(dataloader):
            # print("For")

            # Initialize two (1, 1, 320) tensors
            # layer_1_tensor = torch.randn((1, 1, 320))
            # layer_2_tensor = torch.randn((1, 1, 320))

            # Convert rcs1 and rcs 2 to tensors
            # print("rcs1 type:", type(rcs1))
            # print("rcs1 content:", rcs1)
            # print(f"rcs1 shape: {rcs1.shape}")
            rcs1 = rcs1.float().to(device)
            rcs2 = rcs2.float().to(device)
            label = label.to(device)

            # print("rcs1: ", rcs1)
            # print("rcs2: ", rcs2)

            # print(f"rcs1 before unsqueeze: {rcs1.shape}")
            # print(f"rcs2 before unsqueeze: {rcs2.shape}")
            rcs1 = rcs1.unsqueeze(1)
            rcs2 = rcs2.unsqueeze(1)

            # print(f"rcs1 after shape: {rcs1.shape}")
            # print(f"rcs2 after shape: {rcs2.shape}")
            # print(f"phoneme1: {phoneme1}")
            # print(f"phoneme2: {phoneme2}")
            # print(f"audio1: {audio1}")
            # print(f"audio2: {audio2}")
            # print(f"Speaker1: {speaker1}")
            # print(f"Speaker2: {speaker2}")
            # print("for texts:")
            # print(f"text1: {text1}")
            # print(f"text2: {text2}")

            # Forward pass
            optimizer.zero_grad()

            # Pass in the two layers
            output1, output2 = network(rcs1, rcs2)
            output1 = output1.to(device)
            output2 = output2.to(device)

            # Loss
            loss = criterion(output1, output2, label)

            # Backpropagate
            loss.backward()
            optimizer.step()

        print("loss_prev: ", loss_prev)
        print("loss: ", loss.item())
        # If loss improvement is less than threshold, stop training
        if loss_prev - loss.item() < threshold_vc:
            count += 1
        
        if count == 5:
            print(f"Loss improvement less than threshold for 5 times, stop training at epoch {epoch}, batch {i}")
            break

        loss_prev = loss.item()
        print(f"Epoch {epoch} loss: {loss.item()}")
        print(f"Time elapsed: {time.time() - since}s")
    
    # save the model
    torch.save(network.state_dict(), f'models/final/full/agg_{agg_num}_model_{model_name}_siamese_margin_{margin}_balanced.pth')
    print("Training Done")

# train_loop_agg + model name
def train_loop_agg_margin(network, dataloader, criterion, optimizer, epochs, rcs, sr, threshold_vc, num_tubes, vowels, offset, 
                   device, margin, agg_num, model_name):
    loss_prev = np.inf

    # reset_model_params(network)
    reinitialize_model(network)

    network.to(device)
    criterion.to(device)
    count = 0

    for epoch in range(epochs):
        since = time.time()

        print(f"Epoch {epoch + 1}\n-------------------------------")
        # batches
        for i, (audio1, phoneme1, text1, speaker1, rcs1, audio2, phoneme2, text2, speaker2, rcs2, label) in enumerate(dataloader):
            # print("For")

            # Initialize two (1, 1, 320) tensors
            # layer_1_tensor = torch.randn((1, 1, 320))
            # layer_2_tensor = torch.randn((1, 1, 320))

            # Convert rcs1 and rcs 2 to tensors
            # print("rcs1 type:", type(rcs1))
            # print("rcs1 content:", rcs1)
            # print(f"rcs1 shape: {rcs1.shape}")
            rcs1 = rcs1.float().to(device)
            rcs2 = rcs2.float().to(device)
            label = label.to(device)

            # print("rcs1: ", rcs1)
            # print("rcs2: ", rcs2)

            # print(f"rcs1 before unsqueeze: {rcs1.shape}")
            # print(f"rcs2 before unsqueeze: {rcs2.shape}")
            rcs1 = rcs1.unsqueeze(1)
            rcs2 = rcs2.unsqueeze(1)

            # print(f"rcs1 after shape: {rcs1.shape}")
            # print(f"rcs2 after shape: {rcs2.shape}")
            # print(f"phoneme1: {phoneme1}")
            # print(f"phoneme2: {phoneme2}")
            # print(f"audio1: {audio1}")
            # print(f"audio2: {audio2}")
            # print(f"Speaker1: {speaker1}")
            # print(f"Speaker2: {speaker2}")
            # print("for texts:")
            # print(f"text1: {text1}")
            # print(f"text2: {text2}")

            # Forward pass
            optimizer.zero_grad()

            # Pass in the two layers
            output1, output2 = network(rcs1, rcs2)
            output1 = output1.to(device)
            output2 = output2.to(device)

            # Loss
            loss = criterion(output1, output2, label)

            # Backpropagate
            loss.backward()
            optimizer.step()

        print("loss_prev: ", loss_prev)
        print("loss: ", loss.item())
        # If loss improvement is less than threshold, stop training
        if loss_prev - loss.item() < threshold_vc:
            count += 1
        
        if count == 5:
            print(f"Loss improvement less than threshold for 5 times, stop training at epoch {epoch}, batch {i}")
            break

        loss_prev = loss.item()
        print(f"Epoch {epoch} loss: {loss.item()}")
        print(f"Time elapsed: {time.time() - since}s")
    
    # save the model
    # torch.save(network.state_dict(), f'models/ablation/agg_{agg_num}_model_{model_name}_siamese_margin_{margin}.pth')
    print("Training Done")