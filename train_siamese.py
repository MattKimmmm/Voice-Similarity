from process_audio import audio_single
import torch
import time
import numpy as np

def train_loop(network, dataloader, criterion, optimizer, epochs, rcs, sr, threshold_vc, num_tubes, vowels, offset, device):
    loss_prev = np.inf

    network.to(device)
    criterion.to(device)

    for epoch in range(epochs):
        since = time.time()

        print(f"Epoch {epoch + 1}\n-------------------------------")
        # batches
        for i, (audio1, phoneme1, text1, speaker1, rcs1, audio2, phoneme2, text2, speaker2, rcs2, label) in enumerate(dataloader):

            print(f"phoneme1: {phoneme1[0]}")
            print(f"phoneme2: {phoneme2}")
            print(f"audio1: {audio1}")
            print(f"audio2: {audio2}")
            print(f"Speaker1: {speaker1}")
            print(f"Speaker2: {speaker2}")
            print("for texts:")
            print(f"text1: {text1}")
            print(f"text2: {text2}")

            # Initialize two (1, 1, 320) tensors
            # layer_1_tensor = torch.randn((1, 1, 320))
            # layer_2_tensor = torch.randn((1, 1, 320))

            # Convert rcs1 and rcs 2 to tensors
            rcs1 = rcs1.float().to(device)
            rcs2 = rcs2.float().to(device)

            print("rcs1: ", rcs1)
            print("rcs2: ", rcs2)

            # Make it 3D if the tensor is 1D (Batch size 1)
            if len(rcs1.shape) == 1:
                rcs1 = rcs1.unsqueeze(0).unsqueeze(0)
                rcs2 = rcs2.unsqueeze(0).unsqueeze(0)
            else:
                rcs1 = rcs1.unsqueeze(1)
                rcs2 = rcs2.unsqueeze(1)

            # print(f"rcs1 shape: {rcs1.shape}")
            # print(f"rcs2 shape: {rcs2.shape}")

            # Forward pass
            optimizer.zero_grad()

            # Pass in the two layers
            output1, output2 = network(rcs1, rcs2)

            # Loss
            loss = criterion(output1, output2, label)

            # Backpropagate
            loss.backward()
            optimizer.step()

        print("loss_prev: ", loss_prev)
        print("loss: ", loss.item())
        # If loss improvement is less than threshold, stop training
        if abs(loss_prev - loss.item()) < 1:
            print("Loss improvement less than threshold, stop training")
            break

        loss_prev = loss.item()
        print(f"Epoch {epoch} loss: {loss.item()}")
        print(f"Time elapsed: {time.time() - since}s")
    
    # save the model
    torch.save(network.state_dict(), 'models/siamese_1201.pth')
    print("Training Done")

    