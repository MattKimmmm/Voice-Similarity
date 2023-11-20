from process_audio import audio_single
import torch
import time
import numpy as np

def train_loop(network, dataloader, criterion, optimizer, epochs, rcs, sr, threshold_vc, num_tubes, vowels, offset, device):
    loss_prev = 0

    network.to(device)
    criterion.to(device)

    for epoch in range(epochs):
        since = time.time()

        print(f"Epoch {epoch + 1}\n-------------------------------")
        # batches
        for i, (audio1, phoneme1, text1, speaker1, audio2, phoneme2, text2, speaker2, label) in enumerate(dataloader):

            # print(f"phoneme1: {phoneme1[0]}")
            # print(f"phoneme2: {phoneme2}")
            # print(f"audio1: {audio1}")
            # print(f"audio2: {audio2}")
            # print(f"Speaker1: {speaker1}")
            # print(f"Speaker2: {speaker2}")
            # print("for texts:")
            # print(f"text1: {text1}")
            # print(f"text2: {text2}")
            
            # Time for audio_single()
            since_audio = time.time()

            layer_1 = audio_single(rcs, epochs, sr, threshold_vc, num_tubes, audio1, phoneme1, vowels, offset)
            layer_2 = audio_single(rcs, epochs, sr, threshold_vc, num_tubes, audio2, phoneme2, vowels, offset)
            # print(f"layer_1 shape: {layer_1.shape}")
            # print(f"layer_1: {layer_1}")
            print(f"Time taken for two audio_single(): {time.time() - since_audio}s")

            layer_1_tensor = torch.from_numpy(layer_1).float().to(device)  # Convert to tensor and ensure dtype is float
            layer_2_tensor = torch.from_numpy(layer_2).float().to(device)  # Convert to tensor
            label = torch.from_numpy(np.array(label)).float().to(device)

            # Initialize two (1, 1, 320) tensors
            # layer_1_tensor = torch.randn((1, 1, 320))
            # layer_2_tensor = torch.randn((1, 1, 320))

            # Make it 3D if the tensor is 1D (Batch size 1)
            if len(layer_1_tensor.shape) == 1:
                layer_1_tensor = layer_1_tensor.unsqueeze(0).unsqueeze(0)
                layer_2_tensor = layer_2_tensor.unsqueeze(0).unsqueeze(0)

            # Forward pass
            optimizer.zero_grad()

            # Pass in the two layers
            output1, output2 = network(layer_1_tensor, layer_2_tensor)

            # Loss
            loss = criterion(output1, output2, label)

            # Backpropagate
            loss.backward()
            optimizer.step()

        # If loss improvement is less than threshold, stop training
        if abs(loss_prev - loss.item()) < 1:
            print("Loss improvement less than threshold, stop training")
            break

        loss_prev = loss.item()
        print(f"Epoch {epoch} loss: {loss.item()}")
        print(f"Time elapsed: {time.time() - since}s")
    
    # save the model
    torch.save(network.state_dict(), 'models/siamese_1115.pth')
    print("Training Done")

    