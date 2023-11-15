from process_audio import audio_single
import torch
import time
import numpy as np

def test_loop(network, dataloader, criterion, epochs, rcs, sr, threshold_vc, num_tubes, vowels, offset, device):
    
    losses = []

    network.eval()
    network.to(device)
    criterion.to(device)
    
    with torch.no_grad():
        # batches
        for i, (audio1, phoneme1, text1, speaker1, audio2, phoneme2, text2, speaker2, label) in enumerate(dataloader):
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

            layer_1 = audio_single(rcs, epochs, sr, threshold_vc, num_tubes, audio1, phoneme1, vowels, offset)
            layer_2 = audio_single(rcs, epochs, sr, threshold_vc, num_tubes, audio2, phoneme2, vowels, offset)
            # print(f"layer_1 shape: {layer_1.shape}")
            # print(f"layer_1: {layer_1}")

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

            # Pass in the two layers
            output1, output2 = network(layer_1_tensor, layer_2_tensor)

            # Loss
            loss = criterion(output1, output2, label)
            losses.append(loss.item())

            print(f"Batch {i} loss: {loss.item()}")
            print(f"Time elapsed: {time.time() - since}s")
    
    print(f"Average loss: {sum(losses) / len(losses)}")

    print("Test Done")

    

