# Voice-Similarity

Dataset: TIMIT with a sampling rate (SR) of 16 kHz

Vocal Tract Reconstruction
As a novel feature extraction methods, the shape of human vocal tract is estimated for vowel articulations. This is done by modeling the vocal tract as a concatenated uniform N-tube and estimating the reflection coefficents which describes the relation between two consecutive tubes. The shape of whole vocal tract in articulating a certain vowel is estimated by extracting reflection coefficents at every tube junction.

Workflow
1. Frequency Response from Speech:
- For a given vowel segment from the TIMIT dataset, compute its frequency response using the FFT. This will act as the target frequency response.
2. Transfer Function Computation:
- Initialize the reflection coefficients. These coefficients will define the characteristics of the vocal tract for the given vowel.
- Using the initialized reflection coefficients, calculate the frequency response of the vocal tract model via a transfer function over a specified range of frequencies.
3. Optimization:
- Compare the target frequency response with the modeled frequency response.
- Adjust the reflection coefficients iteratively to minimize the difference between the two frequency responses.

Voice Comparison Model
Sets of reflection coefficents from all vowel segments in a single audio input are aggregated. A pair of audios to be compared is fed into the Siamese Network consisting of two convolutional neural networks and are evaluated via contrastive loss.

Current Model Performance
On Single Audio Dataset
Accuracy: 0.7661    F-Score: 0.7715

On Aggregated Audio Dataset
Accuracy: 0.9705    F-Score: 0.9166

