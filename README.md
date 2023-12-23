# Voice Similarity Analysis with TIMIT Dataset

## Dataset
- **Source:** TIMIT
- **Sampling Rate:** 16 kHz

## Vocal Tract Reconstruction
Vocal tract shapes for vowel articulations are estimated using a novel feature extraction method. This involves modeling the vocal tract as a series of concatenated uniform tubes and estimating the reflection coefficients at each tube junction. These coefficients describe the relationship between consecutive tubes, allowing for the estimation of the vocal tract shape during various vowel articulations.

## Workflow
1. **Frequency Response from Speech:**
   - Compute the frequency response of a given vowel segment from the TIMIT dataset using the Fast Fourier Transform (FFT). This serves as the target frequency response.

2. **Transfer Function Computation:**
   - Initialize the reflection coefficients, which define the characteristics of the vocal tract for the specific vowel.
   - Calculate the frequency response of the vocal tract model using these coefficients, applying a transfer function over a specified frequency range.

3. **Optimization:**
   - Compare the target frequency response with the modeled frequency response.
   - Iteratively adjust the reflection coefficients to minimize the discrepancy between the two responses.

## Voice Comparison Model
The model aggregates sets of reflection coefficients from all vowel segments in a single audio input. Pairs of audio samples are fed into a Siamese Network, consisting of two convolutional neural networks, and are evaluated using contrastive loss.

## Current Model Performance

| Dataset                 | Accuracy | F-Score |
|-------------------------|----------|---------|
| Single Audio            | 0.7661   | 0.7715  |
| Aggregated Audio        | 0.9705   | 0.9166  |