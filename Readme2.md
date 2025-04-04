# Part 3: Documentation & Analysis
## Challenges Encountered & Solutions
| Challenge | Solution |
| --------- | -------- |
| Variable-length audio clips	| 	Padded/truncated spectrograms to fixed length (max_len=200) |
| Class imbalance	 | Used stratified splits in train_test_split |
| Slow training on large datasets | Implemented batched loading with DataLoader and CUDA acceleration | 
| Overfitting | Added dropout (0.3) and batch normalization |

## Assumptions Made
- Audio format: Assumes .wav at 16kHz.
- Fixed input size: All spectrograms resized to (128, 200).
- Binary classification: Only "real" (0) and "fake" (1) labels.

## Analysis
### Why CRNN for Implementation?
- CNNs extract spatial features from spectrograms effectively.
- LSTMs capture temporal dependencies in audio signals.
- Proven Success: CRNNs have been used effectively in speech recognition tasks.
- Comparison: WaveNet excels in high-fidelity speech synthesis but is computationally expensive. Squeezeformer achieves high accuracy but requires specialized optimization for real-time use.

### How the Model Works (high-level technical explanation)
- Input: Audio file → converted to mel-spectrogram.
- Feature Extraction: CNN layers detect local spectral patterns (e.g., artifacts in fake audio).
- Temporal Modeling: LSTM processes CNN features sequentially to detect anomalies.
- Classification: Final Linear layer outputs probabilities for "real" vs. "fake."

## Performance Results on ASVspoof 5
| Metric | Value |
| ------ | ----- | 
| Accuracy | 73.72% |
| Precision | 97.73% | 
| Recall  | 49.60% |  
| F1-Score | 65.81% |

## Strengths & Weaknesses
| Strengths | Weakness | 
| --------- | -------- |
| Handles variable-length audio well | Requires fixed spectrogram size |
|  Robust to noise (CNN filters) | Slow on very long audio (>5 sec) |
| Works on multiple datasets | May overfit if dataset is small |

### Future Improvements
- Incorporate Data Augmentation: Time-shifting, noise injection, etc.
- Hybrid Models: Combine CRNN with transformer embeddings.

## Reflection Questions

1. Significant Challenges in Implementation?
- Dataset formatting and preprocessing were time-consuming.
- Balancing model size and performance was tricky.
  
2. Real-World vs. Research Dataset Performance?
- Likely lower performance in real-world settings due to unseen deepfake variations.
- Research datasets may have less noise than real-world data.

3. Additional Data or Resources for Improvement?
- More real-world deepfake samples.
- Higher-quality pretrained models for transfer learning.
- Computational power to test larger models.

4. Deployment Considerations?
- Cloud-based API for real-time analysis.
- Regular model updates to adapt to evolving deepfake techniques.
