# Part 1: Research & Selection
## 🔍 Three Promising Audio Forgery Detection Approaches
### 1) Wav2Vec2 + Fine-Tuning for Deepfake Detection
🔹 Key Technical Innovation
- Self-supervised pretraining on raw audio
- Learns speech representations without labels
- Fine-tuned on deepfake datasets for forgery detection
  
🔹 Reported Performance Metrics
- Achieves ~83.24% accuracy on deepfake audio datasets
- Lowers error rates by learning deep feature representations

🔹 Why It's Promising
- Works well for real-time detection when optimized
- Captures subtle speech artifacts from AI-generated voices
- Can be fine-tuned for different types of deepfake speech

🔹 Potential Limitations
- High computational cost for real-time inference
- Requires fine-tuning for best performance

### 2)  Squeezeformer (Lightweight Transformer for Speech Analysis)
🔹 Key Technical Innovation
- Optimized Transformer with smaller layers for speed
- Uses squeeze-and-attention blocks to reduce computation
- 3× faster than traditional speech models (e.g., Conformer)

🔹 Reported Performance Metrics
- ~85% accuracy on speech deepfake datasets
- Runs real-time on edge devices

🔹 Why It's Promising
- Works for low-latency real-time deepfake detection
- Efficient & lightweight → Ideal for streaming & mobile apps
- Faster inference than large models like Wav2Vec2

🔹 Potential Limitations
- Slightly lower accuracy than Wav2Vec2
- Needs more labeled deepfake audio data for best performance

### 3) CRNN (Convolutional Recurrent Neural Network) for Audio Forgery Detection
🔹 Key Technical Innovation
- Combines CNNs + RNNs to detect fake speech patterns
- CNN extracts spectral features, while RNN learns temporal dependencies

🔹 Reported Performance Metrics
- Achieves ~89% accuracy on deepfake datasets
- Low latency (~5–10ms inference per audio chunk)

🔹 Why It's Promising
- Balanced approach (good accuracy + real-time performance)
- Less computationally expensive than transformers
- Works well with short speech clips (1–2 sec segments)

🔹 Potential Limitations
- May struggle with long audio sequences
- Lower accuracy than deep transformers like Wav2Vec2

### Final Comparison Table
| Model         | Feature Extraction | Architecture | Accuracy | 	Computational Cost | Best Use Case |
| ------------- | ------------------ | ------------ | -------- | -------------------- | ------------ |
| CRNN Model    | Mel Spectrograms   | CNN + BiLSTM | ~89%     | Moderate             | Balanced deepfake detection |
| Squeezeformer | Raw Waveform/ Spectrograms       | Lightweight Transformer | ~85% | Moderate-High | Optimized real-time deepfake detection |
| WaveNet       | Raw Waveforms      | Dilated CNNs + ResNet Blocks | ~83.24 | Very High	 | High-performance offline deepfake detection | 
