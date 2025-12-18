# Research-Paper-Assistant

# 🚀 Decoder-Only Transformer - Complete Setup Guide

## 📋 Overview

This guide will help you train a decoder-only transformer on **real ArXiv machine learning papers**.

---

## ⚡ Quick Start (2 Steps)

### **Step 1: Install Dependencies**

Run this in your first Colab cell:

```python
!pip install datasets tokenizers torch matplotlib tqdm
```

### **Step 2: Run Training**

Copy the **Final Production-Ready Decoder Transformer** code and run:

```python
if __name__ == "__main__":
    model, tokenizer, config = main()
```

---

## 📊 What You'll Get

### ✅ Real Dataset
- **10,000 real ArXiv ML abstracts** from Hugging Face
- Dataset: `CShorten/ML-ArXiv-Papers`
- **117,592 papers available** (we use 10k for faster training)
- Verified working as of December 2024

### ✅ Proper Training Setup
- **Train/Validation split** (90/10)
- **Learning rate scheduling**
- **Model checkpointing** (saves best model)
- **Progress bars** with tqdm
- **Training curves** visualization

### ✅ Model Architecture
- **Decoder-only transformer** (like GPT)
- **Causal self-attention** masking
- **Sinusoidal positional encoding**
- **Residual connections** & layer normalization
- **~6.7M parameters** (efficient for demonstration)

---

## 🎯 Training Configuration

```python
config = {
    'd_model': 256,        # Model dimension
    'num_layers': 4,       # Transformer layers
    'num_heads': 8,        # Attention heads
    'd_ff': 1024,          # Feed-forward dimension
    'vocab_size': 8000,    # Vocabulary size
    'batch_size': 32,      # Batch size
    'num_epochs': 50,      # Training epochs
    'lr': 3e-4,            # Learning rate
    'num_samples': 10000   # Number of abstracts
}
```

### ⏱️ Expected Training Time

- **With GPU (T4)**: ~2-3 hours
- **Without GPU**: ~10-15 hours (not recommended)

---

## 📁 What Gets Saved

After training, these files are saved to Google Drive at `/content/drive/MyDrive/decoder_transformer/`:

1. **`model.pt`** - Trained model weights
2. **`tokenizer.json`** - WordPiece tokenizer
3. **`config.pkl`** - Model configuration
4. **`training_history.pkl`** - Training metrics
5. **`training_curves.png`** - Visualization plots

---

## 🎓 Demonstration

The evaluation script demonstrates:

### 1. **Model Architecture**
- Parameter count
- Layer structure
- Model size

### 2. **Text Generation**
- Multiple prompts
- Academic writing style
- Coherent output

### 3. **Temperature Control**
- Low temp (0.3): Focused, deterministic
- High temp (1.5): Creative, diverse

### 4. **Top-k Sampling**
- Shows vocabulary diversity
- Different sampling strategies

### 5. **Next-Token Prediction**
- Probability distribution
- Top-10 predictions

### 6. **Long-Form Generation**
- Extended coherent text
- Up to 150 tokens

---

## 💡 Key Features

### ✨ What Makes This Good

1. **Real Data**: Uses actual ArXiv papers, not synthetic data
2. **Proper Splits**: Train/validation separation
3. **Best Practices**: Learning rate scheduling, gradient clipping
4. **Reproducible**: Saved checkpoints and configs
5. **Demonstrable**: Clear evaluation metrics


---

## 🔧 Troubleshooting

### Problem: Dataset download fails
**Solution**: The code has automatic fallback to synthetic data

### Problem: CUDA out of memory
**Solution**: Reduce `batch_size` from 32 to 16 or 8

### Problem: Training is slow
**Solution**: 
- Use GPU runtime in Colab (Runtime → Change runtime type → GPU)
- Reduce `num_samples` from 10000 to 5000

### Problem: Loss not decreasing
**Solution**: 
- Check you're using the real dataset (not synthetic)
- Try increasing `num_epochs` to 100
- Verify GPU is being used

---

## 📈 Understanding Results

### Good Signs ✅
- Loss decreases over epochs
- Perplexity goes down
- Generated text is coherent
- Validation loss tracks training loss

### Warning Signs ⚠️
- Loss increases after initial decrease (learning rate too high)
- Perplexity > 100 (model not learning)
- Generated text is gibberish (train longer)
- Large gap between train/val loss (overfitting)

---

## 🎨 Customization Options

### To make model bigger:
```python
config = {
    'd_model': 512,      # Increase
    'num_layers': 6,     # Increase
    'num_heads': 8,      # Keep same or increase
    'd_ff': 2048,        # Increase
}
```

### To train faster:
```python
config = {
    'num_samples': 5000,  # Reduce dataset
    'batch_size': 64,     # Increase (if memory allows)
    'num_epochs': 30,     # Reduce
}
```

### To improve quality:
```python
config = {
    'num_samples': 20000, # More data
    'num_epochs': 100,    # More training
    'vocab_size': 16000,  # Bigger vocabulary
}
```

---

## 📚 Theory Behind the Model

### Architecture Components

1. **Token Embedding**: Maps words to vectors
2. **Positional Encoding**: Adds position information
3. **Self-Attention**: Learns relationships between words
4. **Causal Mask**: Prevents looking at future tokens
5. **Feed-Forward**: Non-linear transformations
6. **Layer Norm**: Stabilizes training
7. **Residual Connections**: Helps gradient flow

### Training Process

1. **Input**: `[BOS] We present a novel` → **Target**: `We present a novel approach`
2. Model predicts next token at each position
3. Loss computed only on predicted tokens
4. Backpropagation updates weights
5. Repeat for all batches and epochs

---

## 🎯 Demonstration Script

The evaluation script shows:

```python
# 1. Load model
model, tokenizer, config = load_from_gdrive()

# 2. Generate text
text = model.generate(tokenizer, 
                     prompt="We present", 
                     max_length=100,
                     temperature=0.8,
                     top_k=50)

# 3. Show architecture
show_model_architecture(model, config)

# 4. Run comprehensive demo
comprehensive_demo(model, tokenizer, config)
```

---

## 🤔Questions & Answers

### Q: Why decoder-only and not encoder-decoder?
**A**: Decoder-only (like GPT) is simpler and sufficient for text generation. We only need to predict next tokens, not encode-decode.

### Q: What is causal masking?
**A**: It prevents the model from seeing future tokens. At position i, the model can only see tokens 0 to i-1, ensuring autoregressive generation.

### Q: How does attention work?
**A**: Each token creates Query, Key, Value vectors. Attention scores are computed between Q and K, then used to weight the V vectors.

### Q: Why use WordPiece tokenization?
**A**: It handles unknown words better than word-level, and is more efficient than character-level. Used by BERT and many modern models.

### Q: How do you prevent overfitting?
**A**: We use dropout, train/val split, early stopping, and monitor validation loss.

---

## 🚀 Next Steps

After successful demonstration:

1. **Experiment with hyperparameters**
2. **Try different datasets** (poetry, code, etc.)
3. **Implement beam search** for better generation
4. **Add temperature scheduling** during training
5. **Fine-tune on specific domains**
6. **Compare with other architectures**

---

## 📖 Resources

- **Original Transformer Paper**: "Attention Is All You Need" (Vaswani et al., 2017)
- **GPT Paper**: "Improving Language Understanding by Generative Pre-Training" (Radford et al., 2018)
- **PyTorch Tutorial**: https://pytorch.org/tutorials/
- **Hugging Face Datasets**: https://huggingface.co/datasets

---


---

*Last updated: December 2024*
