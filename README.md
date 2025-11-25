# DeBERTa Sentiment Analysis Project

## 1. Project Overview

This project implements a state-of-the-art Sentiment Analysis system capable of classifying social media text (specifically tweets) into **Positive** or **Negative** sentiments. Unlike traditional approaches that rely on simple keyword matching, this solution leverages **DeBERTa V3 (Decoding-enhanced BERT with Disentangled Attention)**, a transformer-based model that achieves superior performance by understanding context, sarcasm, and complex sentence structures.

The system is built on the **Sentiment140** dataset and involves a rigorous pipeline including data preprocessing, a statistical baseline model (TF-IDF + Logistic Regression), and a deep learning model (DeBERTa). The final deliverable includes a trained model and a user-friendly web interface powered by **Gradio** for real-time inference.

### Key Objectives
*   **High Accuracy**: Surpass traditional machine learning baselines using Deep Learning.
*   **Robustness**: Handle noisy social media text, including emojis, slang, and HTML artifacts.
*   **Deployability**: Provide a clean, interactive web interface for end-users.

---

## 2. Theoretical Background

### 2.1. Sentiment Analysis
Sentiment Analysis is a Natural Language Processing (NLP) task defined as the computational study of people's opinions, sentiments, evaluations, attitudes, and emotions from written language. In this project, we treat it as a **Binary Classification** problem:
*   $Input (X)$: A sequence of text (tweet).
*   $Output (Y)$: A label $y \in \{0, 1\}$, where $0$ represents Negative and $1$ represents Positive.

### 2.2. DeBERTa V3 Architecture
We utilize **DeBERTa V3 (Decoding-enhanced BERT with Disentangled Attention)**, which improves upon the original BERT and RoBERTa architectures through two key innovations:

1.  **Disentangled Attention**: Unlike BERT, which adds position embeddings to word embeddings, DeBERTa represents each word using two vectors: one for its content and one for its position. The attention scores are computed using disentangled matrices on content and relative positions, allowing the model to better understand the dependency between words based on their spatial relationships.
    
    $$Attention(Q, K, V) = Softmax(\frac{Q \cdot K^T}{\sqrt{d}})V$$
    
    In DeBERTa, the $Q \cdot K^T$ term is decomposed into content-to-content, content-to-position, and position-to-content interactions.

2.  **Enhanced Mask Decoder (EMD)**: DeBERTa incorporates absolute word positions in the decoding layer to predict masked tokens, which is crucial for tasks requiring precise syntactical understanding.

3.  **ELECTRA-Style Pretraining (V3 specific)**: DeBERTa V3 replaces the Masked Language Modeling (MLM) objective with Replaced Token Detection (RTD), where a generator replaces tokens and the discriminator (DeBERTa) must identify them. This is far more sample-efficient.

---

## 3. Project Structure

The project follows a structured organization to separate data, code, and artifacts:

```
DeBERTa-Sentiment-Analysis/
├── data/
│   └── training.1600000.processed.noemoticon.csv  # Raw Sentiment140 dataset
├── output/
│   ├── results/
│   │   └── checkpoint-12000/       # Intermediate training checkpoint
│   │       ├── model.safetensors   # Model weights
│   │       ├── config.json         # Model configuration
│   │       ├── tokenizer_config.json
│   │       └── ... (optimizer states, etc.)
│   ├── saved_model/                # Final export-ready model
│   │   ├── model.safetensors
│   │   ├── config.json
│   │   ├── spm.model               # SentencePiece model
│   │   └── ...
│   ├── __results___files/          # Generated plots from notebook
│   │   └── __results___8_0.png
│   ├── confusion_matrix.png        # Performance visualization
│   ├── final_metrics.json          # Evaluation scores
│   └── __huggingface_repos__.json
├── app.py                          # Gradio web application
├── nlp-kaggle1.ipynb               # Training notebook
├── log.log                         # Application logs
├── requirements.txt                # Dependencies
└── README.md                       # Project documentation
```

---

## 4. Implementation Details

This section details every step of the pipeline, explaining the logic and code found in `nlp-kaggle1.ipynb`.

### Step 1: Environment Setup
We utilize a specific stack of libraries to ensure reproducibility and performance:
*   `transformers`: For accessing the pre-trained DeBERTa model.
*   `torch`: The PyTorch deep learning framework.
*   `scikit-learn`: For metrics and splitting data.
*   `emoji`: To translate emojis into text (e.g., "😊" $\rightarrow$ ":smile:").

### Step 2: Data Loading & Stratified Sampling
**Source**: Sentiment140 dataset (1.6 million tweets).
**Logic**: Processing 1.6M rows can be computationally expensive. We perform **Stratified Sampling** to select a balanced subset of **1,000,000 samples** (500k Positive, 500k Negative). This ensures the model does not become biased toward one class.

*   **Label Mapping**: The raw dataset uses `0` for Negative and `4` for Positive. We map these to standard binary labels: `0` (Negative) and `1` (Positive).

### Step 3: Advanced Preprocessing
Social media text is noisy. We implement a `clean_text` function to sanitize inputs before they reach the model.

1.  **HTML Decoding**: Converts entities like `&amp;` back to `&`.
2.  **Demojization**: We use the `emoji` library to convert emojis into text strings.
    *   *Why?* DeBERTa understands text better than unicode symbols. Converting "I am 😠" to "I am :angry:" allows the model to leverage the semantic meaning of the word "angry".
3.  **Regex Cleaning**:
    *   Removes URLs (`http...`) as they rarely contain sentiment.
    *   Removes User Mentions (`@user`) to protect privacy and reduce noise.

### Step 4: Data Splitting
We divide the 1M samples into three sets:
1.  **Training Set (95%)**: Used to update model weights.
2.  **Validation Set (5%)**: Used to evaluate loss during training and trigger early stopping.
3.  **Held-out Test Set (10,000 samples)**: A completely unseen dataset used ONLY for the final performance report.

### Step 5: Baseline Model (TF-IDF + Logistic Regression)
To scientifically validate the need for a deep learning model, we first train a baseline.
*   **TF-IDF (Term Frequency-Inverse Document Frequency)**: Converts text into sparse numerical vectors based on word counts.
*   **Logistic Regression**: A linear classifier that learns a decision boundary.
*   *Result*: This typically achieves ~80% accuracy, setting the "bar" for DeBERTa.

### Step 6: Deep Learning Model (DeBERTa V3)
This is the core of the project.

#### 6.1. Tokenization
We use the `AutoTokenizer` from `microsoft/deberta-v3-base`.
*   **Truncation**: Max length set to **128 tokens**. This covers 99% of tweets while keeping memory usage efficient.

#### 6.2. Training Configuration
We use the Hugging Face `Trainer` API with "Platinum" enterprise settings:
*   **Batch Size**: 32 per device with **Gradient Accumulation** of 4 steps. This simulates a large batch size of 128, stabilizing training.
*   **Learning Rate**: `2e-5` with a **Cosine Scheduler**. The LR starts high to escape local minima and decays to fine-tune.
*   **FP16 (Mixed Precision)**: Uses 16-bit floating-point numbers to speed up training on GPUs and reduce memory usage.
*   **Label Smoothing (0.1)**: Prevents the model from becoming over-confident (e.g., predicting 0.999 probability), which improves generalization.

#### 6.3. Evaluation Metrics
We track multiple metrics to ensure a holistic view of performance. The final model achieved:
*   **Accuracy**: **88.38%** (Significant improvement over baseline)
*   **F1-Score**: **0.8827**
*   **MCC (Matthews Correlation Coefficient)**: **0.7679**
*   **Precision**: **0.8952**
*   **Recall**: **0.8706**

---

## 5. Web Application (`app.py`)

The project includes a production-ready web interface built with **Gradio**.

### Logic Flow
1.  **Load Model**: On startup, the script loads the saved model and tokenizer from `./output/saved_model`.
2.  **Input Processing**: The user enters text. The app applies the *exact same* `clean_text` function used during training.
3.  **Inference**:
    *   The text is tokenized.
    *   The model outputs **Logits** (raw scores).
    *   We apply a **Softmax** function to convert logits into probabilities summing to 1.
4.  **Visualization**:
    *   If Positive Probability > Negative Probability, display **POSITIVE** (Green).
    *   Else, display **NEGATIVE** (Red).
    *   A detailed "Analysis Breakdown" HTML block shows the cleaned text, token count, and inference time.

---

## 6. Usage Instructions

### Prerequisites
Ensure you have Python 3.8+ installed.

### 1. Install Dependencies
Run the following command to install all necessary libraries:
```bash
pip install -r requirements.txt
```

### 2. Train the Model
To reproduce the training results, open the Jupyter Notebook:
```bash
jupyter notebook nlp-kaggle1.ipynb
```
Run all cells. Note that training on 1 million samples requires a GPU and may take several hours.

### 3. Run the Web App
Once the model is trained and saved to `./output/saved_model`, launch the interface:
```bash
python app.py
```
Open the provided local URL (usually `http://127.0.0.1:7860`) in your browser.

---

## 7. Conclusion

This project demonstrates the power of modern NLP. By fine-tuning DeBERTa V3, we achieve a significant performance leap over traditional statistical methods. The inclusion of a robust preprocessing pipeline and a user-friendly deployment interface makes this a complete, end-to-end machine learning solution.
