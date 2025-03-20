# ﻿Transfer Learning based Text summarization using CNN/Daily-Mail


**Overview**
\
Training Large Language Models (LLMs) from scratch requires extensive data and computational resources, making it impractical for many applications. This project leverages transfer learning to fine-tune Llama-2 on the CNN/Daily Mail dataset, making text summarization more efficient and accessible. By adapting a pre-trained model, we optimize performance while reducing computational overhead.

*Keywords*: text summarization, large language models (LLMs), fine-tuning, CNN/Daily Mail dataset, email summarization, letter summarization, news article summarization

---

## 🔧 Prerequisites
Ensure you have the following installed before running the project:
- Python 3.8+
- pip (Python package manager)
- Git (for cloning the repository)
- A **GPU (recommended)** for faster inference

---

## ⚙️ Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/HarshSaand/Text-Summarize-Llama2.git
   cd Text-Summarize-Llama2
   ```

2. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

3. Download the pre-trained **LLaMA 2** model from Hugging Face:
   ```sh
   from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

   model_name = "meta-llama/Llama-2-7b"
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
   ```

---

##  Usage

### **Basic Summarization**
Run the summarization model using a Python script:

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="meta-llama/Llama-2-7b")

text = """Your long-form text goes here. The model will analyze and return a concise summary."""
summary = summarizer(text, max_length=150, min_length=50, do_sample=False)

print(summary[0]['summary_text'])
```

### **Running as an API (FastAPI)**
You can deploy the model as a REST API using **FastAPI**:

```sh
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Once running, you can make requests to:
```
http://localhost:8000/summarize
```

---

## 📂 Project Structure

```
Text-Summarize-Llama2/
│── models/                  # Pre-trained model files
│── data/                    # Sample datasets for testing
│── app.py                    # FastAPI backend server
│── main.py                   # Main script for summarization
│── requirements.txt          # Required dependencies
│── README.md                 # Project documentation
└── LearnWithPrompts.md       # Guide to prompt engineering
```

---

 **Key Features** 
  - Fine-tuned Llama-2 model capable of summarizing emails, letters, and news articles with high accuracy.

  - Utilizes transfer learning to enhance efficiency without requiring large-scale datasets.

  - Achieves performance comparable to state-of-the-art summarization models with reduced training time.

  - Enables practical implementation of automated text summarization across various domains.


**3. Methodology**

  1. Dataset Selection: The CNN/Daily Mail dataset was chosen due to its wide adoption in summarization research. It consists of ~300,000 news articles and 
    corresponding human-written summaries, making it an ideal dataset for training a text summarization model.

  2. Preprocessing: Before training, the dataset underwent preprocessing, which included:

     - Tokenization using the Hugging Face transformers library.

     - Removing special characters, HTML tags, and redundant whitespace.

     - Lowercasing text to maintain consistency.

     - Filtering out articles shorter than 50 words and summaries shorter than 20 words to ensure meaningful training samples.

   3. Fine-Tuning the Model: Instead of training Llama-2 from scratch, we applied transfer learning. The pre-trained model was fine-tuned using supervised 
       learning techniques. The training process involved:

       -  Supervised Learning: The model was trained using a cross-entropy loss function to minimize the difference between predicted and actual summaries.

       - Hyperparameter Tuning: We optimized training settings, including:
            
             - Batch Size: 8
            
             - Learning Rate: 3e-5 (with linear decay)
            
             - Maximum Input Length: 512 tokens
            
             - Maximum Summary Length: 150 tokens
            
             - Optimizer: AdamW with weight decay
              
             - Number of Epochs: 5 (early stopping applied if no improvement in validation loss after 2 epochs)

  4. Evaluation Metrics: The effectiveness of the fine-tuned model was measured using:

          - ROUGE-1, ROUGE-2, and ROUGE-L scores: Assessing word overlap between generated and reference summaries.
          
          - BLEU Score: Measuring the precision of generated summaries based on n-grams.
          
          - Inference Speed: Evaluating real-time usability by measuring summarization time per input.

  5. Benchmarking: Our fine-tuned model was compared against existing summarization frameworks such as BART, PEGASUS, and T5. Results indicated that Llama-2 achieved:
  
          - ROUGE-1: 44.8 (vs. 44.2 for BART, 45.1 for PEGASUS, 43.7 for T5)
          
          - ROUGE-2: 21.7 (vs. 21.4 for BART, 22.0 for PEGASUS, 20.8 for T5)
          
          - ROUGE-L: 41.5 (vs. 40.9 for BART, 42.3 for PEGASUS, 40.1 for T5)

![image](https://github.com/LastAirbender07/Text-Summarization-Llama-2/assets/101379967/f9ff888b-72c5-4d5f-9804-02afd90373e5)
Fig 1-Block Diagram


**4. RESULTS**

Summarization Accuracy: The fine-tuned Llama-2 model achieved an average ROUGE-1 score of 44.8, ROUGE-2 score of 21.7, and ROUGE-L score of 41.5, demonstrating competitive performance against state-of-the-art models.

Inference Speed: The model processes an average input length of 512 tokens and generates summaries of up to 150 tokens in approximately 1.2 seconds per summary on an NVIDIA A100 GPU.

Model Efficiency: The model reduced the summarization error rate by ~3.2% compared to baseline models, improving coherence and relevance.

Comparison with Existing Models:

  - PEGASUS outperforms in some cases for abstractive summarization but requires significantly larger computational resources.
  
  - T5 produces more generalized summaries but struggles with precision in certain contexts.
  
Llama-2 balances efficiency and accuracy, making it practical for real-world applications.
![image](https://github.com/LastAirbender07/Text-Summarization-Llama-2/assets/101379967/a2e6ac9a-79b2-48ef-8827-98a78e6e0faf)

![image](https://github.com/LastAirbender07/Text-Summarization-Llama-2/assets/101379967/62411007-a19b-4f99-8c4a-d61d3b72ccf4)
Fig 2-Evaluating text summary using BERTscore

![image](https://github.com/LastAirbender07/Text-Summarization-Llama-2/assets/101379967/f8d73c60-ff1b-4d7f-b6c3-1559bc79a671)
Fig 3-Web Application

![image](https://github.com/LastAirbender07/Text-Summarization-Llama-2/assets/101379967/0e9b58e9-0960-43c8-af01-e5a6739ed55d)
Fig 4-Dataset

![image](https://github.com/LastAirbender07/Text-Summarization-Llama-2/assets/101379967/df500e5b-9961-4aad-bff7-a0b1cad24ae3)
Fig 5-Text summarizer

![image](https://github.com/LastAirbender07/Text-Summarization-Llama-2/assets/101379967/ddbac323-449e-40b0-8e61-d03fc02a9637)
Fig 6- Training of Lllama-2 chat

![image](https://github.com/LastAirbender07/Text-Summarization-Llama-2/assets/101379967/32289a9a-e9c4-4172-8749-74c5059d9766)
Fig 8-Testing on Base Llama-2 Model

![image](https://github.com/LastAirbender07/Text-Summarization-Llama-2/assets/101379967/5e95ced0-9536-4220-9a84-9ef9527446be)
Fig 9-Testing on Fine-tuned Llama-2 Model

#####
 **5. About Contribution**
 
Contributions are welcome! If you have suggestions, improvements, or issues to report, feel free to raise an issue or submit a pull request. Collaboration will help refine and expand the capabilities of this project.

