# Learn with Prompts - Text-Summarize-Llama2

Welcome to **Text-Summarize-Llama2**, an AI-powered text summarization tool built using **LLaMA 2**. This project leverages **Hugging Face’s Transformers library** to generate concise and meaningful summaries of long-form text using **state-of-the-art language models**.

This guide will help you explore different **prompting techniques** and fine-tune your approach for optimal text summarization.

## 1️⃣ Basic Prompting
Start with a simple prompt to generate a summary:

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="meta-llama/Llama-2-7b")

text = """Your long text goes here. The model will analyze and return a concise summary."""
summary = summarizer(text, max_length=150, min_length=50, do_sample=False)

print(summary[0]['summary_text'])
```

🔹 **Tip:** Adjust `max_length` and `min_length` to fine-tune the summary length.

---

## 2️⃣ Length-Controlled Summarization
You can guide the model to produce **short, medium, or long summaries** by modifying the prompt structure:

```python
prompt = f"Summarize this text in one sentence:\n{text}"
summary = summarizer(prompt, max_length=50, min_length=20, do_sample=False)
```

🔹 **Tip:** Use phrases like *"Explain in a few words"* or *"Provide a detailed summary"* to guide output quality.

---

## 3️⃣ Extractive vs. Abstractive Summarization
- **Extractive Summarization:** Retains original phrases from the text.
- **Abstractive Summarization:** Generates new phrases with the same meaning.

### Extractive Example (Keyword-Based Prompting)
```python
prompt = f"Extract key points from this text:\n{text}"
summary = summarizer(prompt, max_length=100, do_sample=False)
```

### Abstractive Example (Paraphrasing for Clarity)
```python
prompt = f"Rewrite and simplify the key ideas from this text:\n{text}"
summary = summarizer(prompt, max_length=120, do_sample=True)
```

🔹 **Tip:** Use **paraphrasing prompts** for more natural and human-like summaries.

---

## 4️⃣ Few-Shot Prompting
To improve accuracy, provide **examples** in the prompt:

```python
prompt = f"""Summarize the following examples:
1. Original: 'The stock market crashed, causing widespread panic.'
   Summary: 'Stock market crash led to panic.'
2. Original: '{text}'
   Summary:
"""
summary = summarizer(prompt, max_length=100, do_sample=False)
```

🔹 **Tip:** Providing examples helps **steer** the model towards **desired output formats**.

---

## 5️⃣ Multi-Document Summarization
You can summarize **multiple texts at once** by concatenating them:

```python
texts = ["First article text here.", "Second article text here."]
combined_text = " ".join(texts)

summary = summarizer(combined_text, max_length=200, min_length=80, do_sample=False)
```

🔹 **Tip:** Use **separators (--- or bullet points)** for structured summaries.

---

## 6️⃣ Topic-Specific Summarization
Guide the model to **focus on specific themes**:

```python
prompt = f"Summarize the text focusing only on financial aspects:\n{text}"
summary = summarizer(prompt, max_length=100, do_sample=False)
```

🔹 **Tip:** Modify **focus words** like *"health," "technology," "politics"* for domain-specific summaries.

---

## 🚀 Experiment & Improve!
Try experimenting with:
✅ **Temperature & Top-K Sampling** to **control randomness**  
✅ **Custom Prompts** for different text types (news, research, blogs)  
✅ **Fine-Tuning** LLaMA 2 for domain-specific summarization  

This guide should help you make the most of **Text-Summarize-Llama2** and master **AI-powered text summarization**!

📌 **Contribute & Improve:** Found a better prompt? Share your findings in discussions or submit a PR!
