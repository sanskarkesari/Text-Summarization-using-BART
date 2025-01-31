# Text Summarization using BART Model

## Overview
This project demonstrates text summarization using the **BART (Bidirectional and Auto-Regressive Transformers) model**. The summarization is performed in three stages:
1. **Without Fine-tuning** – Using the pre-trained BART model for summarization.
2. **With Fine-tuning** – Training the BART model on a custom dataset for improved summarization.
3. **Comparison of Summarized Text** – Evaluating and comparing the generated summaries against the given articles.

## Dataset
The dataset consists of articles for which we generate summarized versions using the BART model. Fine-tuning is performed on a labeled dataset containing original and summarized texts.

## Installation
To run this project, ensure you have the following dependencies installed:

```bash
pip install torch transformers datasets nltk
```

## Usage

### 1. Summarization without Fine-tuning
Run the following script to generate a summary using the pre-trained BART model:

```python
from transformers import BartForConditionalGeneration, BartTokenizer

tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

article = """ Your input article here """
inputs = tokenizer.encode("summarize: " + article, return_tensors="pt", max_length=1024, truncation=True)
summary_ids = model.generate(inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print("Summary:", summary)
```

### 2. Summarization with Fine-tuning
Fine-tuning the BART model requires a dataset with article-summary pairs. The following script fine-tunes the model:

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
```

### 3. Comparing Summaries
After obtaining the summarized text, the outputs from both models (pre-trained and fine-tuned) are compared based on readability, coherence, and conciseness.

## Results
The project evaluates the effectiveness of fine-tuning by comparing:
- **Pre-trained BART summary** vs. **Fine-tuned BART summary**
- Quality improvements in text coherence and contextual accuracy

## Conclusion
Fine-tuning the BART model significantly improves the quality of text summarization, making it more contextually relevant and accurate compared to using the model without fine-tuning.

## License
This project is open-source and available under the MIT License.

## Acknowledgments
- Hugging Face's `transformers` library
- PyTorch for model training

## Contact
For any queries or suggestions, feel free to reach out via GitHub Issues.

---
Happy Coding! 🚀

