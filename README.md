# **Fake News Detection with BERT**

This project focuses on detecting fake news using a fine-tuned BERT-based classification model. It includes preprocessing, model training, evaluation, and inference. The best-performing model and tokenizer setup are saved and can be used directly without retraining.

## **Done by**

* Sandesh (CS22B1076)


## **Project Structure**

python  
CopyEdit  
`.`  
`├── Fake.csv`  
`├── True.csv`  
`├── saved_model/`  
`│   └── model_epoch_1.bin`  
`├── notebook.ipynb`  
`├── README.md`

---

## **Requirements**

You can install the required packages using the following:

bash  
CopyEdit  
`pip install -r requirements.txt`

Or manually:

bash  
CopyEdit  
`pip install torch transformers scikit-learn pandas matplotlib tqdm`

---

## **Model Overview**

* **Base Model**: BERT (`bert-base-cased`)

* **Max Length**: 512 tokens

* **Batch Size**: 8

* **Optimizer**: AdamW

* **Scheduler**: Linear warmup with 10% warmup steps

* **Training Epochs**: 3 (early stopping enabled)

* **Metrics**: Accuracy, Precision, AUC, EER

---

## **Dataset**

* `Fake.csv` → Label 0

* `True.csv` → Label 1

* Title \+ text combined as input.

* Duplicate rows removed.

* Cleaned using regex to remove URLs, mentions, hashtags, etc.

---

## **Evaluation Metrics (Best Model)**

| Model | Accuracy | Precision | AUC | EER |
| ----- | ----- | ----- | ----- | ----- |
| RNN | 0.69 | 0.63 | 0.78 | 0.21 |
| LSTM | 0.91 | 0.90 | 0.94 | 0.07 |
| **BERT** | **0.99** | **0.99** | **1.00** | **0.0003** |

---

## **Quickstart**

### **1\. Clone the repo / upload the notebook**

If running on Kaggle, simply upload the notebook and CSV files.

### **2\. Load saved model and tokenizer**

To skip training and run inference:

python  
CopyEdit  
`from transformers import BertTokenizer, BertForSequenceClassification`  
`import torch`

`# Load tokenizer`  
`tokenizer = BertTokenizer.from_pretrained('bert-base-cased')`

`# Load model`  
`model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=2)`  
`model.load_state_dict(torch.load('./saved_model/model_epoch_1.bin'))`  
`model.eval()`

### **3\. Run inference**

Use the loaded `model` and `tokenizer` to classify new articles or test sets.

---

## **Saved Files**

* Trained BERT model: `saved_model/model_epoch_1.bin`

* Tokenizer parameters: Loaded via `bert-base-cased` (no need to re-save separately)

---

## **Notes**

* Ensure the `saved_model` folder exists and contains the `model_epoch_1.bin` file.

* If using GPU, ensure CUDA is available (`torch.cuda.is_available()`).

* Modify batch size or sequence length as needed for your compute environment.

