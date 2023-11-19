# Text-Summarization
Finetuning Flan-T5 for Text Summarization

The dataset used for finetuning is CNN DailyMail dataset. Dataset Link : https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail

Install the following packages from requirements.txt ( run the following command in the terminal )

pip install -r requirements.txt

**Overview**

In this Project, we used PyTorch, the Transformers library, and Hugging Face's FlanT5 model for fine-tuning on a text summarization task. The training and evaluation loops were written in PyTorch, optimizing for GPU acceleration during the fine-tuning process. The primary focus of the project was text summarization, a challenging natural language processing task. The FlanT5 model, part of the Transformers library, was fine-tuned to generate concise and coherent summaries from given text inputs. The meric used for evaluation process is rouge score.

**Dependencies**

- PyTorch
- Transformers Library
- Hugging Face FlauBERT Model
- GPU for accelerated fine-tuning

