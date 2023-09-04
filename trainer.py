
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch.nn as nn
from config import CFG
from evaluate import load
import nltk
import numpy as np



def get_model(model_name):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model


def get_tokenizer(tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    return tokenizer


args = Seq2SeqTrainingArguments(
    "./flan-t5-finetuned-cnn/",
    evaluation_strategy = "epoch",
    learning_rate=0.0003,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0,
    save_total_limit=3,
    num_train_epochs=5,
    predict_with_generate=True,
)


class CustomTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")

        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Assuming you have a sequence-to-sequence model, you need to compute the loss differently
        # You can use a loss function like nn.CrossEntropyLoss, but applied to each timestep and averaged
        loss_fct = nn.CrossEntropyLoss(reduction="mean")

        # Flatten the logits and labels to make them compatible with the loss function
        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = labels.view(-1)

        # Compute the loss
        loss = loss_fct(logits_flat, labels_flat)
        #loss = calculate_sparse_categorical_cross_entropy_loss(logits, labels)
        #return (loss,outputs)
        return (loss, outputs) if return_outputs else loss



def compute_metrics(eval_pred):

    metric = load("rouge")
    tokenizer = get_tokenizer(CFG['tokenizer']['tokenizer_name'])

    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    # Note that other metrics may not have a `use_aggregator` parameter
    # and thus will return a list, computing a metric for each sentence.
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True, use_aggregator=True)
    # Extract a few results
    result = {key: value * 100 for key, value in result.items()}

    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}




def training_function(tokenized_train_dataset, tokenized_val_dataset):

    model = get_model(CFG['model']['model_name'])   
    tokenizer = get_tokenizer(CFG['tokenizer']['tokenizer_name'])

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = CustomTrainer(
        model,
        args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )


   # start the training and display the message after completion
    print("Training is started")
   # trainer.train()
    print("Training is completed")

    return trainer


