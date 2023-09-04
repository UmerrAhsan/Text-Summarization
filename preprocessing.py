import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from config import CFG

def read_data(path,nrows=None):
    df = pd.read_csv(path,nrows = nrows)
    return df


def re_clean(text):
    import re
    text = re.sub(r'https?:\/\/.*[\r\n]*', ' ', text, flags=re.MULTILINE)
    text = re.sub(r'\<a href', ' ', text)
    text = re.sub(r'&amp;', ' ', text)
    text = re.sub(r'[_\-;%()|+&=*%:#$@\[\]/]', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'\'', ' ', text)
    text=re.sub(r'\n',' ',text)
    text=re.sub(' est ',' ',text)
    text=re.sub(r'[?!]','.',text)
    return text
#used to expand contractions
def expand(text):
    import contractions
    text=text.split()
    final=[]
    for word in text:
        try:
            final.append(contractions.fix(word)+" ")
        except:
            final.append(word+" ")
            print(word)
    return "".join(final)

def remove(texts):
    final=[]
    for text in texts:
        sents=[]
        sentences=text.split(".")
        for sentence in sentences:
            if(len(sentence.split())>=5):
                sents.append(sentence+".")
        final.append("".join(sents))
    return final

def removeTag(texts):
    final=[]
    #removing cnn and est
    for text in texts:
        cnn=text.find("cnn")
        if(cnn!=-1 and cnn<len(text)//10):
            text=text[cnn+3:]
        found=False
        for i in range(2):
            est=text.find(" est,")
            if(est<len(text)//5 and est!=-1):
                text=text[est+5:]
                found=True
        fs=text.find(".")
        if(fs<20 and fs!=-1 and found):
            text=text[fs:]
        final.append(text)
    return final



def preprocess_data(df):
    
    df = df[["article","highlights"]]

    texts=df.iloc[:,0]
    new_texts=[]
    for i in texts:
        new_texts.append(expand(re_clean(i)).lower())
    finalcleaned=removeTag(remove(new_texts))

    summaries=df.iloc[:,1]
    new_summaries=[]
    for i in summaries:
        new_summaries.append(expand(re_clean(i)).lower())



    df = df.drop(columns=['article'])
    df = df.drop(columns=['highlights'])


    df['article'] = finalcleaned
    df['highlights'] = new_summaries

    df["article"] = "summarize: " + df["article"]

    return df



def train_validation_test_split(df,training_proportion,validation_proportion,testing_proportion):

    size = len(df)

    training_size = int(training_proportion*size)
    validation_size = int(validation_proportion*size)

    train_df = df[0:training_size]
    val_df = df[training_size:training_size+validation_size]
    test_df = df[training_size+validation_size:]
    return train_df,val_df,test_df


def pandas_to_dataset(df1,df2,df3):
    hf_train_dataset = Dataset.from_pandas(df1)
    hf_val_dataset = Dataset.from_pandas(df2)
    hf_test_dataset = Dataset.from_pandas(df3)
    return hf_train_dataset,hf_val_dataset,hf_test_dataset




def get_tokenizer(tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    return tokenizer



def preprocess_function(examples):
    tokenizer = get_tokenizer(CFG['tokenizer']['tokenizer_name'])
    model_inputs = tokenizer(examples['article'], max_length=512, truncation=True)
    labels = tokenizer(text_target=examples["highlights"], max_length=150,truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs






def tokenize_hf_dataset(hf_train_dataset,hf_val_dataset, hf_test_dataset):

    tokenized_train_data= hf_train_dataset.map(preprocess_function, batched=True)
    tokenized_test_data= hf_test_dataset.map(preprocess_function, batched=True)
    tokenized_val_data= hf_val_dataset.map(preprocess_function, batched=True)
    tokenized_train_data = tokenized_train_data.remove_columns("article")
    tokenized_train_data = tokenized_train_data.remove_columns("highlights")
    tokenized_val_data = tokenized_val_data.remove_columns("article")
    tokenized_val_data = tokenized_val_data.remove_columns("highlights")
    tokenized_test_data = tokenized_test_data.remove_columns("article")
    tokenized_test_data = tokenized_test_data.remove_columns("highlights")
    return tokenized_train_data,tokenized_val_data,tokenized_test_data













