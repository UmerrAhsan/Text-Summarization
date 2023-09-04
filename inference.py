from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from config import CFG


def load_model_tokenizer_for_inference():
    model = AutoModelForSeq2SeqLM.from_pretrained("./finetuned-flanT5_final/")
    tokenizer = AutoTokenizer.from_pretrained(CFG['tokenizer']['tokenizer_name'])
    return model,tokenizer


def generate_response(prompt_text, model,tokenizer, max_length=150, num_return_sequences=1):


    tokenized_input = tokenizer("summarize: " + prompt_text, return_tensors="pt")

    summary_ids = model.generate(

        input_ids= tokenized_input["input_ids"],
        max_length=max_length,  
        num_beams=4,     
        no_repeat_ngram_size=3,  
        early_stopping=True,
        num_return_sequences=num_return_sequences,
        )
    
    
    responses = []
    for response_id in summary_ids :
        response = tokenizer.decode(response_id, skip_special_tokens=True)
        responses.append(response)

    return responses
    

