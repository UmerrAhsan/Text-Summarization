'''
This is the configuration file for the project.
if you want to change any of the parameters, you can change it here. We will change the model and tokenizer in this file.
'''
CFG = {

    'data_path': 'cnn_dailymail\Test.csv',
    'train_size': 0.7,
    'val_size': 0.1,
    'test_size': 0.2,

    'tokenizer': {
        'tokenizer_type': 'AutoTokenizer',
        'tokenizer_name': 'google/flan-t5-small',
    },



    'model': {
        'model_type': 'AutoModelForSeq2SeqLM',
        'model_name': 'google/flan-t5-small'
    },
    'trainer': {
        'output_dir': 'weights',
        'evaluation_strategy': 'epoch',
        'learning_rate': 2e-5,
        'per_device_train_batch_size': 16,
        'per_device_eval_batch_size': 16,
        'num_train_epochs': 1,
        'weight_decay': 0.01,
        'push_to_hub': True
    },
    'inference': {
        'inferece_task': 'Text-Summarization',
        'model_path': 'weights'
    }

    }
