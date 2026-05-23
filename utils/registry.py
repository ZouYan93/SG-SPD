from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
from transformers import PegasusTokenizer, PegasusForConditionalGeneration, PegasusConfig
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from model.encoder import AdditionalTransformerEncoder, FullyConnectedEncoder, CNNEncoder, LSTMEncoder, SemKeyStyleEncoder

MODEL_REGISTRY = {
    'Bart':    {'model_id': 'facebook/bart-large', 
                'config': "BartConfig", 
                'generation': "BartForConditionalGeneration", 
                'tokenizer': "BartTokenizer", 
                "local_model_path": "./pretrained_models/Bart"
                },
    'Pegasus': {'model_id': 'google/pegasus-xsum', 
                'config': "PegasusConfig", 
                'generation': "PegasusForConditionalGeneration", 
                'tokenizer': "PegasusTokenizer", 
                "local_model_path": "./pretrained_models/Pegasus"
                },
    'T5':      {'model_id': 't5-large', 
                'config': "T5Config", 
                'generation': "T5ForConditionalGeneration", 
                'tokenizer': "T5Tokenizer",
                'local_model_path': "./pretrained_models/T5"
                },
    'Flan-T5':  {'model_id': 'google/flan-t5-large', 
                 'config': "T5Config", 
                 'generation': "T5ForConditionalGeneration", 
                 'tokenizer': "T5Tokenizer",
                 'local_model_path': "./pretrained_models/Flan-T5"
                }
}

ENCODER_REGISTRY = {
    'Transformer': {'model': "AdditionalTransformerEncoder",
                    'args' : {'in_feature': 840, 'nhead': 8, 'feedforward_dim': 2048, 'num_layers': 6}
                    },
    'MLP':         {'model': "FullyConnectedEncoder",
                    'args' : {'in_feature': 840, 'hidden_dim': [512, 840]}
                    },
    'CNN':         {'model': "CNNEncoder",
                    'args' : {'in_feature': 840, 'out_feature': 840}
                    },
    'LSTM':        {'model': "LSTMEncoder",
                    'args' : {'in_feature': 840, 'hidden_dim': 840, 'num_layers': 6}
                    },
    'SemKeyStyle': {'model': "SemKeyStyleEncoder",
                    'args' : {
                        'in_feature': 840,
                        'input_len': 60,
                        'hidden_dim': 128,
                        'hidden_len': 96,
                        'n_in_blocks': 6,
                        'n_out_blocks': 6,
                        'nhead': 8,
                        'dropout': 0.1
                        }
                    }
}

CLASS_MAP = {
    'BartConfig': BartConfig,
    'BartForConditionalGeneration': BartForConditionalGeneration,
    'BartTokenizer': BartTokenizer,
    'PegasusConfig': PegasusConfig,
    'PegasusForConditionalGeneration': PegasusForConditionalGeneration,
    'PegasusTokenizer': PegasusTokenizer,
    'T5Config': T5Config,
    'T5ForConditionalGeneration': T5ForConditionalGeneration,
    'T5Tokenizer': T5Tokenizer,
    'AdditionalTransformerEncoder': AdditionalTransformerEncoder,
    'FullyConnectedEncoder': FullyConnectedEncoder,
    'CNNEncoder': CNNEncoder,
    'LSTMEncoder': LSTMEncoder,
    'SemKeyStyleEncoder': SemKeyStyleEncoder
}

ATTRIBUTE_MODEL_REGISTRY = {
    'SemKey': {
        'topic': {
            'task': 'zero-shot-classification',
            'model_id': 'facebook/bart-large-mnli',
            'local_model_path': './pretrained_models/SemKey-topic-bart-large-mnli',
            'use_safetensors': False,
            'candidate_labels': [
                'Movie Reviews and Sentiment',
                'Biographies and Factual Knowledge',
            ],
        },
        'sentiment': {
            'task': 'sentiment-analysis',
            'model_id': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
            'local_model_path': './pretrained_models/SemKey-sentiment-twitter-roberta',
            'use_safetensors': False,
            'neutral_label': 'neutral',
            'non_neutral_label': 'non_neutral',
        },
        'surprisal': {
            'model_id': 'gpt2',
            'local_model_path': './pretrained_models/SemKey-surprisal-gpt2',
            'use_safetensors': False,
            'tokenizer': 'GPT2Tokenizer',
            'generation': 'GPT2LMHeadModel',
        },
    }
}
