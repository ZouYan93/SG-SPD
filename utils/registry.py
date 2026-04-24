from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
from transformers import PegasusTokenizer, PegasusForConditionalGeneration, PegasusConfig
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from model.encoder import AdditionalTransformerEncoder, FullyConnectedEncoder, CNNEncoder, LSTMEncoder

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
    'LSTMEncoder': LSTMEncoder
}