import argparse
import os
import sys

import pandas as pd
import torch
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

# Allow both:
#   python utils/data/generate_semkey_attributes.py
#   python -m utils.data.generate_semkey_attributes
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.registry import ATTRIBUTE_MODEL_REGISTRY


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate SemKey-style sentiment/topic/length/surprisal labels for ZuCo text.'
    )
    parser.add_argument(
        '--input_df',
        default='./dataset/ZuCo/zuco_label_8variants.df',
        help='Input dataframe/csv containing at least an "input text" column.',
    )
    parser.add_argument(
        '--output_df',
        default='./dataset/ZuCo/zuco_semkey_labels.df',
        help='Output dataframe path used by train_decoder.py.',
    )
    parser.add_argument('--cache_dir', default='./pretrained_models/hf_cache')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument(
        '--attribute_model_profile',
        default='SemKey',
        choices=sorted(ATTRIBUTE_MODEL_REGISTRY.keys()),
        help='Model profile from utils.registry.ATTRIBUTE_MODEL_REGISTRY.',
    )
    return parser.parse_args()


def load_table(path):
    if path.endswith('.csv'):
        return pd.read_csv(path)
    return pd.read_pickle(path)


def calculate_surprisal(text, tokenizer, model):
    inputs = tokenizer(str(text), return_tensors='pt').to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
    return float(outputs.loss.item())


def resolve_model_path(config):
    local_path = config.get('local_model_path')
    if local_path:
        if not os.path.isabs(local_path):
            local_path = os.path.abspath(os.path.join(PROJECT_ROOT, local_path))
        if os.path.exists(local_path):
            return local_path
    return config['model_id']


def build_model_kwargs(config, cache_dir):
    kwargs = {'cache_dir': cache_dir}
    if 'use_safetensors' in config:
        kwargs['use_safetensors'] = bool(config['use_safetensors'])
    return kwargs


def main():
    args = parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)
    profile = ATTRIBUTE_MODEL_REGISTRY[args.attribute_model_profile]

    input_path = args.input_df
    if not os.path.isabs(input_path):
        input_path = os.path.abspath(input_path)
    output_path = args.output_df
    if not os.path.isabs(output_path):
        output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    data = load_table(input_path)
    if 'input text' not in data.columns:
        raise KeyError('input dataframe must contain an "input text" column')

    data = data.drop_duplicates('input text').reset_index(drop=True)
    texts = data['input text'].astype(str).tolist()
    device = 0 if torch.cuda.is_available() else -1
    torch_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    print('Generating topic labels...')
    topic_config = profile['topic']
    topic_model_path = resolve_model_path(topic_config)
    topic_classifier = pipeline(
        topic_config['task'],
        model=topic_model_path,
        device=device,
        model_kwargs=build_model_kwargs(topic_config, args.cache_dir),
    )
    topic_results = topic_classifier(texts, topic_config['candidate_labels'], batch_size=args.batch_size)
    topic_labels = [result['labels'][0] for result in topic_results]

    print('Generating sentiment labels...')
    sentiment_config = profile['sentiment']
    sentiment_model_path = resolve_model_path(sentiment_config)
    sentiment_classifier = pipeline(
        sentiment_config['task'],
        model=sentiment_model_path,
        device=device,
        model_kwargs=build_model_kwargs(sentiment_config, args.cache_dir),
    )
    sentiment_labels = []
    for text in tqdm(texts, desc='Sentiment'):
        result = sentiment_classifier(text)[0]['label']
        sentiment_labels.append(
            sentiment_config['neutral_label']
            if result == sentiment_config['neutral_label']
            else sentiment_config['non_neutral_label']
        )

    print('Calculating length...')
    lengths = [len(text.split()) for text in texts]

    print('Calculating GPT-2 surprisal...')
    surprisal_config = profile['surprisal']
    surprisal_model_path = resolve_model_path(surprisal_config)
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained(surprisal_model_path, cache_dir=args.cache_dir)
    gpt2_model = GPT2LMHeadModel.from_pretrained(
        surprisal_model_path,
        **build_model_kwargs(surprisal_config, args.cache_dir),
    ).to(torch_device)
    gpt2_model.eval()
    surprisals = [
        calculate_surprisal(text, gpt2_tokenizer, gpt2_model)
        for text in tqdm(texts, desc='Surprisal')
    ]

    output = data.copy()
    output['topic_label'] = topic_labels
    output['sentiment label'] = sentiment_labels
    output['length'] = lengths
    output['surprisal'] = surprisals
    output.to_pickle(output_path)
    output.to_csv(os.path.splitext(output_path)[0] + '.csv', index=False)
    print(f'Saved SemKey attributes to {output_path}')


if __name__ == '__main__':
    main()
