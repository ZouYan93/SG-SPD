import math
import os
import re
import unicodedata
from difflib import SequenceMatcher
from functools import lru_cache


SENTIMENT_LABELS = ['non_neutral', 'neutral']
TOPIC_LABELS = ['Biographies and Factual Knowledge', 'Movie Reviews and Sentiment']

SEMKEY_TYPO_BOOK = (
    ('emp11111ty', 'empty'),
    ('film.1', 'film.'),
    ('\u2013', '-'),
    ('\u2019s', "'s"),
    ('\ufffds', "'s"),
    ('`s', "'s"),
    ('Maria', 'Mari\u0107'),
    ('1Universidad', 'Universidad'),
    ('1902\u201419', '1902 - 19'),
    ('1902\u2013?9', '1902 - 19'),
    ('1902\u9225?9', '1902 - 19'),
    ('Wuerttemberg', 'W\u00fcrttemberg'),
    ('long -time', 'long-time'),
    ('Jose', 'Jos\u00e9'),
    ('Bucher', 'B\u00f4cher'),
    ('1839 ? May', '1839 - May'),
    ('G\ufffdn\ufffdration', 'Generation'),
    ('Bragan\u00e7a', 'Bragana'),
    ('1837?October', '1837 - October'),
    ('nVera-Ellen', 'Vera-Ellen'),
    ('write Ethics', 'wrote Ethics'),
    ('Adams-Onis', 'Adams-On\u00eds'),
    ('(40 km?)', '(40 km\u00b2)'),
    ('(40 km\u02dd)', '(40 km\u00b2)'),
    (' (IPA: /?g?nz?b?g/) ', ' '),
    ('""Canes""', '"Canes"'),
)


def normalize_text(text):
    text = str(text).strip()
    for src, tgt in SEMKEY_TYPO_BOOK:
        text = text.replace(src, tgt)
    # Some EEG-side strings contain numeric artifacts glued to words, e.g.
    # "U.S. 111Senator" while the label table has "U.S. Senator".
    text = re.sub(r'(?<!\w)\d{2,}(?=[A-Z][A-Za-z])', '', text)
    return ' '.join(text.split())


def aggressive_normalize_text(text):
    text = normalize_text(text)
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', ' ', text)
    return ' '.join(text.split())


def transliterated_normalize_text(text):
    text = aggressive_normalize_text(text)
    # Match common German ASCII transliterations against diacritic-stripped text.
    # Example: Eidgenoessische -> Eidgenossische, Württemberg -> Wurttemberg.
    for src, tgt in (('ae', 'a'), ('oe', 'o'), ('ue', 'u')):
        text = text.replace(src, tgt)
    return text


def matching_text_keys(text):
    keys = []
    for key in (
        normalize_text(text),
        aggressive_normalize_text(text),
        transliterated_normalize_text(text),
    ):
        if key and key not in keys:
            keys.append(key)
    return keys


def compact_normalize_text(text):
    text = transliterated_normalize_text(text)
    text = re.sub(r'\d{2,}(?=[a-z])', '', text)
    text = re.sub(r'[^a-z0-9]+', '', text)
    return text


def normalize_task_key(task_key):
    task_key = str(task_key)
    mapping = {
        '<SR>': 'task1',
        '<NR>': 'task2',
        '<TSR>': 'task3',
        'task1': 'task1',
        'task2': 'task2',
        'task3': 'task3',
    }
    return mapping.get(task_key, task_key)


def normalize_sentiment_label(label):
    if label is None:
        return None
    try:
        if isinstance(label, float) and math.isnan(label):
            return None
    except TypeError:
        pass
    label = str(label).strip().lower()
    if label in ('neutral', 'neu'):
        return 'neutral'
    if label in ('non_neutral', 'non-neutral', 'negative', 'positive', 'neg', 'pos', '-1', '1'):
        return 'non_neutral'
    if label == '0':
        return 'neutral'
    return None


def encode_sentiment(label):
    label = normalize_sentiment_label(label)
    return SENTIMENT_LABELS.index(label) if label in SENTIMENT_LABELS else -1


def normalize_topic_label(label):
    if label is None:
        return None
    try:
        if isinstance(label, float) and math.isnan(label):
            return None
    except TypeError:
        pass
    label = str(label).strip()
    if label in TOPIC_LABELS:
        return label
    lower = label.lower()
    if 'movie' in lower or 'sentiment' in lower:
        return 'Movie Reviews and Sentiment'
    if 'biograph' in lower or 'factual' in lower or 'knowledge' in lower:
        return 'Biographies and Factual Knowledge'
    return None


def encode_topic(label):
    label = normalize_topic_label(label)
    return TOPIC_LABELS.index(label) if label in TOPIC_LABELS else -1


def fallback_topic_from_task(task_key):
    task_key = normalize_task_key(task_key)
    if task_key == 'task1':
        return 'Movie Reviews and Sentiment'
    if task_key in ('task2', 'task3'):
        return 'Biographies and Factual Knowledge'
    return None


def _coerce_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float('nan')
    if math.isfinite(value):
        return value
    return float('nan')


def default_label_path_candidates(repo_root):
    return [
        os.path.join(repo_root, 'dataset', 'ZuCo', 'zuco_semkey_labels.df'),
        os.path.join(repo_root, 'dataset', 'ZuCo', 'zuco_semkey_labels.csv'),
        os.path.join(repo_root, 'dataset', 'ZuCo', 'zuco_label_input_text.df'),
        os.path.join(repo_root, 'dataset', 'ZuCo', 'zuco_label_input_text.csv'),
        os.path.abspath(os.path.join(
            repo_root,
            '..',
            'SemKey-main',
            'SemKey-main',
            'data',
            'zuco_preprocessed_dataframe',
            'zuco_label_input_text.df',
        )),
    ]


def resolve_label_path(repo_root, attribute_path=None):
    if attribute_path:
        path = attribute_path
        if not os.path.isabs(path):
            path = os.path.join(repo_root, path)
        return os.path.abspath(path) if os.path.exists(path) else os.path.abspath(path)
    for path in default_label_path_candidates(repo_root):
        if os.path.exists(path):
            return os.path.abspath(path)
    return None


@lru_cache(maxsize=8)
def load_attribute_lookup(resolved_path):
    lookup = {}
    if not resolved_path or not os.path.exists(resolved_path):
        return lookup

    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError('SemKey attribute labels require pandas to load .df/.csv files.') from exc

    if resolved_path.endswith('.csv'):
        df = pd.read_csv(resolved_path)
    else:
        df = pd.read_pickle(resolved_path)

    key_columns = [col for col in ('input text', 'raw text', 'text', 'sentence') if col in df.columns]
    if not key_columns:
        return lookup

    for _, row in df.iterrows():
        item = {}
        if 'sentiment label' in df.columns:
            item['sentiment_label'] = normalize_sentiment_label(row.get('sentiment label'))
        elif 'sentiment_label' in df.columns:
            item['sentiment_label'] = normalize_sentiment_label(row.get('sentiment_label'))

        if 'topic_label' in df.columns:
            item['topic_label'] = normalize_topic_label(row.get('topic_label'))
        elif 'topic label' in df.columns:
            item['topic_label'] = normalize_topic_label(row.get('topic label'))

        item['length'] = _coerce_float(row.get('length')) if 'length' in df.columns else float('nan')
        item['surprisal'] = _coerce_float(row.get('surprisal')) if 'surprisal' in df.columns else float('nan')

        if not any(v is not None and not (isinstance(v, float) and math.isnan(v)) for v in item.values()):
            continue

        for key_col in key_columns:
            for key in matching_text_keys(row.get(key_col, '')):
                if key and key not in lookup:
                    lookup[key] = item

    return lookup


@lru_cache(maxsize=8)
def load_compact_attribute_lookup(resolved_path):
    compact_lookup = {}
    if not resolved_path or not os.path.exists(resolved_path):
        return compact_lookup

    lookup = load_attribute_lookup(resolved_path)
    for key, item in lookup.items():
        compact_key = compact_normalize_text(key)
        if compact_key and compact_key not in compact_lookup:
            compact_lookup[compact_key] = item
    return compact_lookup


def fuzzy_lookup(text, compact_lookup, min_ratio=0.985):
    target = compact_normalize_text(text)
    if not target:
        return {}
    exact = compact_lookup.get(target)
    if exact:
        return exact

    best_ratio = 0.0
    best_item = {}
    target_len = len(target)
    for key, item in compact_lookup.items():
        key_len = len(key)
        if not key_len or abs(key_len - target_len) / max(key_len, target_len) > 0.04:
            continue
        ratio = SequenceMatcher(None, target, key, autojunk=False).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_item = item
    return best_item if best_ratio >= min_ratio else {}


def validate_attribute_table(resolved_path, required_attributes=None):
    if not resolved_path or not os.path.exists(resolved_path):
        raise FileNotFoundError(
            'SemKey attributes are required, but no usable label file was found. '
            'Run scripts/prepare_labels.sh to generate dataset/ZuCo/zuco_semkey_labels.df first.'
        )

    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError('SemKey attribute labels require pandas to load .df/.csv files.') from exc

    if resolved_path.endswith('.csv'):
        df = pd.read_csv(resolved_path)
    else:
        df = pd.read_pickle(resolved_path)

    if required_attributes is None:
        required_attributes = ['sentiment', 'topic', 'length']
    column_map = {
        'sentiment': 'sentiment label',
        'topic': 'topic_label',
        'length': 'length',
        'surprisal': 'surprisal',
    }
    required_columns = [column_map[attr] for attr in required_attributes if attr in column_map]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f'SemKey attribute file is incomplete: {resolved_path}. '
            f'Missing columns: {missing_columns}. '
            'This usually means you are using zuco_label_8variants.df directly. '
            'Run scripts/prepare_labels.sh and use dataset/ZuCo/zuco_semkey_labels.df.'
        )

    empty_columns = [
        col for col in required_columns
        if df[col].isna().any() or (df[col].astype(str).str.strip() == '').any()
    ]
    if empty_columns:
        raise ValueError(
            f'SemKey attribute file has missing values: {resolved_path}. '
            f'Columns with missing values: {empty_columns}. '
            'Regenerate it with scripts/prepare_labels.sh; do not use the original MTV label file.'
        )


class SemKeyAttributeProvider:
    def __init__(self, repo_root, attribute_path=None, strict=False, required_attributes=None):
        self.repo_root = repo_root
        self.strict = bool(strict)
        self.required_attributes = set(required_attributes or ['sentiment', 'topic', 'length'])
        self.resolved_path = resolve_label_path(repo_root, attribute_path)
        if self.strict:
            validate_attribute_table(self.resolved_path, self.required_attributes)
        self.lookup = load_attribute_lookup(self.resolved_path)
        self.compact_lookup = load_compact_attribute_lookup(self.resolved_path)

    def get(self, text, task_key='<UNK>'):
        key = normalize_text(text)
        item = {}
        for lookup_key in matching_text_keys(text):
            item = self.lookup.get(lookup_key, {})
            if item:
                break
        if not item:
            item = fuzzy_lookup(text, self.compact_lookup)
        item = dict(item)

        if not item.get('topic_label'):
            item['topic_label'] = fallback_topic_from_task(task_key)
        if not math.isfinite(_coerce_float(item.get('length'))):
            item['length'] = float(len(key.split()))
        if 'surprisal' not in item:
            item['surprisal'] = float('nan')

        encoded = {
            'sentiment_label': encode_sentiment(item.get('sentiment_label')),
            'topic_label': encode_topic(item.get('topic_label')),
            'length': _coerce_float(item.get('length')),
            'surprisal': _coerce_float(item.get('surprisal')),
        }

        if self.strict:
            missing = []
            if 'sentiment' in self.required_attributes and encoded['sentiment_label'] < 0:
                missing.append('sentiment label')
            if 'topic' in self.required_attributes and encoded['topic_label'] < 0:
                missing.append('topic_label')
            if 'length' in self.required_attributes and not math.isfinite(encoded['length']):
                missing.append('length')
            if 'surprisal' in self.required_attributes and not math.isfinite(encoded['surprisal']):
                missing.append('surprisal')
            if missing:
                raise KeyError(f"Missing SemKey attributes {missing} for text: {key}")

        return encoded
