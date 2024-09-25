import random
import datasets
import pandas as pd
import numpy as np

SEPARATOR = '<<<SEP>>>'


DATASETS = ['writing', 'english', 'german', 'pubmed', 'guardian', 'medical']


def load_pubmed(cache_dir):
    data = datasets.load_dataset('pubmed_qa', 'pqa_labeled', split='train', cache_dir=cache_dir)
    
    # combine question and long_answer
    data = [f'Question: {q} Answer:{SEPARATOR}{a}' for q, a in zip(data['question'], data['long_answer'])]

    return data


def process_prompt(prompt):
    return prompt.replace('[ WP ]', '').replace('[ OT ]', '')


def process_spaces(story):
    return story.replace(
        ' ,', ',').replace(
        ' .', '.').replace(
        ' ?', '?').replace(
        ' !', '!').replace(
        ' ;', ';').replace(
        ' \'', '\'').replace(
        ' ’ ', '\'').replace(
        ' :', ':').replace(
        '<newline>', '\n').replace(
        '`` ', '"').replace(
        ' \'\'', '"').replace(
        '\'\'', '"').replace(
        '.. ', '... ').replace(
        ' )', ')').replace(
        '( ', '(').replace(
        ' n\'t', 'n\'t').replace(
        ' i ', ' I ').replace(
        ' i\'', ' I\'').replace(
        '\\\'', '\'').replace(
        '\n ', '\n').strip()


def load_writing(cache_dir=None):
    writing_path = 'data/writingPrompts'
    
    with open(f'{writing_path}/valid.wp_source', 'r') as f:
        prompts = f.readlines()
    with open(f'{writing_path}/valid.wp_target', 'r') as f:
        stories = f.readlines()
    
    prompts = [process_prompt(prompt) for prompt in prompts]
    joined = [process_spaces(prompt + " " + story) for prompt, story in zip(prompts, stories)]
    filtered = [story for story in joined if 'nsfw' not in story and 'NSFW' not in story]

    random.seed(0)
    random.shuffle(filtered)

    return filtered


def load_language(language, cache_dir):
    # load either the english or german portion of the wmt16 dataset
    assert language in ['en', 'de']
    d = datasets.load_dataset('wmt16', 'de-en', split='train', cache_dir=cache_dir)
    docs = d['translation']
    desired_language_docs = [d[language] for d in docs]
    lens = [len(d.split()) for d in desired_language_docs]
    sub = [d for d, l in zip(desired_language_docs, lens) if l > 100 and l < 150]
    return sub


def load_german(cache_dir):
    return load_language('de', cache_dir)


def load_english(cache_dir):
    return load_language('en', cache_dir)


def load(name, cache_dir, **kwargs):
    if name in DATASETS:
        load_fn = globals()[f'load_{name}']
        return load_fn(cache_dir=cache_dir, **kwargs)
    else:
        raise ValueError(f'Unknown dataset {name}')


def create_range_point_dict(point_indices, len_sample, min_words=55, max_words=200):
    result = {}
    for point in point_indices:
        max_range = len_sample - point + 1
        if max_range < min_words:
            result[point] = None
        else:
            max_range = min(max_words, max_range)
            result[point] = max_range

    if all(value is None for value in result.values()):
        return False
    else:
        result = {k: v for k, v in result.items() if v is not None}
        return result


def sample_text(sample, min_words=55, max_words=200):
    sample = sample.split()
    point_indices = [i for i in range(len(sample)) if "." in sample[i]]

    if len(sample) < min_words or len(point_indices) == 0:
        return None

    ranges = create_range_point_dict(point_indices, len(sample))

    if ranges == False:
        return None

    # Plus one, as we want to start after a dot
    point_chosen_begin = np.random.choice(list(ranges.keys())) + 1

    # Minus one, as we want to choose the specific item
    chosen_length = min_words if ranges[point_chosen_begin - 1] == min_words else np.random.randint(min_words, ranges[point_chosen_begin - 1])
    point_chosen_end = point_chosen_begin + chosen_length

    return ' '.join(sample[point_chosen_begin : point_chosen_end])


def sample_beginning_text(sample, min_words=55, max_words=200):
    sample = sample.split()
    if len(sample) < min_words:
        return None

    # Plus one, as we want to start after a dot
    point_chosen_begin = 0

    max_range = min(max_words, len(sample) - point_chosen_begin)

    chosen_length = np.random.randint(min_words, max_range)
    point_chosen_end = point_chosen_begin + chosen_length

    return ' '.join(sample[point_chosen_begin : point_chosen_end])


def load_job_posts(cache_dir=None, min_words=55, max_words=200):
    job_posts_path = 'data/monster_com-job_sample.csv'
    data = pd.read_csv(job_posts_path)
    data = data.dropna()

    data["sampled"] = data["job_description"].apply(lambda x: sample_beginning_text(x))
    data = data.dropna()

    data["sampled"] = data["sampled"].apply(lambda x: process_spaces(x))

    samples = data["sampled"].values

    random.seed(0)
    random.shuffle(samples)

    return samples


def load_guardian(cache_dir=None, min_words=55, max_words=200):
    guardian_path = 'data/guardian_articles.csv'
    data = pd.read_csv(guardian_path)
    data = data.dropna()

    data["sampled"] = data["bodyContent"].apply(lambda x: sample_text(x))
    data = data.dropna()

    data["sampled"] = data["sampled"].apply(lambda x: process_spaces(x))

    samples = data["sampled"].values

    random.seed(0)
    random.shuffle(samples)

    return samples


def load_medical(cache_dir=None, min_words=55, max_words=200):
    medical_path = 'data/train_medical.dat'
    data = pd.read_csv(medical_path, sep='\t', header=None)
    data.rename(columns = {0: 'conditions', 1: 'full_text'}, inplace=True)

    data = data.dropna()

    data["sampled"] = data["full_text"].apply(lambda x: sample_text(x))
    data = data.dropna()

    data["sampled"] = data["sampled"].apply(lambda x: process_spaces(x))

    samples = data["sampled"].values

    random.seed(0)
    random.shuffle(samples)

    return samples