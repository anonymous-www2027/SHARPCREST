"""Frozen paper defaults for train-only TSF and whole-series sizing."""
import copy
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PRESETS = {
    'extended_wikipedia': dict(sampling_key='wikipedia', period=7, threshold=0.4,
        high_operator='>=', short_labels=False, anchor_series=61, channel=0,
        L=576, H=7, sources=['extended_wikipedia'], tsf_input_policy='canonical'),
    'quito_hour': dict(sampling_key='quito_hour', period=24, threshold=0.4,
        high_operator='>', short_labels=True, anchor_series=10, channel=0,
        L=576, H=288, sources=['quitobench_hour', 'quito_corpus_hour'], tsf_input_policy='canonical'),
    'tailedts_202401': dict(sampling_key='tailedts', period=24, threshold=0.4,
        high_operator='>=', short_labels=False, anchor_series=518, channel=0,
        L=168, H=48, sources=['tailedts_202401'], tsf_input_policy='canonical'),
}


def protocol(dataset, overrides=None):
    """Resolve dataset defaults, allowing explicit settings for other populations."""
    if not isinstance(dataset, str) or Path(dataset).name != dataset or dataset in ('', '.', '..'):
        raise ValueError('dataset must be a directory name under canonical_root')
    if dataset in PRESETS:
        p = copy.deepcopy(PRESETS[dataset])
        published = json.loads((ROOT / 'configs/cohorts' / dataset / '100000.json').read_text())
        p['split'] = published['split']
    else:
        p = dict(sampling_key=dataset, period=None, threshold=0.4, high_operator='>=',
                 short_labels=False, anchor_series=None, channel=0, L=None, H=None,
                 sources=[dataset], tsf_input_policy='canonical', split={})
    p.update(dataset=dataset, selection_seed=0)
    overrides = overrides or {}
    if set(overrides) - set(p):
        raise ValueError('Unknown protocol keys: ' + ', '.join(sorted(set(overrides) - set(p))))
    overrides = copy.deepcopy(overrides)
    split = overrides.pop('split', {})
    if not isinstance(split, dict):
        raise ValueError('split must be an object with train_end, valid_end and test_end')
    p.update(overrides)
    p['split'].update(split)
    if 'test_end' in split and 'expected_length' not in split:
        p['split']['expected_length'] = split['test_end']
    missing = [k for k in ('L', 'H', 'period') if p[k] is None]
    missing += ['split.' + k for k in ('train_end', 'valid_end', 'test_end') if k not in p['split']]
    if missing:
        raise ValueError('New datasets require explicit settings: ' + ', '.join(missing))
    for key in ['L', 'H', 'period', 'channel', 'selection_seed']:
        if not isinstance(p[key], int) or isinstance(p[key], bool):
            raise ValueError(f'{key} must be an integer')
    if p['anchor_series'] is not None and (type(p['anchor_series']) is not int or p['anchor_series'] < 1):
        raise ValueError('anchor_series must be a positive integer or null for automatic sizing')
    if p['selection_seed'] < 0:
        raise ValueError('selection_seed must be nonnegative')
    if p['dataset'] != dataset:
        raise ValueError('Protocol dataset differs from CLI dataset')
    s = p['split']
    for key in ['train_end', 'valid_end', 'test_end', 'expected_length']:
        if not isinstance(s[key], int) or isinstance(s[key], bool):
            raise ValueError(f'split.{key} must be an integer')
    if not (0 < s['train_end'] < s['valid_end'] < s['test_end'] == s['expected_length']):
        raise ValueError('Require 0 < train_end < valid_end < test_end == expected_length')
    if min(p['L'], p['H']) < 1 or p['period'] < 2:
        raise ValueError('L and H must be positive; period must be >=2')
    if p['channel'] < 0:
        raise ValueError('channel must be nonnegative')
    if not (p['L'] + p['H'] <= s['train_end'] and p['H'] <= s['valid_end'] - s['train_end']
            and p['H'] <= s['test_end'] - s['valid_end']):
        raise ValueError('Reference L/H do not fit the train/validation/test splits')
    if p['high_operator'] not in ('>', '>=') or not 0 <= p['threshold'] <= 1:
        raise ValueError('Invalid TSF threshold/operator')
    if p['tsf_input_policy'] not in ('canonical', 'missing_preserving'):
        raise ValueError('tsf_input_policy must be canonical or missing_preserving')
    if not p['sources'] or len(p['sources']) != len(set(p['sources'])):
        raise ValueError('Source directories must be nonempty and unique')
    if any(Path(x).name != x or x in ('.', '..') for x in p['sources']):
        raise ValueError('Sources must be directory names under canonical_root')
    if dataset != 'quito_hour' and p['sources'] != [dataset]:
        raise ValueError('Non-Quito source directory must match dataset name')
    return p


def regimes(p):
    labels = ['L', 'H'] if p['short_labels'] else ['LOW', 'HIGH']
    separator = '' if p['short_labels'] else '_'
    return [separator.join(x) for x in itertools.product(labels, repeat=3)] + ['UNDEFINED']
