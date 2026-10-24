"""Export TSF-balanced nested cohorts consumable by reproduce.py."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .io import exclusive_output, file_hash, json_hash, write_json
from .population import Population, analyze, validate_table
from .protocol import ROOT, protocol
from .selection import select_scales


def load_table(path, population):
    """Accept only a sealed local TSF table or a checksum-pinned paper table."""
    path = Path(path); digest = file_hash(path); p = population.protocol
    historical = json.loads(Path(__file__).with_name('published_population.json').read_text()).get(p['dataset'])
    if historical and digest == historical['table_sha256']:
        base = protocol(p['dataset'])
        if p != base:
            raise ValueError('Historical TSF import requires the unmodified paper protocol')
        for name, expected in historical['canonical_files'].items():
            if population.files.get(name) != expected:
                raise ValueError(f'Historical TSF table/canonical source mismatch: {name}')
        actual = {n for n in population.files if n.endswith('.parquet')}
        if actual != set(historical['canonical_files']):
            raise ValueError('Historical TSF source file set differs')
        frame = pd.read_parquet(path)
        column = 'source_id' if p['dataset'] == 'quito_hour' else 'series_id'
        frame = frame.rename(columns={column: 'logical_id'})
        origin = 'checksum_verified_paper_population'
    else:
        seal = path.with_name('population.json')
        if not seal.exists():
            raise ValueError('Unrecognized TSF table: require its generated population.json provenance')
        saved = json.loads(seal.read_text())
        if not saved.get('completed') or saved['table_sha256'] != digest:
            raise ValueError('TSF table completion/checksum mismatch')
        if saved['key']['protocol'] != p or saved['key']['inputs'] != population.files:
            raise ValueError('TSF table uses different inputs or protocol')
        frame = pd.read_parquet(path); origin = 'verified_precomputed_population'
    return validate_table(frame, population), dict(origin=origin, table_sha256=digest)


def cohort(population, selection, scale, anchor, tsf_provenance):
    """Freeze source mapping, selected raw values, and the fixed-test anchor."""
    p = population.protocol; ds = p['dataset']; ids = selection['ids']
    selected = population.identities.set_index('logical_id').loc[ids]
    identities = [[('hour' if ds == 'quito_hour' else ds), r.source_dataset,
                   int(sid), int(r.canonical_series_id)] for sid, r in selected.iterrows()]
    selected_hash = hashlib.sha256(np.array(ids, dtype='<i8').tobytes()).hexdigest()
    identity_hash = hashlib.sha256(json.dumps(identities, separators=(',', ':')).encode()).hexdigest()
    raw_hash = population.bank_hash(ids)
    m = dict(schema_version=1, population_mode='nested_whole_series', dataset=ds,
        L=p['L'], H=p['H'], split=p['split'], nominal_scale=scale,
        selected_series_count=len(ids), ordered_source_ids=ids, anchor_source_ids=anchor,
        selected_identities=identities, selection_seed=p['selection_seed'],
        selected_series_hash=selected_hash, cohort_hash=identity_hash if ds == 'quito_hour' else selected_hash,
        selected_raw_bank_hash=raw_hash, actual_candidate_count=selection['actual_candidate_count'],
        provenance=dict(tsf=tsf_provenance, protocol=p, source_files_sha256=population.files,
                        selection='largest remainder quotas; nested deficit sampling; whole series'))
    reference_path = ROOT / 'configs/cohorts' / ds / f'{scale}.json'
    parity = dict(available=reference_path.exists(), matched=False)
    if reference_path.exists():
        ref = json.loads(reference_path.read_text())
        fields = ['L', 'H', 'split', 'ordered_source_ids', 'anchor_source_ids', 'selected_identities',
                  'cohort_hash', 'selected_series_hash', 'selected_raw_bank_hash', 'selection_seed']
        parity['checks'] = {key: ref[key] == m[key] for key in fields}
        parity['matched'] = all(parity['checks'].values())
        if parity['matched']:
            m['expected'] = ref.get('expected', {})
    m['published_cohort_verification'] = parity
    return m


def prepare(canonical_root, dataset, scales, output, overrides=None, workers=1,
            batch_size=256, tsf_root=None, tsf_table=None, require_published=False):
    """Analyze or reuse TSF, then write immutable nested cohort manifests."""
    with exclusive_output(output):
        return _prepare(canonical_root, dataset, scales, output, overrides, workers,
                        batch_size, tsf_root, tsf_table, require_published)


def _prepare(canonical_root, dataset, scales, output, overrides, workers,
             batch_size, tsf_root, tsf_table, require_published):
    if workers < 1 or batch_size < 1:
        raise ValueError('workers and batch_size must be positive')
    if tsf_root and tsf_table:
        raise ValueError('Use tsf_root or tsf_table, not both')
    p = protocol(dataset, overrides)
    if require_published and not (ROOT / 'configs/cohorts' / dataset / '100000.json').exists():
        raise ValueError('No published cohort exists for this dataset; omit --require-published')
    if require_published and dataset == 'extended_wikipedia' and not (tsf_root or tsf_table):
        raise ValueError('Paper Wiki TSF requires missing-preserving --tsf-root or a verified --tsf-table; '
                         'the zero-filled forecasting Parquet is insufficient')
    if tsf_root:
        p['tsf_input_policy'] = 'missing_preserving'
    elif p['tsf_input_policy'] != 'canonical' and not tsf_table:
        raise ValueError('missing_preserving mode requires tsf_root')
    output = Path(output)
    population = Population(canonical_root, p)
    # Changing the protocol or source population cannot reuse earlier outputs.
    request = dict(protocol=p, inputs=population.files, scales=sorted(set(scales)),
                   require_published=require_published,
                   tsf_table_sha256=file_hash(tsf_table) if tsf_table else None)
    alternate = Population(tsf_root, p) if tsf_root else None
    if alternate is not None:
        if not population.identities.equals(alternate.identities):
            raise ValueError('Alternate TSF source identity/order mismatch')
        for name in p['sources']:
            if not population.indexes[name]['start_timestamp'].equals(alternate.indexes[name]['start_timestamp']):
                raise ValueError('Alternate TSF source timestamps differ')
        request['tsf_inputs'] = alternate.files
    output.mkdir(parents=True, exist_ok=True)
    frozen = output / 'request.json'
    if frozen.exists():
        if json.loads(frozen.read_text()) != request:
            raise ValueError('Output belongs to another request; use a new output directory')
    elif any(output.iterdir()):
        raise ValueError('Output directory is nonempty and has no matching request.json')
    else:
        write_json(frozen, request)
    completion = output / 'completion.json'
    if completion.exists():
        saved = json.loads(completion.read_text())
        for name, digest in saved['files'].items():
            if file_hash(output / name) != digest:
                raise ValueError(f'Completed output checksum mismatch: {name}')
        return saved
    population.validate()
    if alternate is not None:
        alternate.validate()
        for _ in population.train_rows(alternate):
            pass
    if tsf_table:
        frame, provenance = load_table(tsf_table, population)
    else:
        frame = analyze(population, output / 'population', workers, batch_size, alternate)
        provenance = dict(origin=p['tsf_input_policy'],
                          table_sha256=file_hash(output / 'population/population_tsf.parquet'))
    selections = select_scales(frame, scales, p)
    anchor = selections[100000]['ids']; summary = []; files = {}
    for scale, selection in selections.items():
        m = cohort(population, selection, scale, anchor, provenance)
        if require_published and not m['published_cohort_verification']['matched']:
            raise ValueError(f'{dataset}/{scale}: generated cohort does not match a published cohort')
        directory = output / 'cohorts' / dataset
        path = directory / f'{scale}.json'
        if path.exists() and json.loads(path.read_text()) != m:
            raise ValueError('Existing cohort differs; refusing to overwrite')
        write_json(path, m)
        quota_path = directory / f'{scale}_quotas.csv'
        pd.DataFrame(selection['quotas']).to_csv(quota_path, index=False)
        table_path = directory / f'{scale}_series_tsf.parquet'
        selected = frame.set_index('logical_id').loc[selection['ids']].reset_index()
        selected.insert(0, 'selection_order', np.arange(len(selected)))
        selected['is_anchor'] = selected.logical_id.isin(anchor)
        selected.to_parquet(table_path, index=False)
        for artifact in (path, quota_path, table_path):
            files[str(artifact.relative_to(output))] = file_hash(artifact)
        summary.append(dict(scale=scale, selected_series=len(selection['ids']),
            actual_candidates=selection['actual_candidate_count'],
            published_match=m['published_cohort_verification']['matched'], cohort=str(path.relative_to(output))))
        print('COHORT_COMPLETE', dataset, summary[-1], flush=True)
    files['request.json'] = file_hash(frozen)
    if not tsf_table:
        for name in ['population/population.json', 'population/population_tsf.parquet']:
            files[name] = file_hash(output / name)
    saved = dict(status='PASS', cohorts=summary, files=files, protocol_hash=json_hash(p),
                 tsf_provenance=provenance, input_verified=True)
    write_json(completion, saved)
    return saved
