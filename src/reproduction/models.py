"""Dispatch optional RQ1 frameworks while retaining their native model contracts."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from datetime import datetime, timezone
from src.reproduction.model_data import ROOT, GENERAL, WEB_LENGTHS, HORIZONS, general_csv

METHODS = ('BASE', 'SHARP', 'CREST', 'S&C')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--framework', choices=('pir', 'knn_mts'), required=True)
    p.add_argument('--data-type', choices=('web', 'general'), required=True)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--dataset', choices=GENERAL+tuple(WEB_LENGTHS), required=True)
    p.add_argument('--method', choices=METHODS+('PIR', 'KNN'), required=True)
    p.add_argument('--scale', type=int, default=200000, choices=[200000])
    p.add_argument('--pred-len', type=int)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output', type=Path)
    p.add_argument('--cache-dir', type=Path, default=ROOT/'outputs/model-cache')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('extra', nargs=argparse.REMAINDER)
    return p


def resolve(options, materialize=False):
    """Translate common options to each model's independently preserved parser."""
    web = options.dataset in WEB_LENGTHS
    if web != (options.data_type == 'web'):
        raise ValueError('Dataset does not belong to data-type')
    aliases = {'pir': 'PIR', 'knn_mts': 'KNN'}
    method = 'BASE' if options.method == aliases[options.framework] else options.method
    if method not in METHODS:
        raise ValueError('Baseline method does not match framework')
    h = options.pred_len or (HORIZONS[options.dataset][0] if web else 96)
    if h not in (HORIZONS[options.dataset] if web else [96, 192, 336, 720]):
        raise ValueError('Horizon is outside RQ1')
    output = options.output or ROOT/'outputs'/options.framework/(options.dataset+f'-H{h}-s{options.seed}-'+method.replace('&','_')+'-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
    output = output.resolve(); cache = options.cache_dir.resolve()
    extra = options.extra
    if extra and extra[0] != '--':
        raise ValueError('Native options must follow --')
    extra = extra[1:] if extra else []
    if options.framework == 'pir':
        from src.pir.runner import build_parser
        name = {'BASE': 'PIR', 'SHARP': 'PIR_SHARP', 'CREST': 'PIR_CREST', 'S&C': 'PIR_SHARP_CREST'}[method]
        argv = ['--result_dir', str(output), '--retrieval_method', name, '--seed', str(options.seed),
                '--pred_len', str(h), '--shared_pretraining_cache', str(cache/'pir-pretraining')]
        if web:
            from src.reproduction.model_data import cohort
            cohort(options.dataset)
            directory = ROOT/'configs/cohorts'/options.dataset
            argv += ['--panel_root', str(options.data_dir.resolve()), '--panel_manifest', str(directory/'200000.json'),
                     '--panel_anchor_manifest', str(directory/'100000.json'), '--data', 'custom',
                     '--enc_in', '1', '--dec_in', '1', '--c_out', '1']
        else:
            path = general_csv(options.data_dir, options.dataset)
            channels = 21 if options.dataset == 'Weather' else 8 if options.dataset == 'Exchange' else 7
            argv += ['--root_path', str(path.parent), '--data_path', path.name,
                     '--data', {'Weather':'weather', 'Exchange':'exchange_rate'}.get(options.dataset, options.dataset),
                     '--enc_in', str(channels), '--dec_in', str(channels), '--c_out', str(channels)]
        args = build_parser().parse_args(argv+extra)
        if args.learning_rate != .0001 or args.refine_lr != .0001 or list(args.stp_weights) != [.5,.25,.25]:
            raise ValueError('Public RQ1 fixes LR=1e-4 and CREST weights=(0.5,0.25,0.25)')
        if (args.retrieval_method, args.seed, args.pred_len, Path(args.result_dir)) != (name, options.seed, h, output):
            raise ValueError('Use common options for method, seed, horizon and output')
    else:
        from src.knn_mts.runner import parser as native_parser
        name = {'BASE':'KNN','SHARP':'KNN_SHARP','CREST':'KNN_CREST','S&C':'KNN_SHARP_CREST'}[method]
        argv = ['--dataset', options.dataset, '--output', str(output), '--seed', str(options.seed),
                '--method', name, '--pred-len', str(h), '--value-transfer', 'query_last',
                '--search-backend', 'auto' if 'SHARP' in name else 'faiss_cpu']
        if web:
            from src.reproduction.model_data import prepare_knn_panel
            panel = cache/'panels'/options.dataset/f'H{h}'
            if materialize:
                prepare_knn_panel(options.data_dir, options.dataset, h, panel)
            argv += ['--panel-artifact', str(panel), '--seq-len', str(WEB_LENGTHS[options.dataset]),
                     '--epochs', '100', '--patience', '10']
        else:
            argv += ['--csv', str(general_csv(options.data_dir, options.dataset))]
        args = native_parser().parse_args(argv+extra)
        if args.learning_rate != .0001 or list(args.stp_weights) != [.5,.25,.25]:
            raise ValueError('Public RQ1 fixes LR=1e-4 and CREST weights=(0.5,0.25,0.25)')
        if (args.method, args.seed, args.pred_len, args.output) != (name, options.seed, h, output):
            raise ValueError('Use common options for method, seed, horizon and output')
    return args


def run_knn(args, cache):
    """Share one validated DLinear checkpoint across retrieval methods."""
    from src.knn_mts import runner
    from src.preprocessing.io import exclusive_output, file_hash
    if args.preflight_only or args.checkpoint:
        return runner.run(args)
    data = (runner.PanelData(args.panel_artifact, args.dataset, args.seq_len, args.pred_len)
            if args.dataset in WEB_LENGTHS else runner.GeneralData(args.dataset, args.csv, args.seq_len, args.pred_len))
    fields = ('seq_len','pred_len','seed','individual','epochs','patience','batch_size','eval_batch_size',
              'learning_rate','weight_decay','lradj','lr_milestones','lr_gamma','num_workers','threads','device')
    import torch
    payload = dict(data=data.audit, settings={k:getattr(args,k) for k in fields},
                   sources=runner.training_source_hashes(), torch=torch.__version__, cuda=torch.version.cuda,
                   gpu=torch.cuda.get_device_name() if args.device == 'cuda' and torch.cuda.is_available() else None)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    directory = cache/'dlinear'/digest
    with exclusive_output(directory):
        checkpoint = directory/'checkpoint.pt'; receipt = directory/'receipt.json'
        if receipt.is_file():
            record = json.loads(receipt.read_text())
            if record['key'] != payload or file_hash(checkpoint) != record['sha256']:
                raise ValueError('Shared DLinear checkpoint or training contract changed')
            args.checkpoint = checkpoint
        runner.run(args)
        if not receipt.exists():
            directory.mkdir(parents=True, exist_ok=True)
            temporary = directory/'checkpoint.tmp'
            shutil.copy2(args.output/'checkpoint.pt', temporary); temporary.replace(checkpoint)
            runner.write_json(receipt, dict(key=payload, sha256=file_hash(checkpoint)))


def main(argv=None):
    options = parser().parse_args(argv)
    args = resolve(options, materialize=not options.dry_run)
    if options.dry_run:
        print(json.dumps(dict(framework=options.framework, arguments=vars(args)), default=str, indent=2))
        return
    if options.framework == 'pir':
        from src.pir.runner import run
        run(args)
    else:
        run_knn(args, options.cache_dir.resolve())
    if getattr(args, 'preflight_only', False):
        return
    from src.preprocessing.io import write_json, file_hash
    output = Path(args.result_dir) if options.framework == 'pir' else args.output
    method = {'PIR':'BASE','KNN':'BASE'}.get(options.method, options.method)
    write_json(output/'public_protocol.json', dict(framework=options.framework, dataset=options.dataset,
        horizon=args.pred_len, seed=options.seed, method=method, learning_rate=.0001,
        crest_weights=[.5,.25,.25], training_batch=args.batch_size,
        arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if not k.startswith('_')}))
    write_json(output/'completion.json', dict(state='complete', hashes={str(p.relative_to(output)):file_hash(p)
        for p in sorted(output.rglob('*')) if p.is_file() and p.name not in ('completion.json','status.json')}))
