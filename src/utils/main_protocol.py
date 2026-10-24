"""Frozen main-study configuration; no result-dependent routing."""
METHODS = {'RAFT': ('exact','baseline'), 'RAFT_SHARP': ('hnsw','baseline'),
           'RAFT_CREST': ('exact','crest'), 'RAFT_SHARP_CREST': ('hnsw','crest')}
WEB = {
    'quito_hour': dict(L=576,H=[48,288,512],C=1,N=10,m=24,panel='quito_corpus_hour',hash='f8d563d0d6d7924384456a465ad1b37039fe9d88135547be2426eef0bea9a6a9'),
    'quito_min': dict(L=576,H=[48,288,512],C=1,N=83,m=144,panel='quito_corpus_min',hash='57cda7de4c8489726cc88d52b274a89b8ce081a6ee8f12efb9e58e48c6ba91f6'),
    'tailedts_202401': dict(L=168,H=[24,48,96],C=1,N=518,m=168,panel='tailedts_202401',hash='480ebe4cbed717f1d453a69fcd71f0398930860a88c604f7f2272b9bd74bb40f'),
    'extended_wikipedia': dict(L=576,H=[7,28,59],C=1,N=61,m=7,panel='extended_wikipedia',hash='481ce76593df73ecaf052213a40256f30ec33ce41b2c36f671dd3307ebd23332'),
}
GENERAL = {'ETTh1':7,'ETTh2':7,'ETTm1':7,'ETTm2':7,'Exchange':8,'Weather':21}


def validate_main_args(args):
    if isinstance(args.seed, bool) or args.seed not in (0,1,2):
        raise ValueError('main experiment seed must be 0, 1, or 2')
    expected = dict(hnsw_m=32,hnsw_ef_construction=400,hnsw_ef_search=400,hnsw_seed=args.seed,
        hnsw_oversample_factor=5,candidate_search_strategy='two_stage',seed=args.seed,n_period=1,topm=40,
        mmr_candidate_pool=100,profile_reliability_k=40,crest_profile='stp',profile_sections=6,
        stp_state_weight=.5,stp_dispersion_weight=.25,stp_transition_weight=.25,
        mean_profile_temperature=1.,lambda_min=.30,lambda_max=.90,sigma_min=.05,sigma_max=.30,
        mmr_redundancy_source='value',mmr_value_repr='raw',batch_size=32,learning_rate=1e-4,
        lradj='type1',features='M',min_delta=0.,num_workers=0,itr=1,use_amp=False,inverse=False,
        candidate_stride=1,query_stride=1)
    if args.main_method not in METHODS:
        raise ValueError('main_method must be one of the four frozen methods')
    backend,controller=METHODS[args.main_method]
    expected.update(retrieval_backend=backend,retrieval_controller=controller)
    if args.main_dataset in WEB:
        spec=WEB[args.main_dataset]
        expected.update(data='panel',panel_dataset=spec['panel'],seq_len=spec['L'],enc_in=1,
                        train_epochs=100,patience=10,crest_weighting='adaptive_softmax',label_len=0)
        horizons=spec['H']
        if args.series_fraction is not None or args.max_series is not None or args.series_subset_path or args.panel_group:
            raise ValueError('main web cohorts are frozen; subset/group overrides forbidden')
    elif args.main_dataset in GENERAL:
        expected.update(data=args.main_dataset if args.main_dataset.startswith('ETT') else 'custom',
                        seq_len=720,enc_in=GENERAL[args.main_dataset],train_epochs=10,patience=0,
                        crest_weighting='gaussian')
        horizons=[96,192,336,720]
    else:
        raise ValueError('unknown frozen main dataset')
    if args.pred_len not in horizons:
        raise ValueError('horizon outside frozen main protocol')
    for key,value in expected.items():
        if getattr(args,key,None)!=value:
            raise ValueError(f'main protocol requires {key}={value!r}; got {getattr(args,key,None)!r}')
    return expected
