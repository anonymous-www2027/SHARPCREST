"""RAFT forecasting with optional SHARP search and CREST augmentation."""

import time

import torch
import torch.nn as nn

from src.layers.Retrieval import RetrievalTool


class Model(nn.Module):
    def __init__(self, configs, individual=False):
        super().__init__()
        if configs.task_name != 'long_term_forecast':
            raise ValueError('RAFT currently supports long_term_forecast only')
        if getattr(configs, 'fusion_mode', 'concat_projection') != 'concat_projection':
            raise ValueError('runtime branch supports concat_projection only')

        self._panel_config = configs
        self.device = configs.device
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.linear_x = nn.Linear(self.seq_len, self.pred_len)
        self.n_period = configs.n_period
        self.topm = configs.topm
        self.runtime_breakdown = getattr(configs, 'runtime_breakdown', False)
        self.candidate_search_strategy = getattr(
            configs, 'candidate_search_strategy', 'two_stage'
        )
        gpu_index = getattr(configs, 'gpu', 0)
        postprocess_device = getattr(
            configs, 'retrieval_postprocess_device', 'cpu'
        )
        mmr_device = getattr(configs, 'mmr_device', 'cpu')
        if postprocess_device == 'cuda':
            postprocess_device = f'cuda:{gpu_index}'
        if mmr_device == 'cuda':
            mmr_device = f'cuda:{gpu_index}'
        self.rt = RetrievalTool(
            crest_weighting=getattr(configs, "crest_weighting", "gaussian"),
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            channels=self.channels,
            n_period=self.n_period,
            topm=self.topm,
            hnsw_m=configs.hnsw_m,
            hnsw_ef_construction=configs.hnsw_ef_construction,
            hnsw_ef_search=configs.hnsw_ef_search,
            hnsw_seed=configs.hnsw_seed,
            retrieval_batch_size=configs.retrieval_batch_size,
            retrieval_backend=getattr(configs, 'retrieval_backend', 'hnsw'),
            panel_exact_max_candidates=getattr(configs, 'panel_exact_max_candidates', 6000000),
            retrieval_controller=getattr(
                configs, 'retrieval_controller', 'baseline'
            ),
            mmr_candidate_pool=getattr(configs, 'mmr_candidate_pool', 100),
            crest_profile=getattr(configs, 'crest_profile', 'stp'),
            stp_state_weight=getattr(configs, 'stp_state_weight', 0.5),
            stp_dispersion_weight=getattr(configs, 'stp_dispersion_weight', 0.25),
            stp_transition_weight=getattr(configs, 'stp_transition_weight', 0.25),
            profile_sections=getattr(configs, 'profile_sections', 6),
            profile_reliability_k=getattr(
                configs, 'profile_reliability_k', -1
            ),
            mean_profile_temperature=getattr(
                configs, 'mean_profile_temperature', 1.0
            ),
            tau_v=getattr(configs, 'tau_v', 1.0),
            sigma_min=getattr(configs, 'sigma_min', 0.05),
            sigma_max=getattr(configs, 'sigma_max', 0.30),
            lambda_min=getattr(configs, 'lambda_min', 0.30),
            lambda_max=getattr(configs, 'lambda_max', 0.90),
            hnsw_oversample_factor=getattr(
                configs, 'hnsw_oversample_factor', 5
            ),
            candidate_search_strategy=self.candidate_search_strategy,
            retrieval_seed=getattr(configs, 'seed', 0),
            retrieval_postprocess_device=postprocess_device,
            mmr_device=mmr_device,
            mmr_redundancy_source=getattr(
                configs, 'mmr_redundancy_source', 'value'
            ),
            mmr_value_repr=getattr(configs, 'mmr_value_repr', 'raw'),
            mmr_paa_segments=getattr(configs, 'mmr_paa_segments', 8),
            mmr_rp_dim=getattr(configs, 'mmr_rp_dim', 32),
            mmr_rp_seed=getattr(configs, 'mmr_rp_seed', 0),
            value_mmr_chunk_size=getattr(configs, 'value_mmr_chunk_size', 8),
            retrieval_aggregation_chunk_size=getattr(
                configs, 'retrieval_aggregation_chunk_size', 32
            ),
            runtime_breakdown=self.runtime_breakdown,
            hnsw_build_batch_size=getattr(
                configs, 'hnsw_build_batch_size', 4096
            ),
            memory_report=getattr(configs, 'memory_report', False),
            hnsw_build_num_threads=getattr(configs, 'hnsw_build_num_threads', -1),
            hnsw_search_num_threads=getattr(configs, 'hnsw_search_num_threads', 32),
            hnsw_recall_check=getattr(
                configs, 'hnsw_recall_check', False
            ),
            hnsw_recall_queries=getattr(
                configs, 'hnsw_recall_queries', 100
            ),
            hnsw_recall_seed=getattr(configs, 'hnsw_recall_seed', 0),
            hnsw_recall_splits=getattr(
                configs, 'hnsw_recall_splits', ('valid',)
            ),
            hnsw_recall_candidate_chunk_size=getattr(
                configs, 'hnsw_recall_candidate_chunk_size', 4096
            ),
            hnsw_recall_query_batch_size=getattr(
                configs, 'hnsw_recall_query_batch_size', 32
            ),
            hnsw_recall_device=getattr(
                configs, 'hnsw_recall_device', 'cpu'
            ),
            hnsw_recall_cache_dir=getattr(
                configs, 'hnsw_recall_cache_dir', ''
            ),
        )

        self.period_num = self.rt.period_num[-self.n_period:]
        self.retrieval_pred = nn.ModuleList([
            nn.Linear(self.pred_len // period, self.pred_len)
            for period in self.period_num
        ])
        # The only forecasting fusion in this branch: [direct; retrieval] H -> H.
        self.linear_pred = nn.Linear(2 * self.pred_len, self.pred_len)

    @staticmethod
    def _print_prepare_runtime(runtime):
        print('[RetrievalRuntime][prepare]')
        for name in (
            'data_pack_ms', 'profile_bank_iqr_ms', 'hnsw_build_ms',
            'exact_key_bank_ms', 'value_bank_ms', 'value_norm_bank_ms',
            'prepare_total_ms',
        ):
            print(f'{name}={runtime[name]:.3f}')

    def _print_split_runtime(self, split, runtime):
        print(f'[RetrievalRuntime][{split}]')
        print(f"queries={runtime['queries']}")
        for name in (
            'hnsw_search_ms', 'exact_search_ms', 'masking_fallback_ms',
            'profile_reliability_ms', 'mmr_ms',
            'baseline_aggregation_ms', 'gaussian_aggregation_ms',
            'split_total_ms', 'ms_per_query',
        ):
            print(f'{name}={runtime[name]:.3f}')
        if split == 'train':
            if self.candidate_search_strategy == 'wide':
                print(
                    f"wide_searches={runtime['wide_searches']}/"
                    f"{runtime['period_queries']}"
                )
            else:
                print(
                    f"wide_fallbacks={runtime['wide_fallbacks']}/"
                    f"{runtime['period_queries']}"
                )

    def prepare_dataset(self, train_data, valid_data, test_data):
        """Build exact/HNSW search state and cache augmentation for all splits."""
        if getattr(train_data, 'is_panel_dataset', False):
            from src.utils.panel_runtime import prepare_panel_retrieval
            return prepare_panel_retrieval(self, self._panel_config,
                                           train_data, valid_data, test_data)
        self.retrieval_dict = {}

        train_start = time.perf_counter()
        self.rt.prepare_dataset(train_data)
        if self.rt.crest_enabled:
            self.reliability_configuration = {
                'formula': 'cr_q=A_q*V_q', 'neighborhood': 'stp_ranked_spaced_up_to_n',
                'min_spacing': self.rt.pred_len, 'shortfall_policy': 'use_available',
                'tau_v': self.rt.tau_v, 'value_scale': self.rt.value_scale.tolist(),
            }
        prepare_ms = self.rt.prepare_runtime['prepare_total_ms']
        train_rt = self.rt.retrieve_all(train_data, train=True, split='train')
        train_split_ms = self.rt.split_runtime['train']['split_total_ms']
        # Preserve the historical meaning: Train Retrieval Time includes index
        # preparation, which permits direct comparison with the source branch.
        train_ms = (time.perf_counter() - train_start) * 1000.0
        print('Train Retrieval Time: {:.3f}ms | {:.6f}ms/sample'.format(
            train_ms, train_ms / max(len(train_data), 1)))

        valid_start = time.perf_counter()
        valid_rt = self.rt.retrieve_all(valid_data, train=False, split='valid')
        valid_ms = (time.perf_counter() - valid_start) * 1000.0
        print('Valid Retrieval Time: {:.3f}ms | {:.6f}ms/sample'.format(
            valid_ms, valid_ms / max(len(valid_data), 1)))

        test_start = time.perf_counter()
        test_rt = self.rt.retrieve_all(test_data, train=False, split='test')
        test_ms = (time.perf_counter() - test_start) * 1000.0
        print('Test Retrieval Time: {:.3f}ms | {:.6f}ms/sample'.format(
            test_ms, test_ms / max(len(test_data), 1)))

        total_ms = train_ms + valid_ms + test_ms
        total_samples = len(train_data) + len(valid_data) + len(test_data)
        print('Total Retrieval Time: {:.3f}ms | {:.6f}ms/sample'.format(
            total_ms, total_ms / max(total_samples, 1)))

        if self.runtime_breakdown:
            self._print_prepare_runtime(self.rt.prepare_runtime)
            for split in ('train', 'valid', 'test'):
                self._print_split_runtime(split, self.rt.split_runtime[split])
            accounted_train_ms = prepare_ms + train_split_ms
            print(
                '[RetrievalRuntime][train_accounting] '
                f'prepare_plus_split_ms={accounted_train_ms:.3f}'
            )

        for split, result in (
            ('train', train_rt), ('valid', valid_rt), ('test', test_rt)
        ):
            self.retrieval_dict[split] = result.detach().cpu()
        if self.rt.hnsw_recall_check:
            self.hnsw_recall_results = self.rt.evaluate_general_hnsw_recall(
                train_data,
                {'train': train_data, 'valid': valid_data, 'test': test_data},
            )
        if getattr(self._panel_config, 'main_protocol', False):
            train_stats = self.rt.split_runtime['train']
            masked = train_stats['period_queries']
            self.retrieval_runtime = dict(
                prepare=self.rt.prepare_runtime, splits=self.rt.split_runtime,
                retrieval_wall_seconds=(train_ms + valid_ms + test_ms)/1000,
                masked_training_queries=masked, wide_fallback_count=train_stats['wide_fallbacks'],
                fallback_ratio=train_stats['wide_fallbacks']/masked if masked else 0.)
            if self.rt.crest_enabled:
                self.retrieval_runtime['reliability_configuration'] = self.reliability_configuration
            import json
            from pathlib import Path
            log = Path(self._panel_config.log_dir)
            log.mkdir(parents=True, exist_ok=True)
            name = self._panel_config.model_id + '_' + self._panel_config.main_method + '_retrieval.json'
            (log/name).write_text(json.dumps(self.retrieval_runtime, indent=2))
        del self.rt

    def encoder(self, x, index, mode):
        """Fuse the direct forecast and learned projections of retrieved values."""
        batch_size, seq_len, channels = x.shape
        if seq_len != self.seq_len or channels != self.channels:
            raise ValueError(
                f'expected input [B, {self.seq_len}, {self.channels}], '
                f'got {tuple(x.shape)}'
            )

        x_offset = x[:, -1:, :].detach()
        x_norm = x - x_offset
        direct_prediction = self.linear_x(
            x_norm.permute(0, 2, 1)
        ).permute(0, 2, 1)

        cache_index = index.detach().cpu().long()
        # Cached values are augmentation representations. Learned projection
        # and fusion precede restoration of the query's last observed level.
        pred_from_retrieval = self.retrieval_dict[mode][
            :, cache_index
        ].to(x.device)
        retrieval_pred_list = []
        for period_idx, period_prediction in enumerate(pred_from_retrieval):
            period = self.period_num[period_idx]
            period_prediction = period_prediction.reshape(
                batch_size, self.pred_len // period, period, channels
            )[:, :, 0, :]
            period_prediction = self.retrieval_pred[period_idx](
                period_prediction.permute(0, 2, 1)
            ).permute(0, 2, 1)
            retrieval_pred_list.append(period_prediction.reshape(
                batch_size, self.pred_len, self.channels
            ))

        retrieval_prediction = torch.stack(
            retrieval_pred_list, dim=1
        ).sum(dim=1)
        prediction = torch.cat(
            [direct_prediction, retrieval_prediction], dim=1
        )
        prediction = self.linear_pred(
            prediction.permute(0, 2, 1)
        ).permute(0, 2, 1)
        return prediction.reshape(
            batch_size, self.pred_len, self.channels
        ) + x_offset

    def forecast(self, x_enc, index, mode):
        return self.encoder(x_enc, index, mode)

    def forward(self, x_enc, index, mode='train'):
        dec_out = self.forecast(x_enc, index, mode)
        return dec_out[:, -self.pred_len:, :]
