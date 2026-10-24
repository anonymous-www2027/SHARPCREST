import os
import time
import copy
import random
import json
from src.utils.main_training import BestValidation, evaluate_main

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from src.data_provider.data_factory import data_provider
from src.models.RAFT import Model
from src.utils.metrics import metric


class Exp_Long_Term_Forecast:
    """Minimal training and evaluation loop for RAFT with SHARP retrieval."""

    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.model = Model(args).float()

        main = getattr(args, 'main_protocol', False)
        if main:
            rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                   torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        train_data, _ = self._get_data('train')
        valid_data, _ = self._get_data('val')
        test_data, _ = self._get_data('test')
        self.model.prepare_dataset(train_data, valid_data, test_data)
        self.model = self.model.to(self.device)
        if main:
            random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
            if rng[3] is not None: torch.cuda.set_rng_state_all(rng[3])

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    @staticmethod
    def _unpack_batch(batch):
        index, batch_x, batch_y, _batch_x_mark, _batch_y_mark = batch
        return index, batch_x, batch_y

    def _loss_for_loader(self, loader, criterion, mode):
        losses = []
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                index, batch_x, batch_y = self._unpack_batch(batch)
                batch_x = batch_x.float().to(self.device)
                target = batch_y[:, -self.args.pred_len:].float().to(self.device)
                output = self.model(batch_x, index, mode=mode)
                feature_start = -1 if self.args.features == 'MS' else 0
                losses.append(criterion(
                    output[:, :, feature_start:],
                    target[:, :, feature_start:],
                ).item())
        self.model.train()
        return float(np.mean(losses))

    def _adjust_learning_rate(self, optimizer, epoch):
        if self.args.lradj == 'constant':
            return
        learning_rate = self.args.learning_rate * (0.5 ** (epoch - 1))
        for group in optimizer.param_groups:
            group['lr'] = learning_rate
        print(f'Updating learning rate to {learning_rate}')

    def train(self, setting):
        _, train_loader = self._get_data('train')
        _, valid_loader = self._get_data('val')
        _, test_loader = self._get_data('test')

        checkpoint_dir = os.path.join(self.args.checkpoints, setting)
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pth')

        optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        criterion = nn.MSELoss()
        best_loss = float('inf')
        best_state = None
        main = getattr(self.args, 'main_protocol', False)
        policy = BestValidation(getattr(self.args, 'patience', 0), getattr(self.args, 'min_delta', 0.))
        if main and self.args.data == 'panel':
            self.args._panel_data_bundle.forecast_scaler.save(os.path.join(checkpoint_dir, 'forecast_scaler.npz'))
        history = []

        for epoch in range(1, self.args.train_epochs + 1):
            start = time.perf_counter()
            losses = []
            self.model.train()
            for step, batch in enumerate(train_loader, start=1):
                index, batch_x, batch_y = self._unpack_batch(batch)
                batch_x = batch_x.float().to(self.device)
                target = batch_y[:, -self.args.pred_len:].float().to(self.device)
                optimizer.zero_grad()

                output = self.model(batch_x, index, mode='train')
                feature_start = -1 if self.args.features == 'MS' else 0
                loss = criterion(
                    output[:, :, feature_start:],
                    target[:, :, feature_start:],
                )
                if main and not all(torch.isfinite(t).all() for t in (batch_x, target, output, loss)):
                    raise ValueError('nonfinite training input/target/output/loss')
                loss.backward()
                if main and not all(torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None):
                    raise ValueError('nonfinite training gradient')
                optimizer.step()
                losses.append(loss.item())

                if step % 100 == 0:
                    print(f'\titers: {step}, epoch: {epoch} | loss: {loss.item():.7f}')

            valid_loss = self._loss_for_loader(valid_loader, criterion, 'valid')
            test_loss = None if main else self._loss_for_loader(test_loader, criterion, 'test')
            train_loss = float(np.mean(losses))
            elapsed = time.perf_counter() - start
            print(
                f'Epoch: {epoch}, cost time: {elapsed:.3f}s | '
                f'Train Loss: {train_loss:.7f} Valid Loss: {valid_loss:.7f} '
                f'Test Loss: {test_loss}'
            )

            improved, stop = policy.update(valid_loss, epoch)
            if improved:
                best_loss = valid_loss
                best_state = copy.deepcopy(self.model.state_dict())

            # RAFT type1 keeps the initial LR for epochs 1 and 2, then halves
            # it after every subsequent epoch.
            self._adjust_learning_rate(optimizer, epoch)
            if main:
                history.append(dict(epoch=epoch, validation_MSE=valid_loss, best_epoch=policy.best_epoch))
                if stop: break

        if best_state is None:
            raise RuntimeError('training completed without a validation checkpoint')
        torch.save(best_state, checkpoint_path)
        if main:
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            with open(os.path.join(checkpoint_dir, 'training_protocol.json'), 'w') as f:
                json.dump(dict(best_epoch=policy.best_epoch, best_validation_MSE=policy.best,
                    epochs_completed=epoch, best_checkpoint_reloaded=True, history=history), f, indent=2)
        # Non-main legacy runs retain their historical final-epoch immediate test.
        # Main runs above always reload the best-validation checkpoint.
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data('test')
        if test:
            checkpoint = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
            print(f'loading model: {checkpoint}')
            self.model.load_state_dict(
                torch.load(checkpoint, map_location=self.device)
            )

        if getattr(self.args, 'main_protocol', False):
            return evaluate_main(self, test_data, test_loader, setting)
        predictions = []
        targets = []
        self.model.eval()
        with torch.no_grad():
            for batch in test_loader:
                index, batch_x, batch_y = self._unpack_batch(batch)
                batch_x = batch_x.float().to(self.device)
                output = self.model(batch_x, index, mode='test').cpu().numpy()
                target = batch_y[:, -self.args.pred_len:].numpy()

                if test_data.scale and self.args.inverse:
                    output_shape = output.shape
                    target_shape = target.shape
                    output = test_data.inverse_transform(
                        output.reshape(-1, output.shape[-1])
                    ).reshape(output_shape)
                    target = test_data.inverse_transform(
                        target.reshape(-1, target.shape[-1])
                    ).reshape(target_shape)

                feature_start = -1 if self.args.features == 'MS' else 0
                predictions.append(output[:, :, feature_start:])
                targets.append(target[:, :, feature_start:])

        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)
        mae, mse, rmse, mape, mspe = metric(predictions, targets)
        print(f'mse:{mse}, mae:{mae}')

        result_dir = os.path.join('./results', setting)
        os.makedirs(result_dir, exist_ok=True)
        np.save(os.path.join(result_dir, 'metrics.npy'),
                np.asarray([mae, mse, rmse, mape, mspe]))
        np.save(os.path.join(result_dir, 'pred.npy'), predictions)
        np.save(os.path.join(result_dir, 'true.npy'), targets)
        with open('result_long_term_forecast.txt', 'a', encoding='utf-8') as result_file:
            result_file.write(f'{setting}\n')
            result_file.write(f'mse:{mse}, mae:{mae}\n\n')
        return mse, mae
