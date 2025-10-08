import warnings
warnings.filterwarnings('ignore')

import torch
import joblib
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from loguru import logger
from copy import deepcopy
from torch.nn import functional as F
from typing import List, Dict, Callable
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions.categorical import Categorical

from .net import NetP, NetG
from .operators import generate_operators, ops_rank, _nanstd


class AlphaForge:
    def __init__(
        self,
        feature_data: Dict[str, torch.Tensor],
        window_sizes: List[int],
        metric: Callable,
        hidden_size=64,
        max_length=20,
        device='cuda',
        ):

        self.feature_data = feature_data
        self.metric = metric
        self.max_length = max_length
        self.device = device

        self.sep = ['SEP']
        self.unary_ops, self.binary_ops = generate_operators(window_sizes)
        self.unary_op_names, self.binary_op_names = list(self.unary_ops.keys()), list(self.binary_ops.keys())
        self.feature_names = list(self.feature_data.keys())
        self.action_space = self.unary_op_names + self.binary_op_names + self.feature_names + self.sep
        self.action_size = len(self.action_space)

        self.offset_unary = 0
        self.offset_binary = self.offset_unary + len(self.unary_ops)
        self.offset_feature = self.offset_binary + len(self.binary_ops)
        self.offset_sep = self.offset_feature + len(self.feature_data)

        self.netp = NetP(self.action_size, hidden_size, dropout=0.1).to(self.device)
        self.netg = NetG(self.action_size, hidden_size, self.max_length + 1).to(self.device)

        self.alpha_pool = {}

    def _reset_net(self):
        self.netp.reset_params()
        self.netg.reset_params()

    def _action_to_token(self, action: int) -> str: 
        if action < self.offset_unary or action > self.offset_sep:
            raise ValueError
        elif action < self.offset_binary:
            return self.unary_op_names[action - self.offset_unary]
        elif action < self.offset_feature:
            return self.binary_op_names[action - self.offset_binary]
        elif action < self.offset_sep:
            return self.feature_names[action - self.offset_feature]
        elif action == self.offset_sep:
            return 'SEP' # the end token
        else:
            raise ValueError

    def _get_valid_actions(self, stack, state):
        valid_actions = torch.zeros(self.action_size, dtype=torch.bool, device=self.device)

        if len(state) == self.max_length: # only SEP is allowed
            valid_actions[self.offset_sep] = 1
        elif len(state) == self.max_length - 1 and len(stack) == 1:
            valid_actions[self.offset_unary: self.offset_binary] = 1 # only unary ops are allowed
        elif len(state) == self.max_length - 1 and len(stack) == 2:
            valid_actions[self.offset_binary: self.offset_feature] = 1 # only binary ops are allowed

        elif len(stack) == 0: # only features are allowed
            valid_actions[self.offset_feature: self.offset_sep] = 1
        elif len(stack) == 1: # only features and unary ops and SEP are allowed
            valid_actions[self.offset_unary: self.offset_binary] = 1
            valid_actions[self.offset_feature: self.offset_sep] = 1
            valid_actions[self.offset_sep] = 1
        elif len(stack) == 2: # only unary ops and binary ops are allowed
            valid_actions[self.offset_unary: self.offset_feature] = 1
        else:
            raise ValueError
        
        if state.shape[0] > 0:
            last_action = state[-1].argmax().item()
            if 'ops_neg' in self._action_to_token(last_action):
                valid_actions[last_action] = 0 # prevent double negation
            elif 'ops_rank' in self._action_to_token(last_action):
                valid_actions[last_action] = 0 # prevent double rank
            elif 'ops_inv' in self._action_to_token(last_action):
                valid_actions[last_action] = 0 # prevent double inversion
            elif 'ops_abs' in self._action_to_token(last_action):
                valid_actions[last_action] = 0 # prevent double absolute

        return valid_actions

    def _step_action(self, stack: List[torch.Tensor], action: int):
        if action < self.offset_unary or action > self.offset_sep:
            raise ValueError

        token = self._action_to_token(action)
        if action < self.offset_binary: # action = unary ops
            data = stack.pop()
            operator = self.unary_ops[token]
            result = operator(data)
            stack.append(result)
        elif action < self.offset_feature: # action = binary ops
            data1 = stack.pop()
            data2 = stack.pop()
            operator = self.binary_ops[token]
            result = operator(data1, data2)
            stack.append(result)
        elif action < self.offset_sep: # action = features
            stack.append(self.feature_data[token])
        elif action == self.offset_sep: # action = SEP
            pass

    def _sample_factor(self):
        stack = []
        state = torch.zeros(self.max_length + 1, self.action_size, device=self.device, dtype=torch.float32) # +1 for SEP

        for t in range(self.max_length + 1):
            valid_actions = self._get_valid_actions(stack, state[:t])
            masked = torch.where(valid_actions, 1.0, -1e5)
            action = Categorical(logits=masked).sample().item()

            state[t, action] = 1.0
            self._step_action(stack, action)
            
            if action == self.offset_sep:
                break
        
        assert len(stack) == 1
        factor = stack.pop()
        metric = self.metric(factor)
        return state, metric
    
    def _sample_factor_from_logits(self, logits):
        stack, one_hot = [], []
        state = torch.zeros(self.max_length + 1, self.action_size, device=self.device, dtype=torch.float32) # +1 for SEP

        for t in range(self.max_length + 1):
            valid_actions = self._get_valid_actions(stack, state[:t])
            masked = torch.where(valid_actions, logits[t], -1e5)
            one_hot.append(masked)

            with torch.no_grad():
                action = masked.argmax().item()

            state[t, action] = 1.0
            self._step_action(stack, action)
            
            if action == self.offset_sep:
                break

        assert len(stack) == 1
        factor = stack.pop()
        metric = self.metric(factor)

        one_hot = torch.stack(one_hot, dim=0)
        one_hot = F.gumbel_softmax(one_hot, hard=True)
        one_hot = F.pad(one_hot, (0, 0, 0, self.max_length + 1 - one_hot.shape[0]), value=0)

        return state, metric, one_hot
    
    def _state_to_expression(self, state) -> str:
        stack = []
        for t in range(state.shape[0]):
            action = state[t].argmax().item()
            token = self._action_to_token(action)

            if token in self.feature_data:
                stack.append(token)
            elif token in self.unary_ops:
                x = stack.pop()
                stack.append(token + '(' + x + ')')
            elif token in self.binary_ops:
                y = stack.pop()
                x = stack.pop()
                stack.append(token + '(' + x + ',' + y + ')')
            elif token == 'SEP':
                break
        return stack[0]        

    def _evaluate_factor(self, expr, corr_threshold) -> bool:
        factor = self.calculate_expression(expr)
        x_s = (factor - factor.nanmean(dim=1, keepdim=True)) / _nanstd(factor, dim=1, keepdim=True)
        for k, v in self.alpha_pool.items():
            alpha = self.calculate_expression(k)
            y_s = (alpha - alpha.nanmean(dim=1, keepdim=True)) / _nanstd(alpha, dim=1, keepdim=True)
            corr = (x_s * y_s).nanmean(dim=1).nanmean()
            corr = torch.abs(corr)
            if corr > corr_threshold:
                return False
        return True

    def _train_netp(self, sampled_matrix, sampled_metrics, batch_size, num_epochs, learning_rate, early_stopping):
        optimizer = torch.optim.Adam(self.netp.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        X, y = torch.stack(sampled_matrix, dim=0), torch.stack(sampled_metrics, dim=0)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
        
        best_val_loss = np.inf
        patience_counter = 0
        best_model = deepcopy(self.netp)

        for epoch in range(num_epochs):
            self.netp.train()
            train_loss = 0.0
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = self.netp(x_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            self.netp.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    y_pred = self.netp(x_batch)
                    loss = criterion(y_pred, y_batch)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            
            # logging
            logger.info(
                f'NetP Epochs: {epoch + 1}, '
                f'Train Loss: {train_loss}, '
                f'Val Loss: {val_loss}'
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model = deepcopy(self.netp)
            else:
                patience_counter += 1
                if patience_counter >= early_stopping:
                    logger.info(f'Early stopping at epoch {epoch + 1}')
                    break

        self.netp = deepcopy(best_model)

    def _train_netg(self, sampled_states, sampled_metrics, batch_size, num_epochs, learning_rate, early_stopping, ic_threshold, corr_threshold):
        optimizer = torch.optim.Adam(self.netg.parameters(), lr=learning_rate)
        best_metric = -np.inf
        patience_counter = 0
        
        self.netp.eval()
        self.netg.train()

        for epoch in range(num_epochs):
            optimizer.zero_grad()

            z_1 = torch.randn(batch_size, self.action_size, device=self.device)
            z_2 = torch.randn(batch_size, self.action_size, device=self.device)
            logits_1, logits_2 = self.netg(z_1), self.netg(z_2)
            
            batch_metric = 0.0
            one_hots_1, one_hots_2 = [], []

            for i in tqdm(range(batch_size), desc='Sampling factors from NetG'):
                state, metric, one_hot = self._sample_factor_from_logits(logits_1[i])
                sampled_states.append(state)
                sampled_metrics.append(metric)
                one_hots_1.append(one_hot)
                batch_metric += metric.item()

                if metric > ic_threshold:
                    expr = self._state_to_expression(state)
                    if expr not in self.alpha_pool and self._evaluate_factor(expr, corr_threshold):
                        self.alpha_pool[expr] = metric.item()

                state, metric, one_hot = self._sample_factor_from_logits(logits_2[i])
                one_hots_2.append(one_hot)

            one_hots_1, one_hots_2 = torch.stack(one_hots_1, dim=0), torch.stack(one_hots_2, dim=0)
            pred_1, pred_2 = self.netp(one_hots_1), self.netp(one_hots_2)

            similarity = torch.sum(one_hots_1 * one_hots_2, dim=-1).sum(dim=-1) / (self.max_length + 1)
            similarity = torch.relu(similarity - corr_threshold) ** 2
            similarity_loss = similarity.mean()
            pred_loss = 1 - pred_1.mean()

            loss = pred_loss + similarity_loss
            loss.backward()
            optimizer.step()

            batch_metric /= batch_size
            logger.info(
                f'NetG Epochs: {epoch + 1}, '
                f'Loss: {loss.item()}, '
                f'Metric: {batch_metric}'
            )

            if batch_metric > best_metric:
                best_metric = batch_metric
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping:
                    logger.info(f'Early stopping at epoch {epoch + 1}')
                    break

    def train_model(
        self,
        max_factors=100,
        init_sample_size=4096,
        max_sample_size=20000,
        batch_size_p=128,
        num_epochs_p=1000,
        learning_rate_p=1e-4,
        early_stopping_p=20,
        batch_size_g=128,
        num_epochs_g=200,
        learning_rate_g=1e-3,
        early_stopping_g=10,
        ic_threshold=0.05,
        corr_threshold=0.7,
        result_dir='./tmp/'
        ):

        sampled_states, sampled_metrics = [], []

        for i in tqdm(range(init_sample_size), desc='Sampling initial factors'):
            state, metric = self._sample_factor()
            sampled_states.append(state)
            sampled_metrics.append(metric)

        while len(self.alpha_pool) < max_factors:
            self._reset_net()
            self._train_netp(sampled_states, sampled_metrics, batch_size_p, num_epochs_p, learning_rate_p, early_stopping_p)
            self._train_netg(sampled_states, sampled_metrics, batch_size_g, num_epochs_g, learning_rate_g, early_stopping_g, ic_threshold, corr_threshold)
            logger.info(f'Current number of alphas in pool: {len(self.alpha_pool)}')
            joblib.dump(self.alpha_pool, result_dir + 'aff_alpha_pool.pkl')

            if len(sampled_states) > max_sample_size:
                sample_idx = np.random.permutation(len(sampled_states))[:max_sample_size]
                sampled_states = [sampled_states[i] for i in sample_idx]
                sampled_metrics = [sampled_metrics[i] for i in sample_idx]

        return self.alpha_pool

    def calculate_expression(self, expression, feature_data=None, rank=False) -> torch.Tensor:
        if feature_data is None:
            feature_data = self.feature_data

        local_env = {}
        for op_name, op_func in self.unary_ops.items():
            local_env[op_name] = op_func
        for op_name, op_func in self.binary_ops.items():
            local_env[op_name] = op_func
        for feature_name in self.feature_names:
            local_env[feature_name] = feature_data[feature_name]
        
        result = eval(expression, {"__builtins__": {}}, local_env)
        return ops_rank(result) if rank else result