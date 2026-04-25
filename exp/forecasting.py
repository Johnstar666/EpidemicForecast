from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn


class RMSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)
        self.reduction = reduction
        
    def forward(self, y_pred, y_true):
        mse = self.mse(y_pred, y_true)
        rmse = torch.sqrt(mse + 1e-6)  
        return rmse



class Forecast(Exp_Basic):
    def __init__(self, args):
        super(Forecast, self).__init__(args)
        
    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args)
        if self.args.use_multi_gpu:
            self.device = torch.device('cuda:{}'.format(self.args.local_rank))
            model = DDP(model.cuda(), device_ids=[self.args.local_rank],find_unused_parameters=True)
        else:
            self.device = torch.device("cuda")
        return model


    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader
    def _select_optimizer(self):
        weight_params = []
        bias_params = []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if 'bias' in n:
                bias_params.append(p)
            else:
                weight_params.append(p)
        
        optimizer = optim.Adam([
            {
                'params': weight_params,
                'weight_decay': self.args.weight_decay,
                'lr': self.args.learning_rate
            },
            {
                'params': bias_params,
                'weight_decay': 0, 
                'lr': self.args.learning_rate * 2  
            }
        ], betas=(0.9, 0.99)) 
        
        if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
            print(f'Optimizer configured with {len(weight_params)} weight params')
            print(f'and {len(bias_params)} bias params')
        if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
            print('next learning rate is {}'.format(self.args.learning_rate))
        
        return optimizer

    def _select_criterion(self):
        criterion = RMSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion, is_test=False):
        total_loss = []
        total_count = []
        time_now = time.time()
        test_steps = len(vali_loader)
        iter_count = 0
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, _, adj_x, adj_y,batch_y) in enumerate(vali_loader):
                iter_count += 1
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                adj_x = adj_x.float().to(self.device)
                adj_y = adj_y.float().to(self.device)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs,adj_pred = self.model(batch_x,adj_x)
                        outputs,adj_pred = outputs.to('cuda'), adj_pred.to('cuda')
                else:
                        outputs,adj_pred = self.model(batch_x,adj_x)
                        outputs,adj_pred = outputs.to('cuda'), adj_pred.to('cuda')
                if is_test:
                    outputs = outputs[:, -self.args.token_len:, :]
                    adj_pred = adj_pred[:, -self.args.token_len:, :]
                    adj_y = adj_y[:,-self.args.token_len:, :]
                    batch_y = batch_y[:, -self.args.token_len:, :].to(self.device)
                else:
                    outputs = outputs[:, :, :]
                    adj_pred = adj_pred[:, :, :]
                    adj_y = adj_y[:, :, :]
                    batch_y = batch_y[:, :, :].to(self.device)

                loss = criterion(outputs, batch_y) 
                loss += criterion(adj_pred, adj_y)/3000
                rmse_l = criterion(outputs, batch_y)
                rmse_l = rmse_l.detach().cpu()
                total_loss.append(rmse_l)
                total_count.append(batch_x.shape[0])
                if (i + 1) % 100 == 0:
                    if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * (test_steps - i)    
                        print("\titers: {}, speed: {:.4f}s/iter, left time: {:.4f}s".format(i + 1, speed, left_time))
                        iter_count = 0
                        time_now = time.time()
        if self.args.use_multi_gpu:
            total_loss = torch.tensor(np.average(total_loss, weights=total_count)).to(self.device)
            dist.barrier()   
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            total_loss = total_loss.item() / dist.get_world_size()
        else:
            total_loss = np.average(total_loss, weights=total_count)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        weight_decey_count = 0 

        path = os.path.join(self.args.checkpoints, setting)
        if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
            if not os.path.exists(path):
                os.makedirs(path)
        
        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(self.args, verbose=True)
        trainable_params = sum(p.numel() for n, p in self.model.named_parameters() if p.requires_grad)
        all_params = sum(p.numel() for n,p in self.model.named_parameters())
        model_optim = self._select_optimizer()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=self.args.tmax, eta_min=1e-8)
        criterion = self._select_criterion()
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0

            loss_val = torch.tensor(0., device="cuda")
            count = torch.tensor(0., device="cuda")
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, _, adj_x, adj_y,batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                adj_x = adj_x.float().to(self.device)
                adj_y = adj_y.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs,adj_pred = self.model(batch_x, adj_x)
                        outputs,adj_pred = outputs.to('cuda'), adj_pred.to('cuda')
                        loss = criterion(outputs, batch_y)+criterion(adj_pred, adj_y) /3000          
                        loss_val += criterion(outputs, batch_y)
                        count += 1
                else:
                    # print(batch_y.shape, batch_y.shape)
                    outputs,adj_pred = self.model(batch_x,adj_x)
                    outputs,adj_pred = outputs.to('cuda'), adj_pred.to('cuda')
                    loss = criterion(outputs, batch_y) + criterion(adj_pred, adj_y)/3000
                    loss_val += criterion(outputs, batch_y) + criterion(adj_pred, adj_y) /3000
                    count += 1
                
                if (i + 1) % 100 == 0:
                    if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
            if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))   
            if self.args.use_multi_gpu:
                dist.barrier()   
                dist.all_reduce(loss_val, op=dist.ReduceOp.SUM)
                dist.all_reduce(count, op=dist.ReduceOp.SUM)      
            train_loss = loss_val.item() / count.item()

            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion, is_test=True)
            if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                print("Epoch: {}, Steps: {} | Train Loss: {:.7f} Vali Loss: {:.7f} Test Loss: {:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                    print("Early stopping")
                break
            if self.args.cosine:
                scheduler.step()
                if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                    print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            if early_stopping.weight_decay:
                weight_decey_count += 1
                adjust_learning_rate(model_optim, weight_decey_count + 1, self.args)
                early_stopping.weight_decay = False
            if self.args.use_multi_gpu:
                train_loader.sampler.set_epoch(epoch + 1)
                
        best_model_path = path + '/' + 'checkpoint.pth'
        if self.args.use_multi_gpu:
            dist.barrier()
            self.model.load_state_dict(torch.load(best_model_path), strict=False)
        else:
            self.model.load_state_dict(torch.load(best_model_path), strict=False)
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        print("info:", self.args.test_seq_len, self.args.test_label_len, self.args.token_len, self.args.test_pred_len) # 18, 15, 3, 3
        if test:
            print('loading model')
            setting = self.args.test_dir
            best_model_path = self.args.test_file_name

            print("loading model from {}".format(os.path.join(self.args.checkpoints, setting, best_model_path)))
            load_item = torch.load(os.path.join(self.args.checkpoints, setting, best_model_path))
            self.model.load_state_dict({k.replace('module.', ''): v for k, v in load_item.items()}, strict=False)

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        time_now = time.time()
        test_steps = len(test_loader)
        iter_count = 0
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, _, adj_x, adj_y,batch_y) in enumerate(test_loader):
                iter_count += 1
                batch_x = batch_x.float().to(self.device)
                adj_x = adj_x.float().to(self.device)
                adj_y = adj_y.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                

                inference_steps = self.args.test_pred_len // self.args.token_len # 3 // 3 = 1
                dis = self.args.test_pred_len - inference_steps * self.args.token_len # 3 - 1 * 3 = 0
                if dis != 0:
                    inference_steps += 1
                pred_y = []
                adj_y = []
                for j in range(inference_steps):
                    if len(pred_y) != 0:
                        y_new = pred_y[-1].unsqueeze(-1).repeat(1, 1, 1, batch_x.shape[3])
                        adj_new = adj_y[-1]
                        batch_x = torch.cat([batch_x[:, self.args.token_len:, :,:], y_new], dim=1)
                        adj_x = torch.cat([adj_x[:, self.args.token_len:, :,:], adj_new], dim=1)
                        
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs,adj_pred = self.model(batch_x, adj_x)
                            outputs,adj_pred = outputs.to('cuda'), adj_pred.to('cuda')
                    else:
                        outputs,adj_pred = self.model(batch_x, adj_x)
                        outputs,adj_pred = outputs.to('cuda'), adj_pred.to('cuda')
                    pred_y.append(outputs[:, -self.args.token_len:, :])
                    adj_y.append(adj_pred[:, -self.args.token_len:, :])
                pred_y = torch.cat(pred_y, dim=1)
                adj_y = torch.cat(adj_y, dim=1)
                if dis != 0:
                    pred_y = pred_y[:, :-(self.args.token_len - dis), :]
                    adj_y = adj_y[:, :-(self.args.token_len - dis), :]
                batch_y = batch_y[:, -self.args.test_pred_len:, :].to(self.device)
                outputs = pred_y.detach().cpu()
                batch_y = batch_y.detach().cpu()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if (i + 1) % 100 == 0:
                    if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * (test_steps - i)
                        print("\titers: {}, speed: {:.4f}s/iter, left time: {:.4f}s".format(i + 1, speed, left_time))
                        iter_count = 0
                        time_now = time.time()
                    
                if self.args.visualize and i == 0:
                    gt = np.array(true[0, :, -1])
                    pd = np.array(pred[0, :, -1])
                    lookback = batch_x[0, :, -1].detach().cpu().numpy()
                    gt = np.concatenate([lookback, gt], axis=0)
                    pd = np.concatenate([lookback, pd], axis=0)
                    dir_path = folder_path + f'{self.args.test_pred_len}/'
                    if not os.path.exists(dir_path):
                        os.makedirs(dir_path)
                    visual(gt, pd, os.path.join(dir_path, f'{i}.png'))
        
        preds = torch.cat(preds, dim=0).numpy()
        trues = torch.cat(trues, dim=0).numpy()
        
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('rmse:{}, mae:{}'.format(rmse, mae))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('rmse:{}, mae:{}'.format(rmse, mae))
        f.write('\n')
        f.write('\n')
        f.close()
        return
