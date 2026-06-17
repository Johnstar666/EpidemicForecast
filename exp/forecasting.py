from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from layers.sstkg_constraint import SSTKGConstraintLayer
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

class RMSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(RMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, x, y):
        mse = torch.mean((x - y) ** 2, dim=-1)
        if self.reduction == 'mean':
            return torch.sqrt(torch.mean(mse))
        elif self.reduction == 'sum':
            return torch.sqrt(torch.sum(mse))
        else:
            return torch.sqrt(mse)

class Exp_Forecasting(Exp_Basic):
    def __init__(self, args):
        super(Exp_Forecasting, self).__init__(args)
        
    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args)
        if self.args.use_multi_gpu:
            self.device = torch.device('cuda:{}'.format(self.args.local_rank))
            model = DDP(model.cuda(), device_ids=[self.args.local_rank],find_unused_parameters=True)
        else:
            self.device = torch.device("cuda")
        return model

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = RMSELoss()
        return criterion

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def train(self, setting):
        # 获取数据加载器
        train_data, train_loader = data_provider(self.args, 'train')
        vali_data, vali_loader = data_provider(self.args, 'val')
        test_data, test_loader = data_provider(self.args, 'test')
        weight_decey_count = 0 

        path = os.path.join(self.args.checkpoints, setting)
        if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
            if not os.path.exists(path):
                os.makedirs(path)
        
        # 创建训练可视化目录
        train_viz_path = os.path.join('./train_results', setting)
        if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
            if not os.path.exists(train_viz_path):
                os.makedirs(train_viz_path)
        
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

        # 初始化loss记录列表
        train_losses = []
        vali_losses = []
        test_losses = []
        learning_rates = []

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
            
            # 记录当前epoch的loss和学习率
            if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                train_losses.append(train_loss)
                vali_losses.append(vali_loss)
                test_losses.append(test_loss)
                learning_rates.append(model_optim.param_groups[0]['lr'])
            
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
        
        # 训练结束后生成loss可视化图表
        if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
            self._plot_training_curves(train_losses, vali_losses, test_losses, learning_rates, train_viz_path)
            self._save_training_log(train_losses, vali_losses, test_losses, learning_rates, train_viz_path)
        
        return self.model

    def vali(self, vali_data, vali_loader, criterion, is_test=False):
        total_loss = []
        total_count = []
        self.model.eval()
        test_steps = len(vali_loader)
        time_now = time.time()
        iter_count = 0
        
        with torch.no_grad():
            for i, (batch_x, _, adj_x, adj_y,batch_y) in enumerate(vali_loader):
                iter_count += 1
                
                batch_x = batch_x.float().to(self.device)
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

    def _plot_training_curves(self, train_losses, vali_losses, test_losses, learning_rates, save_path):
        """绘制训练曲线"""
        epochs = list(range(1, len(train_losses) + 1))
        
        # 绘制loss曲线
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
        plt.plot(epochs, vali_losses, 'r-', label='Validation Loss', linewidth=2)
        plt.plot(epochs, test_losses, 'g-', label='Test Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training, Validation and Test Loss Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 绘制学习率曲线
        plt.subplot(2, 1, 2)
        plt.plot(epochs, learning_rates, 'm-', label='Learning Rate', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 单独保存loss曲线（更清晰）
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
        plt.plot(epochs, vali_losses, 'r-', label='Validation Loss', linewidth=2)
        plt.plot(epochs, test_losses, 'g-', label='Test Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_path, 'loss_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _save_training_log(self, train_losses, vali_losses, test_losses, learning_rates, save_path):
        """保存训练日志到CSV文件"""
        import pandas as pd
        
        epochs = list(range(1, len(train_losses) + 1))
        log_data = {
            'epoch': epochs,
            'train_loss': train_losses,
            'vali_loss': vali_losses,
            'test_loss': test_losses,
            'learning_rate': learning_rates
        }
        
        df = pd.DataFrame(log_data)
        df.to_csv(os.path.join(save_path, 'training_log.csv'), index=False)
        
        # 同时保存为文本文件便于查看
        with open(os.path.join(save_path, 'training_summary.txt'), 'w') as f:
            f.write(f"Training completed with {len(train_losses)} epochs\n")
            f.write(f"Final Train Loss: {train_losses[-1]:.6f}\n")
            f.write(f"Final Validation Loss: {vali_losses[-1]:.6f}\n")
            f.write(f"Final Test Loss: {test_losses[-1]:.6f}\n")
            f.write(f"Best Validation Loss: {min(vali_losses):.6f} (Epoch {vali_losses.index(min(vali_losses)) + 1})\n")

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
                
                # Extract influence layer from model if available
                influence_layer = None
                if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'influence_layer'):
                    influence_layer = self.model.encoder.influence_layer
                
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
                    
                    # Apply SSTKG-based error correction for multi-step predictions (from step 2 onwards)
                    if influence_layer is not None and j > 0 and hasattr(self.args, 'use_sstkg_constraint') and self.args.use_sstkg_constraint:
                        with torch.no_grad():
                            # Simple heuristic-based correction without complex influence computation
                            # Use the fact that predictions should be smooth and consistent with recent trends
                            if len(pred_y) >= 2:
                                prev_pred = pred_y[-2]  # [batch, token_len, N]
                                curr_pred = pred_y[-1]  # [batch, token_len, N]
                                
                                # Compute trend from previous two predictions
                                trend = curr_pred - prev_pred
                                
                                # Project trend forward with damping
                                projected_next = curr_pred + 0.8 * trend
                                
                                # Blend original prediction with trend projection
                                outputs_corrected = 0.6 * outputs + 0.4 * torch.cat([
                                    outputs[:, :-self.args.token_len, :], 
                                    projected_next
                                ], dim=1)
                                
                                outputs = outputs_corrected
                    
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
                    # Handle multi-node data by visualizing the first node (node 0)
                    node_idx = 0
                    if len(true.shape) == 3:  # Multi-node case: (batch, pred_len, num_nodes)
                        gt = np.array(true[0, :, node_idx])
                        pd = np.array(pred[0, :, node_idx])
                        lookback = batch_x[0, :, node_idx, -1].detach().cpu().numpy()
                    else:  # Single-node case: (batch, pred_len)
                        gt = np.array(true[0, :, -1])
                        pd = np.array(pred[0, :, -1])
                        lookback = batch_x[0, :, -1].detach().cpu().numpy()
                    gt = np.concatenate([lookback, gt], axis=0)
                    pd = np.concatenate([lookback, pd], axis=0)
                    dir_path = folder_path + f'{self.args.test_pred_len}/'
                    if not os.path.exists(dir_path):
                        os.makedirs(dir_path)
                    visual(gt, pd, os.path.join(dir_path, f'{i}.png'))
                    
                    # Note: SSTKG propagation graph generation requires model modifications 
                    # to expose influence matrices during inference. For now, focus on 
                    # multi-step prediction error correction during the prediction loop.
        
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

# Alias for compatibility with run.py
Forecast = Exp_Forecasting
