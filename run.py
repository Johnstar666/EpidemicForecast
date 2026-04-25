import argparse
import os
import random
import numpy as np
import torch
import torch.distributed as dist
from exp.forecasting import Forecast
if __name__ == '__main__':
    fix_seed = 2026
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='AutoTimes')

    # basic config
    parser.add_argument('--task_name', type=str, default='epi_forcasting',
                        help='task name')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='Italy_7d', help='model id')
    parser.add_argument('--model', type=str, default='EpiLLm_Gpt2',
                        help='model name, options: [EpiLLm_Gpt2,EpiLLm_DS3,EpiLLm_gemma1]')

    # data loader
    parser.add_argument('--data', type=str,  default='epidataset', help='dataset type')
    parser.add_argument('--root_path', type=str, default='/root/experiment/EpiLLM-backup/dataset/Italy', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='italy_labels.csv', help='data file')
    parser.add_argument('--node_num', type=int, default=105, help='data node number') # france 81, italy 105, spain 34, england 129
    parser.add_argument('--test_data_path', type=str, default='italy_labels.csv', help='test data file used in zero shot forecasting')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--drop_last',  action='store_true', default=False, help='drop last batch in data loader')
    parser.add_argument('--val_set_shuffle', action='store_false', default=True, help='shuffle validation set')
    parser.add_argument('--drop_short', action='store_true', default=False, help='drop too short sequences in dataset')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=21, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=14, help='label length')
    parser.add_argument('--token_len', type=int, default=7, help='token length')
    parser.add_argument('--test_seq_len', type=int, default=21, help='test seq len')
    parser.add_argument('--test_label_len', type=int, default=14, help='test label len')
    parser.add_argument('--test_pred_len', type=int, default=7, help='test pred len')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

    # model define
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--llm_ckp_dir', type=str, default='./gpt', help='llm checkpoints dir')
    parser.add_argument('--mlp_hidden_dim', type=int, default=256, help='mlp hidden dim')
    parser.add_argument('--mlp_hidden_layers', type=int, default=2, help='mlp hidden layers')
    parser.add_argument('--mlp_activation', type=str, default='tanh', help='mlp activation')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=7, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='optimizer learning rate') # italy 0.005 spain england france 0.001
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--cosine', action='store_true', help='use cosine annealing lr', default=False)
    parser.add_argument('--tmax', type=int, default=10, help='tmax in cosine anealing lr')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--mix_embeds', action='store_true', help='mix embeds', default=False)
    parser.add_argument('--test_dir', type=str, default='./test', help='test dir')
    parser.add_argument('--test_file_name', type=str, default='checkpoint.pth', help='test file')
    
    # GPU
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--visualize', action='store_true', help='visualize', default=False)
    args = parser.parse_args()

    if args.use_multi_gpu:
        ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = os.environ.get("MASTER_PORT", "35684")
        rank = int(os.environ.get("RANK", "0")) 
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        gpus = torch.cuda.device_count()
        world_size = gpus 
        args.local_rank = local_rank
        print(ip, port, world_size, rank, local_rank, gpus)
        dist.init_process_group(backend="nccl", init_method=f"tcp://{ip}:{port}", world_size=world_size,
                                rank=rank)
        torch.cuda.set_device(local_rank)

    Exp = Forecast


    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = '{}_{}_{}_{}_sl{}_ll{}_tl{}_lr{}_bt{}_wd{}_hd{}_hl{}_cos{}_mix{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.seq_len,
                args.label_len,
                args.token_len,
                args.learning_rate,
                args.batch_size,
                args.weight_decay,
                args.mlp_hidden_dim,
                args.mlp_hidden_layers,
                args.cosine,
                args.mix_embeds,
                args.des, ii)
            if (args.use_multi_gpu and args.local_rank == 0) or not args.use_multi_gpu:
                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)
            if (args.use_multi_gpu and args.local_rank == 0) or not args.use_multi_gpu:
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_sl{}_ll{}_tl{}_lr{}_bt{}_wd{}_hd{}_hl{}_cos{}_mix{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.seq_len,
            args.label_len,
            args.token_len,
            args.learning_rate,
            args.batch_size,
            args.weight_decay,
            args.mlp_hidden_dim,
            args.mlp_hidden_layers,
            args.cosine,
            args.mix_embeds,
            args.des, ii)
        exp = Exp(args)  # set experiments
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
