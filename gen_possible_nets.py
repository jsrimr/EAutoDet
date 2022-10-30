import argparse
import logging
import math
import os
import random
import time
from copy import deepcopy
from pathlib import Path
from threading import Thread

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from models import darts_cell

import test  # import test.py to get mAP after each epoch
from models.experimental import attempt_load
from models.yolo_search import Model
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader_search, create_dataloader
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer_search, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr
from utils.google_utils import attempt_download
from utils.loss import ComputeLoss
from utils.plots import plot_images, plot_labels, plot_results, plot_evolution
from utils.torch_utils import EMA, ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, is_parallel

from architect import Architect

# for profiling the model
# from models.yolo import parse_model
from models.yolo import Model as Model_yolo
import thop
import json
from utils.torch_utils import time_synchronized


logger = logging.getLogger(__name__)


def gen_graph(hyp, opt, device, tb_writer=None, wandb=None):
    logger.info(colorstr('hyperparameters: ') +
                ', '.join(f'{k}={v}' for k, v in hyp.items()))
    save_dir, epochs, batch_size, total_batch_size, weights, rank = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank

    # Directories
    # wdir = save_dir / 'weights'
    # wdir.mkdir(parents=True, exist_ok=True)  # make dir
    # geno_dir = save_dir / 'genotypes'
    # geno_dir.mkdir(parents=True, exist_ok=True)  # make dir
    # alpha_dir = save_dir / 'alphas'
    # alpha_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # # Save run settings
    # with open(save_dir / 'hyp.yaml', 'w') as f:
    #     yaml.dump(hyp, f, sort_keys=False)
    # with open(save_dir / 'opt.yaml', 'w') as f:
    #     yaml.dump(vars(opt), f, sort_keys=False)

    # Configure
    init_seeds(2 + rank)
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    with torch_distributed_zero_first(rank):
        check_dataset(data_dict)  # check

    nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes
    names = ['item'] if opt.single_cls and len(
        data_dict['names']) != 1 else data_dict['names']  # class names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (
        len(names), nc, opt.data)  # check

    # Model
    model = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get(
        'anchors')).to(device)  # create

    """
    gen maximum
    {len(op_arch_param) + len(ch_arch_param) + len(edge_arch_param)}
    of yaml files
    """
    def gen_yaml(model, op_geno_idx, ch_geno_idx, edge_geno_idx):
        op_geno, ch_geno, edge_geno = [], [], []
        # new yaml
        with open(model.cfg) as f:
            model_yaml = yaml.load(f, Loader=yaml.SafeLoader)  # model dict
        idx_op = 0
        idx_ch = 0
        idx_edge = 0
        gd = model_yaml['depth_multiple']
        for i, tmp in enumerate(model_yaml['backbone'] + model_yaml['head']):
            if isinstance(tmp[3][-1], dict):  # del unused variables
                for key in ['gumbel_channel']:
                    if key in tmp[3][-1].keys():
                        del tmp[3][-1][key]
            if tmp[2] in ['Conv_search', 'Bottleneck_search', 'Conv_search_merge', 'Bottleneck_search_merge', 'SepConv_search_merge']:
                n = tmp[1]
                n = max(round(n * gd), 1) if n > 1 else n  # depth gain
                func_p = tmp[3]
                Cout = func_p[0]
                tmp[2] = tmp[2].split('_')[0]  # set name
                # set kernel-size and dilation-ratio and channel
                k = []
                d = []
                e = []
                for j in range(n):
                    k.append(func_p[1][op_geno_idx[idx_op+j]][0])
                    d.append(func_p[1][op_geno_idx[idx_op+j]][1])
                    if ch_geno_idx[idx_ch+j] is None:
                        e.append(1.0)  # original YOLOv5 uses e=1.0
                    else:
                        e.append(func_p[2][ch_geno_idx[idx_ch+j]])
                op_geno.append(list(zip(k, d)))
                ch_geno.append(e)
                if n == 1:
                    k = k[0]
                    d = d[0]
                    e = e[0]
                tmp[3][1] = k
    #               tmp[3].insert(2, d)
                # originally, tmp[3][2] is candidate_e, which is useless for full-train
                tmp[3][2] = d
                if tmp[2] in ['Bottleneck']:
                    if isinstance(tmp[3][-1], dict):
                        tmp[3][-1]['e_bottleneck'] = e
                    else:
                        tmp[3].append({'e_bottleneck': e})
                else:
                    tmp[3][0] = Cout * e
                idx_op += n
                idx_ch += n
            elif tmp[2] in ['C3_search', 'C3_search_merge']:
                n = tmp[1]
                n = max(round(n * gd), 1) if n > 1 else n  # depth gain
                func_p = tmp[3]
                Cout = func_p[0]
                candidate_e = func_p[2]
                tmp[2] = tmp[2].split('_')[0]  # set name
                # set kernel-size and dilation-ratio and channel
                k = []
                d = []
                e = []
                for j in range(n):
                    k.append(func_p[1][op_geno_idx[idx_op+j]][0])
                    d.append(func_p[1][op_geno_idx[idx_op+j]][1])
                    if ch_geno_idx[idx_ch+j] is None:
                        e.append(1.0)  # original YOLOv5 uses e=1.0
                    else:
                        e.append(candidate_e[ch_geno_idx[idx_ch+j]])
                op_geno.append(list(zip(k, d)))
                ch_geno.append(deepcopy(e))
                if n == 1:
                    k = k[0]
                    d = d[0]
                    e = e[0]
                tmp[3][1] = k
    #               tmp[3].insert(2, d)
                tmp[3][2] = d
                if isinstance(tmp[3][-1], dict):
                    tmp[3][-1]['e_bottleneck'] = e
                else:
                    tmp[3].append({'e_bottleneck': e})
                # for c2
                if isinstance(func_p[-1], dict) and func_p[-1].get('search_c2', False):
                    if isinstance(func_p[-1]['search_c2'], list):
                        tmp[3][0] = Cout * \
                            func_p[-1]['search_c2'][ch_geno_idx[idx_ch+n]]
                        ch_geno[-1].append(func_p[-1]['search_c2']
                                           [ch_geno_idx[idx_ch+n]])
                    else:
                        tmp[3][0] = Cout * candidate_e[ch_geno_idx[idx_ch+n]]
                        ch_geno[-1].append(candidate_e[ch_geno_idx[idx_ch+n]])
                    del tmp[3][-1]['search_c2']
                    idx_ch += n+1
                else:
                    idx_ch += n
                idx_op += n
            elif tmp[2] == 'AFF':
                tmp[2] = 'FF'
                all_edges = tmp[0]
                Cout, all_strides, all_kds = tmp[3][0:3]
                if isinstance(tmp[3][-1], dict):
                    candidate_e = tmp[3][-1].get('candidate_e', None)
                    separable = tmp[3][-1].get('separable', False)
                else:
                    candidate_e = None
                    separable = False
                edges = []
                ks = []
                ds = []
                strides = []
                for j, idx in enumerate(edge_geno_idx[idx_edge]):
                    edges.append(all_edges[idx])
                    strides.append(all_strides[idx])
                    ks.append(all_kds[op_geno_idx[idx_op+idx]][0])
                    ds.append(all_kds[op_geno_idx[idx_op+idx]][1])
                edge_geno.append(edges)
                op_geno.append(list(zip(ks, ds)))
                ch_geno.append([1.0 for _ in range(len(edges))])
            # for Cout
                if ch_geno_idx[idx_ch+len(all_edges)] is not None:
                    Cout = Cout * \
                        candidate_e[ch_geno_idx[idx_ch+len(all_edges)]]
                    ch_geno[-1].append(candidate_e[ch_geno_idx[idx_ch+len(all_edges)]])
                args_dict = {'separable': separable}
                tmp[3] = [Cout, strides, ks, ds, args_dict]
                tmp[0] = edges
                idx_op += len(all_edges)
                idx_ch += len(all_edges)+1
                idx_edge += 1
            elif tmp[2] == 'SPP_search':
                tmp[2] = tmp[2].split('_')[0]  # set name
            elif tmp[2] in ['Cells_search', 'Cells_search_merge']:
                tmp[2] = tmp[2].split('_')[0]  # set name
                steps, multiplier, C, reduction, reduction_prev = tmp[3][0:5]
                op_alpha = op_geno_idx[idx_op]
                genotype, concat = darts_cell.genotype(
                    op_alpha, steps, multiplier, num_input=len(tmp[0]))
                tmp[3] = [genotype, concat, C, reduction, reduction_prev]
                op_geno.append([genotype, concat])
                ch_geno.append([1])
                idx_op += 1
                idx_ch += 1
        assert (idx_ch == len(ch_geno_idx))
        assert (idx_op == len(op_geno_idx))
        assert (idx_edge == len(edge_geno_idx))
        # split the alpha_op and alpha_channal
        geno = [op_geno, ch_geno, edge_geno]
        model_yaml['geno'] = geno
        return geno, model_yaml

    def mutate(model):
        op_geno_idx = [0 for alpha in model.op_arch_parameters]
        ch_geno_idx = [0 if isinstance(
            alpha, torch.Tensor) else None for alpha in model.ch_arch_parameters]
        edge_geno_idx = [[0, 1] for alpha in model.edge_arch_parameters]

        for i in range((len(model.op_arch_parameters) + len(model.ch_arch_parameters) + len(model.edge_arch_parameters))):

            if i < len(model.op_arch_parameters):
                alphas = model.op_arch_parameters[i]
                for j in range(alphas.shape[0]):
                    tmp = op_geno_idx[i]
                    op_geno_idx[i] = j
                    yield gen_yaml(model, op_geno_idx, ch_geno_idx, edge_geno_idx)
                    op_geno_idx[i] = tmp

            elif i < len(model.op_arch_parameters) + len(model.ch_arch_parameters):
                alphas = model.ch_arch_parameters[i -
                                                  len(model.op_arch_parameters)]
                if alphas is None:
                    continue
                else:
                    for j in range(alphas.shape[0]):
                        tmp = ch_geno_idx[i - len(model.op_arch_parameters)]
                        ch_geno_idx[i - len(model.op_arch_parameters)] = j
                        yield gen_yaml(model, op_geno_idx, ch_geno_idx, edge_geno_idx)
                        ch_geno_idx[i - len(model.op_arch_parameters)] = tmp
            else:
                alphas = model.edge_arch_parameters[i - len(
                    model.op_arch_parameters) - len(model.ch_arch_parameters)]
                for j in range(alphas.shape[0]):
                    for k in range(j+1, alphas.shape[0]):
                        tmp = edge_geno_idx[i - len(model.op_arch_parameters) -
                                            len(model.ch_arch_parameters)]
                        edge_geno_idx[i - len(model.op_arch_parameters) -
                                      len(model.ch_arch_parameters)] = [j, k]
                        yield gen_yaml(model, op_geno_idx, ch_geno_idx, edge_geno_idx)
                        edge_geno_idx[i - len(model.op_arch_parameters) -
                                      len(model.ch_arch_parameters)] = tmp

    for geno, model_yaml in mutate(model):  # 뭘 변형시킨건지도 정보를 주자 , name, i, j 
        # print(geno)
        # clone_model, save, layer_info = parse_model(model_yaml, [3])
        # clone_model.to(device)
        # clone_model.eval()

        clone_model = Model_yolo(model_yaml).to(device)  # superkernel 이 아니라 일반적인 yolo 모델
        clone_model.eval()

        # profile_model
        x = torch.randn(1, 3, 640, 640).to(device)
        y, dt = [], []  # outputs and time
        for idx, m in enumerate(clone_model.model):
            # if hasattr(m, 'alphas'):
            # if hasattr(m, 'alphas_ch'):

            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [
                    x if j == -1 else y[j] for j in m.f]  # from earlier layers

            o = thop.profile(m, inputs=(x,), verbose=False)[
                0] / 1E9 * 2 if thop else 0  # FLOPS
            t = time_synchronized()
            for _ in range(10):
                _ = m(x)
            dt.append((time_synchronized() - t) / 10 * 100)
            print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))
            # save profile result as a dict then json
            profile = {'FLOPS': o, 'params': m.np,
                       'time': dt[-1], 'type': str(m)}
            json.dump(profile, open(
                f'latency/{m.get_name(x)}.json', 'w'), indent=4)

            # Estimator[m.name] = dt[-1]

            x = m(x)  # run
            y.append(x if m.i in clone_model.save else None)  # save output

        print('%.1fms total' % sum(dt))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_wandb', action='store_true',
                        help='do not log results to Weights & Biases')

    parser.add_argument('--weights', type=str,
                        default='yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str,
                        default='data/coco128.yaml', help='data.yaml path')
    parser.add_argument(
        '--hyp', type=str, default='data/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16,
                        help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int,
                        default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true',
                        help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True,
                        default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true',
                        help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true',
                        help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true',
                        help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true',
                        help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true',
                        help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true',
                        help='use weighted image selection for training')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true',
                        help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true',
                        help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true',
                        help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true',
                        help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='DDP parameter, do not modify')
    parser.add_argument('--log-imgs', type=int, default=16,
                        help='number of images for W&B logging, max 100')
    parser.add_argument('--log-artifacts', action='store_true',
                        help='log artifacts, i.e. final trained model')
    parser.add_argument('--workers', type=int, default=8,
                        help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train',
                        help='save to project/name')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')

    # For NAS
    parser.add_argument('--arch_learning_rate', type=float,
                        default=3e-4, help='learning rate for arch encoding')
    parser.add_argument('--arch_weight_decay', type=float,
                        default=1e-3, help='weight decay for arch encoding')
    parser.add_argument('--search_warmup', type=int, default=0,
                        help='Epoch to Warmup the operation weights')
    parser.add_argument('--train_portion', type=float, default=0.5,
                        help='portion to split the train set and search set')

    opt = parser.parse_args()
    # '{}-{}'.format(opt.project, time.strftime("%Y%m%d-%H%M%S"))
    opt.project = "possible_nets"
    # print("Experiments dir: %s"%opt.project)
    print("cfg file: %s" % opt.cfg)

    # Set DDP variables
    opt.world_size = int(os.environ['WORLD_SIZE']
                         ) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    set_logging(opt.global_rank)
    if opt.global_rank in [-1, 0]:
        check_git_status()
        check_requirements()

    # Resume
    if opt.resume:  # resume an interrupted run
        # specified or most recent path
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()
        assert os.path.isfile(
            ckpt), 'ERROR: --resume checkpoint does not exist'
        apriori = opt.global_rank, opt.local_rank
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(
                **yaml.load(f, Loader=yaml.SafeLoader))  # replace
        opt.cfg, opt.weights, opt.resume, opt.batch_size, opt.global_rank, opt.local_rank = '', ckpt, True, opt.total_batch_size, *apriori  # reinstate
        logger.info('Resuming training from %s' % ckpt)
    else:
        # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(
            opt.cfg), check_file(opt.hyp)  # check files
        assert len(opt.cfg) or len(
            opt.weights), 'either --cfg or --weights must be specified'
        # extend to 2 sizes (train, test)
        opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))
        opt.name = 'evolve' if opt.evolve else opt.name
        opt.save_dir = increment_path(
            Path(opt.project), exist_ok=opt.exist_ok | opt.evolve)  # increment run

    # DDP mode
    opt.total_batch_size = opt.batch_size
    device = select_device(opt.device, batch_size=opt.batch_size)
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        # distributed backend
        dist.init_process_group(backend='nccl', init_method='env://')
        assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        opt.batch_size = opt.total_batch_size // opt.world_size

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

    # Train
    logger.info(opt)
    try:
        if opt.no_wandb:
            wandb = None
        else:
            import wandb
    except ImportError:
        wandb = None
        prefix = colorstr('wandb: ')
        logger.info(
            f"{prefix}Install Weights & Biases for YOLOv5 logging with 'pip install wandb' (recommended)")

    tb_writer = None  # init loggers
    # if opt.global_rank in [-1, 0]:
    #     logger.info(f'Start Tensorboard with "tensorboard --logdir {opt.project}", view at http://localhost:6006/')
    #     tb_writer = SummaryWriter(opt.save_dir)  # Tensorboard
    gen_graph(hyp, opt, device, tb_writer, wandb)
