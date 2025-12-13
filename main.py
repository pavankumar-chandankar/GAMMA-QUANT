# ------------------------------------------------------------------------
# Main script used to commence Gamma-Quant experiments on inertial data.
# ------------------------------------------------------------------------
# Adaption by: Marius Bock
# E-Mail: marius.bock(at)uni-siegen.de
# ------------------------------------------------------------------------


import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse
import datetime
import json
import os
from pprint import pprint
import sys
import time

import torch
import numpy as np
import neptune

from models.train import run_inertial_network
from utils.torch_utils import fix_random_seed
from utils.os_utils import Logger, load_config
from utils.logging_utils import classification_scores, save_confusion_matrix

def main(args):
    if args.neptune:
        run = neptune.init_run(
            project="", # your credentials
            api_token="", # your credentials
        )
    else:
        run = None

    config = load_config(args.config)
    config['init_rand_seed'] = args.seed
    config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.neptune:
        run_id = run["sys/id"].fetch()
    else:
        run_id = args.run_id
    
    ts = datetime.datetime.fromtimestamp(int(time.time()))
    log_dir = os.path.join('logs', config['name'], str(ts) + '_' + run_id)
    sys.stdout = Logger(os.path.join(log_dir, 'log.txt'))

    # save the current cfg
    with open(os.path.join(log_dir, 'cfg.txt'), 'w') as fid:
        pprint(config, stream=fid)
        fid.flush()
    
    if args.neptune:
        run['config_name'] = args.config
        run['config'].upload(os.path.join(log_dir, 'cfg.txt'))
        run['gamma_quant'] = args.gamma_quant
        if args.gamma_quant is not None:
            run['quant_bits'] = args.quant_bits
            if args.gamma_quant == 'quantize_and_gamma':
                run['gamma_type'] = args.gamma_type
                run['apply_gamma'] = args.apply_gamma
            elif args.gamma_quant == 'quantize_and_calibrate':
                run['quant_parameters'] = args.quant_parameters
    
    rng_generator = fix_random_seed(config['init_rand_seed'], include_cuda=True)    

    all_v_pred = np.array([])
    all_v_gt = np.array([])
    all_v_mAP = np.empty((0, len(config['dataset']['tiou_thresholds'])))
    
    start_split = 17

    for i, anno_split in enumerate(config['anno_json']):
    # i is 0-based → split index = i + 1
        if i + 1 < start_split:
            print('Skipping split {} / {}'.format(i + 1, len(config['anno_json'])))
            continue

        with open(anno_split) as f:
            file = json.load(f)
        anno_file = file['database']
        if config['has_null'] == True:
            config['labels'] = ['null'] + list(file['label_dict'])
        else:
            config['labels'] = list(file['label_dict'])
        config['label_dict'] = dict(zip(config['labels'], list(range(len(config['labels'])))))
        train_sbjs = [x for x in anno_file if anno_file[x]['subset'] == 'Training']
        val_sbjs = [x for x in anno_file if anno_file[x]['subset'] == 'Validation']

        print('Split {} / {}'.format(i + 1, len(config['anno_json'])))
        config['dataset']['json_anno'] = anno_split

        t_losses, v_losses, v_mAP, v_preds, v_gt, net = \
            run_inertial_network(train_sbjs, val_sbjs, config, log_dir, args.ckpt_freq,
                                 args.resume, rng_generator, run,
                                 gamma_quant=args.gamma_quant, quant_bits=args.quant_bits,
                                 gamma_type=args.gamma_type, apply_gamma=args.apply_gamma)
        
        # raw results
        (v_acc, v_prec, v_rec, v_f1) = classification_scores(v_gt, v_preds, len(config['labels']))

        # print to terminal
        block1 = '\nFINAL RESULTS SUBJECT {}'.format(i)
        block2 = 'TRAINING:\tavg. loss {:.2f}'.format(np.nanmean(t_losses))
        block3 = 'VALIDATION:\tavg. loss {:.2f}'.format(np.nanmean(v_losses))
        block4 = ''
        block4  += '\n\t\tAvg. mAP {:>4.2f} (%) '.format(np.nanmean(v_mAP) * 100)
        for tiou, tiou_mAP in zip(config['dataset']['tiou_thresholds'], v_mAP):
            block4 += 'mAP@' + str(tiou) +  ' {:>4.2f} (%) '.format(tiou_mAP*100)
        block5 = ''
        block5  += '\t\tAcc {:>4.2f} (%)'.format(np.nanmean(v_acc) * 100)
        block5  += ' Prec {:>4.2f} (%)'.format(np.nanmean(v_prec) * 100)
        block5  += ' Rec {:>4.2f} (%)'.format(np.nanmean(v_rec) * 100)
        block5  += ' F1 {:>4.2f} (%)\n'.format(np.nanmean(v_f1) * 100)

        print('\n'.join([block1, block2, block3, block4, block5]))
                                
        all_v_mAP = np.append(all_v_mAP, v_mAP[None, :], axis=0)
        all_v_gt = np.append(all_v_gt, v_gt)
        all_v_pred = np.append(all_v_pred, v_preds)

        # save raw confusion matrix
        save_confusion_matrix(v_gt, v_preds, config['labels'], os.path.join(log_dir, 'sbj_' + str(i) + '.png'), 'sbj_' + str(i), normalize='true', neptune_run=run)

    # final raw results across all splits
    (v_acc, v_prec, v_rec, v_f1) = classification_scores(all_v_gt, all_v_pred, len(config['labels']))
    
    # print final results to terminal
    block1 = '\nFINAL AVERAGED RESULTS:'
    block2 = ''
    block2  += '\n\t\tAvg. mAP {:>4.2f} (%) '.format(np.nanmean(all_v_mAP) * 100)
    for tiou, tiou_mAP in zip(config['dataset']['tiou_thresholds'], all_v_mAP.T):
        block2 += 'mAP@' + str(tiou) +  ' {:>4.2f} (%) '.format(np.nanmean(tiou_mAP)*100)
    block2  += '\n\t\tAcc {:>4.2f} (%)'.format(np.nanmean(v_acc) * 100)
    block2  += ' Prec {:>4.2f} (%)'.format(np.nanmean(v_prec) * 100)
    block2  += ' Rec {:>4.2f} (%)'.format(np.nanmean(v_rec) * 100)
    block2  += ' F1 {:>4.2f} (%)'.format(np.nanmean(v_f1) * 100)
    
    print('\n'.join([block1, block2]))

    # save final raw confusion matrix
    save_confusion_matrix(all_v_gt, all_v_pred, config['labels'], os.path.join(log_dir, 'all_raw.png'), 'all', normalize='true', neptune_run=run)

    # submit final values to neptune 
    if run is not None:
        run['final_avg_mAP'] = np.nanmean(all_v_mAP)
        for tiou, tiou_mAP in zip(config['dataset']['tiou_thresholds'], all_v_mAP.T):
            run['final_mAP@' + str(tiou)] = (np.nanmean(tiou_mAP))
        run['final_accuracy'] = np.nanmean(v_acc)
        run['final_precision'] = (np.nanmean(v_prec))
        run['final_recall'] = (np.nanmean(v_rec))
        run['final_f1'] = (np.nanmean(v_f1))

    print("ALL FINISHED")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/deepconvlstm/wear_loso.yaml')
    parser.add_argument('--run_id', default='', type=str)
    parser.add_argument('--neptune', action='store_true', default=False)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--ckpt_freq', default=1, type=int)
    parser.add_argument('--ckpt_dir', default='logs/deepconvlstm/low')
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--gpu', default='cuda:0', type=str)
    parser.add_argument('--gamma_quant', default=None, type=str, choices=['quantize', 'quantize_and_calibrate', 'quantize_and_gamma', None])
    parser.add_argument('--quant_bits', default=4, type=int, help='Number of bits for quantization (default: 4)')
    parser.add_argument('--gamma_type', default='s_shaped', type=str, choices=['s_shaped', 'id'], help='Type of gamma function to use (default: s_shaped)')
    parser.add_argument('--apply_gamma', default='global', type=str, choices=['global', 'local'], help='Apply gamma function globally or locally (default: global)')
    args = parser.parse_args()
    main(args)  

