import torch
from torch.utils.data import DataLoader
import timm
from datasets.dataset import NPY_datasets
from tensorboardX import SummaryWriter
from models.bemunet.bemunet import BEMUNet

from engine import *
import os
import sys
from dotenv import load_dotenv
load_dotenv()

from utils import *
from configs.config_setting import setting_config

import warnings
import requests
warnings.filterwarnings("ignore")
def send_telegram_message(token, chat_id, message):
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = {"chat_id": chat_id, "text": message}
        requests.post(url, data=data, timeout=5)
    except Exception as e:
        print(f"[Telegram] Failed: {e}")

def main(config):
    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)
    global writer
    writer = SummaryWriter(config.work_dir + 'summary')

    log_config_info(config, logger)

    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()

    print('#----------Preparing dataset----------#')
    train_dataset = NPY_datasets(config.data_path, config, train=True)
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=config.num_workers)
    val_dataset = NPY_datasets(config.data_path, config, train=False)
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=config.num_workers,
                            drop_last=True)

    print('#----------Prepareing Model----------#')
    model_cfg = config.model_config
    if config.network == 'bemunet':
        model = BEMUNet(
            num_classes=model_cfg['num_classes'],
            input_channels=model_cfg['input_channels'],
            depths=model_cfg['depths'],
            depths_decoder=model_cfg['depths_decoder'],
            drop_path_rate=model_cfg['drop_path_rate'],
            load_ckpt_path=model_cfg['load_ckpt_path'],
        )
        model.load_from()

    else:
        raise Exception('network in not right!')
    model = model.cuda()

    cal_params_flops(model, 256, logger)

    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)

    print('#----------Set other params----------#')
    start_epoch = 1
    best_miou = -1
    best_epoch = 1


    if config.only_test_and_save_figs:
        checkpoint = torch.load(config.best_ckpt_path, map_location=torch.device('cpu'))
        remapped = {k.replace('vmunet.', 'bemunet.'): v for k, v in checkpoint.items()}
        model.module.load_state_dict(remapped, strict=False)   
        config.work_dir = config.img_save_path
        if not os.path.exists(config.work_dir + 'outputs/'):
            os.makedirs(config.work_dir + 'outputs/')
        loss = test_one_epoch(
            val_loader,
            model,
            criterion,
            logger,
            config,
        )
        return

    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        best_miou = checkpoint['best_miou']
        best_epoch = checkpoint['best_epoch']

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, best_miou: {best_miou:.6f}, best_epoch: {best_epoch}'
        logger.info(log_info)

    step = 0
    print('#----------Training----------#')
    for epoch in range(start_epoch, config.epochs + 1):

        torch.cuda.empty_cache()

        step = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            step,
            logger,
            config,
            writer
        )

        miou = val_one_epoch(
            val_loader,
            model,
            criterion,
            epoch,
            logger,
            config
        )

        if miou > best_miou:
            msg = f"✅ New best model at epoch {epoch}: mIoU improved from {best_miou:.6f} → {miou:.6f}"
            print(msg)
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            best_miou = miou
            best_epoch = epoch
            tg_token   = os.environ.get('TELEGRAM_TOKEN')
            tg_chat_id = os.environ.get('TELEGRAM_CHAT_ID')
            if tg_token and tg_chat_id:
                send_telegram_message(tg_token, tg_chat_id, msg)


        torch.save(
    {
        'epoch': epoch,
        'best_miou': best_miou,
        'best_epoch': best_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, os.path.join(checkpoint_dir, 'latest.pth'))


    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(config.work_dir + 'checkpoints/best.pth', map_location=torch.device('cpu'))
        model.load_state_dict(best_weight)
        test_one_epoch(
            val_loader,
            model,
            criterion,
            logger,
            config,
        )
        os.rename(
            os.path.join(checkpoint_dir, 'best.pth'),
            os.path.join(checkpoint_dir, f'best-epoch{best_epoch}-miou{best_miou:.4f}.pth')
        )

    return best_miou, best_epoch


if __name__ == '__main__':
    import csv

    LR_LIST = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]

    RESULTS_DIR = 'results/hparam_search_isic/'
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, 'results.csv')

    tg_token   = os.environ.get('TELEGRAM_TOKEN')
    tg_chat_id = os.environ.get('TELEGRAM_CHAT_ID')
    enable_tg  = bool(tg_token and tg_chat_id)

    if enable_tg:
        send_telegram_message(tg_token, tg_chat_id,
            f"LR Search ISIC started\nRuns: {len(LR_LIST)} | LRs: {LR_LIST}")

    results = []
    best_miou_overall = -1
    best_lr = None

    for i, lr in enumerate(LR_LIST):
        print(f'\n{"="*50}')
        print(f'Run {i+1}/{len(LR_LIST)}  |  lr={lr}')
        print('='*50)

        class Config(setting_config):
            pass
        Config.lr = lr
        Config.work_dir = os.path.join(RESULTS_DIR, f'run{i+1:02d}_lr{lr}') + '/'

        try:
            best_miou, best_epoch = main(Config)
            result = {'run': i+1, 'lr': lr, 'best_miou': round(best_miou, 4), 'best_epoch': best_epoch}
            print(f'  => best_miou={best_miou:.4f}  best_epoch={best_epoch}')
            if enable_tg:
                send_telegram_message(tg_token, tg_chat_id,
                    f"Run {i+1}/{len(LR_LIST)} done\nlr={lr} | mIoU={best_miou:.4f} | epoch={best_epoch}")
            if best_miou > best_miou_overall:
                best_miou_overall = best_miou
                best_lr = lr
        except Exception as e:
            print(f'  => FAILED: {e}')
            result = {'run': i+1, 'lr': lr, 'best_miou': None, 'best_epoch': None}
            if enable_tg:
                send_telegram_message(tg_token, tg_chat_id, f"Run {i+1} FAILED: lr={lr}\n{e}")

        results.append(result)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['run', 'lr', 'best_miou', 'best_epoch'])
            writer.writeheader()
            writer.writerows(results)

    print(f'\n{"="*50}')
    print(f'SEARCH COMPLETE  |  Best lr={best_lr}  =>  mIoU={best_miou_overall:.4f}')
    print(f'Results: {csv_path}')
    print('='*50)

    if enable_tg:
        send_telegram_message(tg_token, tg_chat_id,
            f"LR Search COMPLETE\nBest lr={best_lr} | mIoU={best_miou_overall:.4f}\nCSV: {csv_path}")