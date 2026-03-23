import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

# --- IMPORTS TÙY CHỈNH ---
from datasets.dataset_acdc import RandomGenerator, BaseDataSets
# Lưu ý: Nếu bạn chưa có engine riêng cho acdc, ta dùng tạm engine_synapse 
# nhưng cẩn thận logic tính toán bên trong hàm val_one_epoch
from engine_synapse import * 
from models.vmunet.vmunet import VMUNet
from utils import *
# Import file config bạn vừa sửa
from configs.config_setting_acdc import setting_config 

import os
import sys
import requests
import warnings

# Tắt cảnh báo để log sạch hơn
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
    log_config_info(config, logger)

    # --- TELEGRAM SETUP ---
    tg_token = getattr(config, 'telegram_token', None)
    tg_chat_id = getattr(config, 'telegram_chat_id', None)
    enable_tg = getattr(config, 'enable_telegram', False)

    if enable_tg and tg_token and tg_chat_id:
        send_telegram_message(tg_token, tg_chat_id, f"🚀 Start Training ACDC: {config.network}")

    print('#----------GPU init----------#')
    set_seed(config.seed)
    gpu_ids = [0]
    torch.cuda.empty_cache()
    gpus_num = torch.cuda.device_count()

    if config.distributed:
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.manual_seed_all(config.seed)
        config.local_rank = torch.distributed.get_rank()

    print('#----------Preparing dataset ACDC----------#')
    # DATASET TRAIN: Load Slice 2D
    train_dataset = config.datasets(
        base_dir=config.data_path, 
        list_dir=config.list_dir, 
        split="train",
        transform=transforms.Compose([
            RandomGenerator(output_size=[config.input_size_h, config.input_size_w])
        ])
    )
    
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if config.distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size // gpus_num if config.distributed else config.batch_size,
        shuffle=(train_sampler is None),
        pin_memory=True,
        num_workers=config.num_workers,
        sampler=train_sampler
    )

    # DATASET VAL: Load Volume 3D (để tính Dice chuẩn y tế)
    val_dataset = config.datasets(
        base_dir=config.volume_path, 
        split="val", # Hoặc "test_vol" tùy vào cách bạn định nghĩa trong BaseDataSets
        list_dir=config.list_dir
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1, 
        shuffle=False,
        pin_memory=True,
        num_workers=config.num_workers,
        drop_last=False
    )

    print('#----------Preparing Model----------#')
    model_cfg = config.model_config
    # Khởi tạo VM-UNet với input_channels=1 (cho ACDC) và num_classes=4
    model = VMUNet(
        num_classes=model_cfg['num_classes'],
        input_channels=model_cfg['input_channels'],
        depths=model_cfg['depths'],
        depths_decoder=model_cfg['depths_decoder'],
        drop_path_rate=model_cfg['drop_path_rate'],
        load_ckpt_path=model_cfg['load_ckpt_path'],
    )
    
    # Load pretrained weights nếu có (bỏ qua nếu load_ckpt_path đã xử lý)
    if config.pretrained_path:
        model.load_from(config.pretrained_path)

    if config.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
        model = DDP(model, device_ids=[config.local_rank], output_device=config.local_rank)
    else:
        model = model.cuda() 
        # Nếu dùng 1 GPU thì không cần DataParallel cho đơn giản, hoặc dùng:
        # model = torch.nn.DataParallel(model.cuda(), device_ids=gpu_ids)

    print('#----------Preparing Loss, Optimizer----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    scaler = GradScaler()

    # --- KHỞI TẠO BIẾN ---
    max_mean_dice = 0.0
    best_epoch = 0
    start_epoch = 1
    min_loss = 999

    # --- RESUME NẾU CÓ ---
    if os.path.exists(resume_model):
        checkpoint = torch.load(resume_model, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict']) # Thêm .module nếu lỗi key mismatch
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        max_mean_dice = checkpoint.get('max_mean_dice', 0.0)
        best_epoch = checkpoint.get('best_epoch', 0)
        logger.info(f'Resumed from epoch {start_epoch-1}. Best Dice so far: {max_mean_dice:.4f}')

    print('#----------Start Training----------#')
    for epoch in range(start_epoch, config.epochs + 1):
        torch.cuda.empty_cache()
        if config.distributed:
            train_sampler.set_epoch(epoch)

        # Train loop
        loss = train_one_epoch(
            train_loader, model, criterion, optimizer, scheduler,
            epoch, logger, config, scaler=scaler
        )

        if loss < min_loss:
            min_loss = loss

        # --- VALIDATION (Chỉ tập trung vào DSC) ---
        # ACDC khá nhẹ, có thể validate thường xuyên hơn, ví dụ mỗi 20 epoch
        # Hoặc dùng logic config.val_interval
        if epoch >= config.epochs // 2 and epoch % config.val_interval == 0:
            print(f"Epoch {epoch}: Validating...")
            
            # Gọi hàm val_one_epoch. 
            # Lưu ý: Hàm này thường trả về (dice, hd95). 
            # Ta sẽ hứng cả 2 nhưng chỉ dùng dice.
            save_path_output = os.path.join(outputs, f"epoch_{epoch}")
            if not os.path.exists(save_path_output): os.makedirs(save_path_output)

            val_metrics = val_one_epoch(
                val_dataset, val_loader, model, epoch,
                logger, config, test_save_path=save_path_output,
                val_or_test=False
            )
            
            # Xử lý kết quả trả về (thường là tuple)
            if isinstance(val_metrics, (tuple, list)):
                mean_dice = val_metrics[0]
                # Bỏ qua hd95 (val_metrics[1])
            else:
                mean_dice = val_metrics

            # --- BEST MODEL LOGIC (Chỉ dựa trên DICE) ---
            if mean_dice > max_mean_dice:
                msg = f"🔥 New Best Dice: {mean_dice:.4f} (Was {max_mean_dice:.4f}) at Epoch {epoch}"
                print("\n" + "="*40 + f"\n{msg}\n" + "="*40 + "\n")
                
                if enable_tg and tg_token:
                    send_telegram_message(tg_token, tg_chat_id, msg)
                
                logger.info(msg)
                
                # Lưu model best
                save_dict = model.module.state_dict() if config.distributed else model.state_dict()
                torch.save(save_dict, os.path.join(checkpoint_dir, 'best_model.pth'))
                
                max_mean_dice = mean_dice
                best_epoch = epoch
            else:
                print(f"Epoch {epoch}: Dice {mean_dice:.4f} (Best: {max_mean_dice:.4f})")

        # Lưu Latest Checkpoint
        save_dict = model.module.state_dict() if config.distributed else model.state_dict()
        torch.save({
            'epoch': epoch,
            'max_mean_dice': max_mean_dice,
            'best_epoch': best_epoch,
            'model_state_dict': save_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, resume_model)

if __name__ == '__main__':
    # Load config từ file config_setting.py
    main(setting_config)