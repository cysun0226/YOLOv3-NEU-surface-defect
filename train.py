from __future__ import division

from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test import evaluate

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

import warnings
warnings.simplefilter('ignore', category=DeprecationWarning)

import logging
from utils.utils import USE_GIOU
from tqdm import tqdm

from shutil import copyfile

report_map = 0

def run(params, exec_main=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    parser.add_argument("--tb_dir", default="logs", help="tensord record file dir")
    parser.add_argument("--resume", default=0, type=int, help="resume epoch")
    parser.add_argument("--iou", default='iou', help="iou type")
    opt = parser.parse_args()

    logger = Logger(opt.tb_dir)
    set_logging(opt.tb_dir)
    logging.info(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Set IOU
    if opt.iou == 'giou':
        USE_GIOU = True

    # Set HPs
    batch_size = opt.batch_size if exec_main else params.get('batch_size')
    step = params.get('step') if params != {} else None
    gamma = params.get('gamma') if params != {} else None
    learn_rate = params.get('learn_rate') if params != {} else 0.001
    weight_decay = params.get('weight_decay') if params != {} else 0

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)

    # Get dataloader
    dataset = ListDataset(train_path, augment=True, multiscale=opt.multiscale_training)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate, weight_decay=weight_decay)
    if params != {}:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step, gamma)

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    max_map = 0
    save_model = False
    best_model = None

    for epoch in range(opt.resume, opt.resume+opt.epochs):
        model.train()
        start_time = time.time()

        logging.info(f"\n [epoch {epoch}]\n")
        total_loss = 0

        pbar = tqdm(total=len(dataloader))
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            loss, outputs = model(imgs, targets)
            loss.backward()

            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()
                if params != {}:
                    lr_scheduler.step()

            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                # Tensorboard logging
                tensorboard_log = []
                for j, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"{name}_{j+1}", metric)]
                tensorboard_log += [("loss", loss.item())]
                logger.list_of_scalars_summary(tensorboard_log, batches_done)

            if i == len(dataloader)-1:
                log_str += AsciiTable(metric_table).table
                log_str += f"\nTotal loss {loss.item()}"
                logging.info(log_str)

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            # log_str += f"\n---- ETA {time_left}"

            # logging.info(log_str)
            # logging.info(f"batch {batch_i}, total loss = {loss.item()}")
            total_loss += loss.item()

            model.seen += imgs.size(0)
            pbar.update(1)
        pbar.close()

        if epoch % opt.evaluation_interval == 0:
            logging.info("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=8,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            logging.info("\n"+AsciiTable(ap_table).table)
            logging.info(f"---- mAP {AP.mean()}, avg_loss {total_loss/len(dataloader)}")

            if AP.mean() >= max_map:
                max_map = AP.mean()
                report_map = max_map
                save_model = True

        if epoch % opt.checkpoint_interval == 0 and save_model == True:
            logging.info(f"mAP = {max_map}, save model epoch = {epoch}")
            torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_%d.pth" % epoch)
            best_model = epoch
            save_model = False
            
    # save the best model to the tb dir
    copyfile(f"checkpoints/yolov3_ckpt_{best_model}.pth", f"{opt.tb_dir}/yolov3_ckpt_{best_model}.pth")


if __name__ == "__main__":
    param = {}
    run(param, exec_main=True)
