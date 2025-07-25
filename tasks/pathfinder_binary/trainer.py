import numpy as np
import os
import torch
import time
from option import Option
import torch.nn as nn
import datetime
import pc_processor
import math
import cv2
import torch.nn.functional as F
import torch.utils.data


class Trainer(object):
    def __init__(self, settings: Option, model: nn.Module, recorder=None):
        # init params
        self.settings = settings
        self.recorder = recorder
        self.model = model.cuda()
        self.remain_time = pc_processor.utils.RemainTime(
            self.settings.n_epochs)

        self.train_loader, self.val_loader, self.train_sampler, self.val_sampler = self._initDataloader()

        self.criterion = self._initCriterion()


        [self.optimizer,
         self.aux_optimizer] = self._initOptimizer()  # self.optimizer是AdamW，self.aux_optimizer是SGD.AdamW

        if self.settings.n_gpus > 1:
            if self.settings.distributed:
                # sync bn
                self.model = pc_processor.layers.sync_bn.replaceBN(
                    self.model).cuda()
                self.model = nn.parallel.DistributedDataParallel(
                    self.model, device_ids=[self.settings.gpu])  # find_unused_parameters=True

            else:
                self.model = nn.DataParallel(self.model)
                # repalce bn with sync_bn
                self.model = pc_processor.layers.sync_bn.replaceBN(
                    self.model).cuda()
                for k, v in self.criterion.items():
                    self.criterion[k] = nn.DataParallel(v).cuda()

        self.metrics = pc_processor.metrics.IOUEval(
            n_classes=self.settings.nclasses, device=torch.device("cpu"),
            ignore=self.ignore_class, is_distributed=self.settings.distributed)
        self.metrics.reset()

        self.metrics_img = pc_processor.metrics.IOUEval(
            n_classes=2, device=torch.device("cpu"),
            ignore=[], is_distributed=self.settings.distributed)
        self.metrics_img.reset()

        self.scheduler = pc_processor.utils.WarmupCosineLR(
            optimizer=self.optimizer,
            lr=0.0005,
            warmup_steps=self.settings.warmup_epochs *
                         len(self.train_loader),
            momentum=self.settings.momentum,
            max_steps=len(self.train_loader) * (
                    self.settings.n_epochs - self.settings.warmup_epochs))

        self.aux_scheduler = pc_processor.utils.WarmupCosineLR(
            optimizer=self.aux_optimizer,
            lr=self.settings.lr,  # 0.0005
            warmup_steps=self.settings.warmup_epochs *
                         len(self.train_loader),
            momentum=self.settings.momentum,
            max_steps=len(self.train_loader) * (self.settings.n_epochs - self.settings.warmup_epochs))

    # ------------------------------------------------------------------
    # functions for initialization
    # ------------------------------------------------------------------
    def _initOptimizer(self):
        # check params
        adam_params = [{"params": self.model.lidar_stream.parameters()}]

        adam_opt = torch.optim.AdamW(
            params=adam_params, lr=0.0005)

        sgd_params = [
            {"params": self.model.camera_stream_encoder.parameters()},
            {"params": self.model.camera_stream_decoder.parameters()}]

        sgd_opt = torch.optim.SGD(
            params=sgd_params, lr=self.settings.lr,
            nesterov=True,
            momentum=self.settings.momentum,
            weight_decay=self.settings.weight_decay)
        optimizer = [adam_opt, sgd_opt]

        return optimizer

    def _initDataloader(self):
        if self.settings.dataset == "Livox":
            data_config_path = "/home/m/Documents/GitHub/PMF_series/PMF_mainWorkSpace/pc_processor/dataset/livox/Livox.yaml"
            trainset = pc_processor.dataset.livox.Livox(
                root=self.settings.data_root,
                sequences=[0, 1, 2, 3, 4, 6, 8, 9],
                # sequences=[0],
                config_path=data_config_path
            )
            self.cls_weight = 1 / (trainset.cls_freq + 1e-8)
            self.ignore_class = []
            for cl, w in enumerate(self.cls_weight):
                print("ignore_class", self.ignore_class)
                if trainset.data_config["learning_ignore"][cl]:
                    self.cls_weight[cl] = 0
                if self.cls_weight[cl] < 1e-10:
                    self.ignore_class.append(cl)
            if self.recorder is not None:
                self.recorder.logger.info("weight: {}".format(self.cls_weight))
            self.mapped_cls_name = trainset.mapped_cls_name

            valset = pc_processor.dataset.livox.Livox(
                root=self.settings.data_root,
                sequences=[5, 7],
                # sequences=[0],
                config_path=data_config_path
            )
        else:
            raise ValueError(
                "invalid dataset: {}".format(self.settings.dataset))

        train_pv_loader = pc_processor.dataset.PerspectiveViewLoader(
            dataset=trainset,
            config=self.settings.config,
            is_train=True, pcd_aug=False, img_aug=True, use_padding=True)

        val_pv_loader = pc_processor.dataset.PerspectiveViewLoader(
            dataset=valset,
            config=self.settings.config,
            is_train=False, use_padding=True)

        if self.settings.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                trainset, shuffle=True, drop_last=True)
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                valset, shuffle=False, drop_last=False)
            train_loader = torch.utils.data.DataLoader(
                train_pv_loader,
                batch_size=self.settings.batch_size[0],
                num_workers=self.settings.n_threads,
                drop_last=True,
                sampler=train_sampler
            )

            val_loader = torch.utils.data.DataLoader(
                val_pv_loader,
                batch_size=self.settings.batch_size[1],
                num_workers=self.settings.n_threads,
                drop_last=False,
                sampler=val_sampler
            )
            return train_loader, val_loader, train_sampler, val_sampler

        else:
            train_loader = torch.utils.data.DataLoader(
                train_pv_loader,
                batch_size=self.settings.batch_size[0],
                num_workers=self.settings.n_threads,
                shuffle=True,
                drop_last=True)

            val_loader = torch.utils.data.DataLoader(
                val_pv_loader,
                batch_size=self.settings.batch_size[1],
                num_workers=self.settings.n_threads,
                shuffle=False,
                drop_last=False
            )
            return train_loader, val_loader, None, None

    def _initCriterion(self):
        criterion = {}
        criterion["lovasz"] = pc_processor.loss.Lovasz_softmax(ignore=0)
        criterion["lovasz_img"] = pc_processor.loss.Lovasz_softmax_img(ignore=None)

        criterion["kl_loss"] = nn.KLDivLoss(reduction="none")

        if self.settings.dataset == "SemanticKitti":
            alpha = self.cls_weight
            alpha = alpha / alpha.max()
        # elif self.settings.dataset == "nuScenes":
        #     alpha = np.ones((self.settings.nclasses))
        elif self.settings.dataset == "Livox":
            alpha = self.cls_weight
            alpha = alpha / alpha.max()
        alpha[0] = 0
        if self.recorder is not None:
            self.recorder.logger.info("focal_loss alpha: {}".format(alpha))
        criterion["focal_loss"] = pc_processor.loss.FocalSoftmaxLoss(
            self.settings.nclasses, gamma=2, alpha=alpha, softmax=False, signal=None)

        # --------------------------------------------------------------------------------------------------

        if self.settings.dataset == "Livox":
            # alpha = np.log(1 + self.cls_weight)
            # alpha = alpha / alpha.max()
            beta_0 = self.cls_weight
            beta = np.array([beta_0[1], beta_0[2]])
            beta = beta / beta.max()
        # beta[0] = 0
        if self.recorder is not None:
            self.recorder.logger.info("focal_loss_img beta: {}".format(beta))

        criterion["focal_loss_img"] = pc_processor.loss.FocalSoftmaxLossImg(
            n_classes=2, gamma=self.settings.gamma, beta=beta, softmax=False,
            signal=None)  # 这里的gamma默认是yaml中的gamma（self.settings.gamma）

        # set device
        for _, v in criterion.items():
            v.cuda()
        return criterion

    # -------------------------------------------------------------------------
    # functions for running
    # -------------------------------------------------------------------------

    def _backward(self, loss):
        self.optimizer.zero_grad()
        self.aux_optimizer.zero_grad()
        loss.backward()

        for param in self.model.parameters():
            if param.grad is not None:
                param.grad[torch.isnan(param.grad)] = 0.0
        self.optimizer.step()
        self.aux_optimizer.step()

    def _computeClassifyLoss(self, pred, label, label_mask, signal):

        loss_foc = self.criterion["focal_loss"](
            pred, label, mask=label_mask, signal=signal)

        loss_lov = self.criterion["lovasz"](
            pred, label)

        # print("loss_foc", np.shape(loss_foc), "loss_lov", np.shape(loss_lov) )
        return loss_lov, loss_foc

    def _computeClassifyLossImg(self, pred, label, label_mask, signal):  # 计算分类损失

        loss_foc = self.criterion["focal_loss_img"](
            pred, label, mask=label_mask, signal=signal)

        loss_lov = self.criterion["lovasz_img"](
            pred, label)

        # print("loss_foc", np.shape(loss_foc), "loss_lov", np.shape(loss_lov) )
        return loss_lov, loss_foc

    def _computePerceptionAwareLoss(
            self, pcd_entropy, img_entropy,
            pcd_pred, pcd_pred_log, img_pred, img_pred_log, label_mask, epoch):

        epoch = self.settings.n_epochs
        pcd_confidence = 1 - pcd_entropy
        img_confidence = 1 - img_entropy
        information_importance = pcd_confidence - img_confidence

        pcd_guide_mask = (pcd_confidence.ge(self.settings.tau) * label_mask).float()
        img_guide_mask = img_confidence.ge(self.settings.tau).float()

        pcd_guide_weight = information_importance.gt(0).float(
        ) * information_importance.abs() * pcd_guide_mask
        img_guide_weight = information_importance.lt(0).float(
        ) * information_importance.abs() * img_guide_mask

        pcd_pred = pcd_pred[:, 1:, :, :]
        pcd_pred_log = pcd_pred_log[:, 1:, :, :]

        # compute kl loss
        loss_per_pcd = (self.criterion["kl_loss"](
            pcd_pred_log, img_pred) * img_guide_weight.unsqueeze(1)).mean()  # 用img_guide_weight 得到loss_per_pcd
        loss_per_img = (self.criterion["kl_loss"](
            img_pred_log, pcd_pred) * pcd_guide_weight.unsqueeze(1)).mean()
        loss_per = loss_per_pcd + loss_per_img
        return loss_per, pcd_guide_weight, img_guide_weight

    def run(self, epoch, mode="Train"):
        if mode == "Train":
            dataloader = self.train_loader
            self.model.train()
            if self.settings.distributed:
                self.train_sampler.set_epoch(epoch)

        elif mode == "Validation":
            dataloader = self.val_loader
            self.model.eval()
        else:
            raise ValueError("invalid mode: {}".format(mode))

        loss_meter = pc_processor.utils.AverageMeter()
        loss_focal_meter = pc_processor.utils.AverageMeter()
        loss_lovasz_meter = pc_processor.utils.AverageMeter()
        entropy_meter = pc_processor.utils.AverageMeter()
        self.metrics.reset()

        loss_img_focal_meter = pc_processor.utils.AverageMeter()
        loss_img_lovasz_meter = pc_processor.utils.AverageMeter()
        entropy_img_meter = pc_processor.utils.AverageMeter()
        self.metrics_img.reset()

        loss_perception_meter = pc_processor.utils.AverageMeter()

        total_iter = len(dataloader)
        t_start = time.time()

        feature_mean = torch.Tensor(self.settings.config["sensor"]["img_mean"]).unsqueeze(
            0).unsqueeze(2).unsqueeze(2).cuda()
        feature_std = torch.Tensor(self.settings.config["sensor"]["img_stds"]).unsqueeze(
            0).unsqueeze(2).unsqueeze(2).cuda()

        for i, (input_feature, input_mask, input_label, input_imglabel) in enumerate(dataloader):
            t_process_start = time.time()
            input_feature = input_feature.cuda()
            input_mask = input_mask.cuda()

            input_feature_old = input_feature.cpu().numpy()
            img_feature_old = input_feature_old[:, 5:8]

            input_feature[:, 0:5] = (
                                            input_feature[:, 0:5] - feature_mean) / feature_std * \
                                    input_mask.unsqueeze(1).expand_as(input_feature[:, 0:5])
            pcd_feature = input_feature[:, 0:5]
            img_feature = input_feature[:, 5:8]
            input_imglabel = (input_imglabel.long()).cuda()
            imgmask_mask = input_imglabel.gt(-1)

            input_label = (input_label.long()).cuda()
            label_mask = input_label.gt(0)

            # ---------------------------------------------------------------------------
            # if epoch >= 9:
            #     focal_weight = 1.0
            # else:
            #     focal_weight = 0.1
            focal_weight = 1.0
            # ---------------------------------------------------------------------------

            # forward propergation
            if mode == "Train":
                # lidar_pred, camera_pred = self.model(pcd_feature, img_feature)
                lidar_pred, camera_pred = self.model(pcd_feature, img_feature)
                lidar_pred_log = torch.log(lidar_pred.clamp(min=1e-8))

                # lidar_tmp = lidar_pred[:, 1:, :, :].sum(dim=1, keepdim=True)
                # lidar_tmp_pred = lidar_pred[:, 1:, :, :] / lidar_tmp
                # lidar_tmp_pred_log = torch.log(lidar_tmp_pred.clamp(min=1e-8))

                # compute pcd entropy: p * log p
                pcd_entropy = -(lidar_pred * lidar_pred_log).sum(1) / \
                              math.log(self.settings.nclasses)  # 这里做了修改

                loss_lov, loss_foc = self._computeClassifyLoss(
                    pred=lidar_pred, label=input_label, label_mask=label_mask, signal=epoch)

                # compute img entropy
                camera_pred_log = torch.log(
                    camera_pred.clamp(min=1e-8))
                # normalize to [0,1)
                img_entropy = - \
                                  (camera_pred * camera_pred_log).sum(1) / \
                              math.log(2)  # 这里进行了修改，把self.settings.nclasses改成了2

                # loss_lov_cam, loss_foc_cam = self._computeClassifyLoss(
                #     pred=camera_pred, label=input_label, label_mask=label_mask)
                loss_lov_cam, loss_foc_cam = self._computeClassifyLossImg(
                    pred=camera_pred, label=input_imglabel, label_mask=imgmask_mask, signal=epoch)

                loss_per, pcd_guide_weight, img_guide_weight = self._computePerceptionAwareLoss(
                    pcd_entropy=pcd_entropy, img_entropy=img_entropy,
                    pcd_pred=lidar_pred, pcd_pred_log=lidar_pred_log,
                    img_pred=camera_pred, img_pred_log=camera_pred_log, label_mask=label_mask,
                    epoch=self.settings.n_epochs
                )

                # print("total_loss", loss_foc, loss_lov, loss_foc_cam, loss_lov_cam, loss_per)
                # total_loss = loss_foc + loss_lov * self.settings.lambda_ + \
                #              loss_foc_cam + loss_lov_cam * self.settings.lambda_ \
                # +loss_per * self.settings.gamma
                total_loss = loss_foc_cam + loss_lov_cam * self.settings.lambda_

                if self.settings.n_gpus > 1:
                    total_loss = total_loss.mean()

                # backward
                self._backward(total_loss)
                # update lr after backward (required by pytorch)
                self.scheduler.step()
                self.aux_scheduler.step()

            else:
                with torch.no_grad():
                    lidar_pred, camera_pred = self.model(pcd_feature, img_feature)
                    lidar_pred_log = torch.log(lidar_pred.clamp(min=1e-8))
                    # compute pcd entropy: p * log p
                    pcd_entropy = -(lidar_pred * lidar_pred_log).sum(1) / \
                                  math.log(self.settings.nclasses)

                    loss_lov, loss_foc = self._computeClassifyLoss(
                        pred=lidar_pred, label=input_label, label_mask=label_mask, signal=epoch)

                    # compute img entropy
                    camera_pred_log = torch.log(
                        camera_pred.clamp(min=1e-8))
                    # normalize to [0,1)
                    img_entropy = - \
                                      (camera_pred * camera_pred_log).sum(1) / \
                                  math.log(2)

                    loss_lov_cam, loss_foc_cam = self._computeClassifyLossImg(
                        pred=camera_pred, label=input_imglabel, label_mask=imgmask_mask, signal=epoch)

                    loss_per, pcd_guide_weight, img_guide_weight = self._computePerceptionAwareLoss(
                        pcd_entropy=pcd_entropy, img_entropy=img_entropy,
                        pcd_pred=lidar_pred, pcd_pred_log=lidar_pred_log,
                        img_pred=camera_pred, img_pred_log=camera_pred_log, label_mask=label_mask,
                        epoch=self.settings.n_epochs
                    )

                    # total_loss = loss_foc + loss_lov * self.settings.lambda_ + \
                    #              loss_foc_cam + loss_lov_cam * self.settings.lambda_ \
                    #              + loss_per * self.settings.gamma
                    total_loss = loss_foc_cam + loss_lov_cam * self.settings.lambda_
                    print("total_loss", loss_foc, loss_lov, loss_foc_cam, loss_lov_cam, loss_per)

                    if self.settings.n_gpus > 1:
                        total_loss = total_loss.mean()

            # measure accuracy and record loss
            loss = total_loss.mean()

            # # check output
            # measure accuracy and record loss
            with torch.no_grad():
                # compute iou and acc
                argmax = lidar_pred.argmax(dim=1)
                self.metrics.addBatch(argmax, input_label)
                mean_iou, class_iou = self.metrics.getIoU()
                mean_acc, class_acc = self.metrics.getAcc()
                mean_recall, class_recall = self.metrics.getRecall()

                argmax_img = camera_pred.argmax(dim=1)
                self.metrics_img.addBatch(argmax_img, input_imglabel)
                mean_iou_img, class_iou_img = self.metrics_img.getIoU()
                mean_acc_img, class_acc_img = self.metrics_img.getAcc()
                mean_recall_img, class_recall_img = self.metrics_img.getRecall()

            loss_meter.update(total_loss.item(), input_feature.size(0))
            loss_focal_meter.update(loss_foc.item(), input_feature.size(0))
            loss_lovasz_meter.update(loss_lov.item(), input_feature.size(0))
            entropy_meter.update(pcd_entropy.mean().item(), input_feature.size(0))

            loss_img_lovasz_meter.update(loss_lov_cam.item(), input_feature.size(0))
            loss_img_focal_meter.update(loss_foc_cam.item(), input_feature.size(0))
            entropy_img_meter.update(img_entropy.mean().item(), input_feature.size(0))

            loss_perception_meter.update(loss_per.item(), input_feature.size(0))

            # timer logger ----------------------------------------
            t_process_end = time.time()

            data_cost_time = t_process_start - t_start
            process_cost_time = t_process_end - t_process_start

            self.remain_time.update(cost_time=(time.time() - t_start), mode=mode)
            remain_time = datetime.timedelta(
                seconds=self.remain_time.getRemainTime(
                    epoch=epoch, iters=i, total_iter=total_iter, mode=mode
                ))
            t_start = time.time()

            if self.recorder is not None:
                for g in self.optimizer.param_groups:
                    lr = g["lr"]
                    break
                log_str = ">>> {} E[{:03d}|{:03d}] I[{:04d}|{:04d}] DT[{:.3f}] PT[{:.3f}] ".format(
                    mode, self.settings.n_epochs, epoch + 1, total_iter, i + 1, data_cost_time, process_cost_time)
                log_str += "LR {:0.6f} Loss {:0.4f} LidarAcc {:0.4f} LidarIOU {:0.4F} LidarAcc_0 {:0.4f} LidarIOU_0 {:0.4F} LidarAcc_1 {:0.4f} LidarIOU_1 {:0.4F} LidarRecall {:0.4f} LidarEntropy {:0.4f} ".format(
                    lr, loss.item(), mean_acc.item(), mean_iou.item(), class_acc[1], class_iou[1], class_acc[2],
                    class_iou[2], mean_recall.item(), entropy_meter.avg)
                log_str += "ImgAcc {:0.4f} ImgIOU {:0.4F} ImgAcc_0 {:0.4f} ImgIOU_0 {:0.4F} ImgAcc_1 {:0.4f} ImgIOU_1 {:0.4F} ImgRecall {:0.4f} ImgEntropy {:0.4f} ".format(
                    mean_acc_img.item(), mean_iou_img.item(), class_acc_img[0], class_iou_img[0], class_acc_img[1],
                    class_iou_img[1], mean_recall_img.item(), entropy_img_meter.avg)
                log_str += "RT {}".format(remain_time)
                self.recorder.logger.info(log_str)

            if self.settings.is_debug:
                break

        # tensorboard logger
        if self.recorder is not None:
            # scalar log
            self.recorder.tensorboard.add_scalar(
                tag="{}_Loss".format(mode), scalar_value=loss_meter.avg, global_step=epoch)
            self.recorder.tensorboard.add_scalar(
                tag="{}_LossFocal".format(mode), scalar_value=loss_focal_meter.avg, global_step=epoch)
            self.recorder.tensorboard.add_scalar(
                tag="{}_LossLovasz".format(mode), scalar_value=loss_lovasz_meter.avg, global_step=epoch)

            self.recorder.tensorboard.add_scalar(
                tag="{}_lr".format(mode), scalar_value=lr, global_step=epoch)
            self.recorder.tensorboard.add_scalar(
                tag="{}_entropy".format(mode), scalar_value=entropy_meter.avg, global_step=epoch)

            self.recorder.tensorboard.add_scalar(
                tag="{}_meanAcc".format(mode), scalar_value=mean_acc.item(), global_step=epoch)
            self.recorder.tensorboard.add_scalar(
                tag="{}_meanIOU".format(mode), scalar_value=mean_iou.item(), global_step=epoch)
            self.recorder.tensorboard.add_scalar(
                tag="{}_meanRecall".format(mode), scalar_value=mean_recall.item(), global_step=epoch)

            for i, (_, v) in enumerate(self.mapped_cls_name.items()):
                self.recorder.tensorboard.add_scalar(
                    tag="{}_{:02d}_{}_Acc".format(mode, i, v), scalar_value=class_acc[i].item(), global_step=epoch)
                self.recorder.tensorboard.add_scalar(
                    tag="{}_{:02d}_{}_Recall".format(mode, i, v), scalar_value=class_recall[i].item(),
                    global_step=epoch)
                self.recorder.tensorboard.add_scalar(
                    tag="{}_{:02d}_{}_IOU".format(mode, i, v), scalar_value=class_iou[i].item(), global_step=epoch)

            # record img branch acc, recall and iou
            self.recorder.tensorboard.add_scalar(
                tag="{}_LossImageFocal".format(mode), scalar_value=loss_img_focal_meter.avg, global_step=epoch)
            self.recorder.tensorboard.add_scalar(
                tag="{}_LossImageLovasz".format(mode), scalar_value=loss_img_lovasz_meter.avg, global_step=epoch)
            self.recorder.tensorboard.add_scalar(
                tag="{}_ImageEntropy".format(mode), scalar_value=entropy_img_meter.avg, global_step=epoch)

            self.recorder.tensorboard.add_scalar(
                tag="{}_LossPerception".format(mode), scalar_value=loss_perception_meter.avg, global_step=epoch)

            self.recorder.tensorboard.add_scalar(
                tag="{}_Image_meanAcc".format(mode), scalar_value=mean_acc_img.item(), global_step=epoch)
            self.recorder.tensorboard.add_scalar(
                tag="{}_Image_meanIOU".format(mode), scalar_value=mean_iou_img.item(), global_step=epoch)
            self.recorder.tensorboard.add_scalar(
                tag="{}_Image_meanRecall".format(mode), scalar_value=mean_recall_img.item(), global_step=epoch)
            # --------------------------------------------------------------------------------------------------------------
            imgdict = {0: 'vegetation', 1: 'road'}

            lidardict = {0: 'unlabeled', 1: 'vegetation', 2: 'road'}
            # --------------------------------------------------------------------------------------------------------------
            for i, (_, v) in enumerate(
                    imgdict.items()):  # self.mapped_cls_name.items()=dict_items([(0, 'unlabeled'), (1, 'vegetation'), (2, 'road')])
                self.recorder.tensorboard.add_scalar(
                    tag="{}_{:02d}_{}_ImageAcc".format(mode, i, v), scalar_value=class_acc_img[i].item(),
                    global_step=epoch)
                self.recorder.tensorboard.add_scalar(
                    tag="{}_{:02d}_{}_ImageRecall".format(mode, i, v), scalar_value=class_recall_img[i].item(),
                    global_step=epoch)
                self.recorder.tensorboard.add_scalar(
                    tag="{}_{:02d}_{}_ImageIOU".format(mode, i, v), scalar_value=class_iou_img[i].item(),
                    global_step=epoch)

            if epoch % self.settings.print_frequency == 0 and self.settings.dataset != "nuScenes":
                # img log
                for i in range(pcd_feature.size(1)):
                    self.recorder.tensorboard.add_image(
                        "{}_PCDFeature_{}".format(mode, i), pcd_feature[0, i:i + 1].cpu(), epoch)

                if camera_pred is not None:
                    for i in range(camera_pred.size(1)):
                        self.recorder.tensorboard.add_image(
                            "{}_RGBPred_cls_{:02d}_{}".format(mode, i, imgdict[i]),
                            camera_pred[0, i:i + 1].cpu(), epoch)

                for i in range(lidar_pred.size(1)):
                    self.recorder.tensorboard.add_image(
                        "{}_LidarPred_cls_{:02d}_{}".format(mode, i, lidardict[i]),
                        lidar_pred[0, i:i + 1].cpu(),
                        epoch)

                for i in range(camera_pred.size(1)):
                    self.recorder.tensorboard.add_image(
                        "{}_RGBPred_cls_{:02d}_{}".format(mode, i, imgdict[i]),
                        camera_pred[0, i:i + 1].cpu(),
                        epoch)

                # record entropy
                self.recorder.tensorboard.add_image(
                    "{}_PredEntropy".format(mode), pcd_entropy[0].unsqueeze(0), epoch)
                self.recorder.tensorboard.add_image(
                    "{}_RGBPredEntropy".format(mode), img_entropy[0].unsqueeze(0), epoch)
                self.recorder.tensorboard.add_image(
                    "{}_RGBGuideWeight".format(mode), img_guide_weight[0].unsqueeze(0), epoch)
                self.recorder.tensorboard.add_image(
                    "{}_PCDGuideWeight".format(mode), pcd_guide_weight[0].unsqueeze(0), epoch)

                for i in range(lidar_pred.size(1)):
                    self.recorder.tensorboard.add_image("{}_LidarLabel_cls_{:02d}_{}".format(
                        mode, i, lidardict[i]), input_label[0:1].eq(i).cpu(), epoch)

                for i in range(camera_pred.size(1)):
                    self.recorder.tensorboard.add_image("{}_RGBLabel_cls_{:02d}_{}".format(
                        mode, i, imgdict[i]), input_imglabel[0:1].eq(i).cpu(), epoch)

                # self.recorder.tensorboard.add_image(
                #     "{}_RGB".format(mode), img_feature[0].cpu(), epoch)

            log_str = ">>> {} Loss {:0.4f} LidarAcc {:0.4f} LidarIOU {:0.4F} LidarRecall {:0.4f} LidarACC_0 {:0.4f} LidarIOU_0 {:0.4f} LidarACC_1 {:0.4F} LidarIOU_1 {:0.4F} ".format(
                mode, loss_meter.avg, mean_acc.item(), mean_iou.item(), mean_recall.item(), class_acc[1].item(),
                class_iou[1].item(), class_acc[2].item(), class_iou[2].item())
            log_str += "ImgAcc {:0.4f} ImgIOU {:0.4F} ImgRecall {:0.4f} ImgACC_0 {:0.4f} ImgIOU_0 {:0.4f} ImgACC_1 {:0.4F} ImgIOU_1 {:0.4F} ".format(
                mean_acc_img.item(), mean_iou_img.item(), mean_recall_img.item(), class_acc_img[0].item(),
                class_iou_img[0].item(), class_acc_img[1].item(), class_iou_img[1].item())
            self.recorder.logger.info(log_str)

        result_metrics = {
            "LidarAcc": mean_acc.item(),
            "LidarAcc_0": class_acc[1].item(),
            "LidarAcc_1": class_acc[2].item(),
            "LidarIOU": mean_iou.item(),
            "LidarIOU_0": class_iou[1].item(),
            "LidarIOU_1": class_iou[2].item(),
            "LidarRecall": mean_recall.item(),
            "ImgAcc": mean_acc_img.item(),
            "ImgAcc_0": class_acc_img[0].item(),
            "ImgAcc_1": class_acc_img[1].item(),
            "ImgIOU": mean_iou_img.item(),
            "ImgIOU_0": class_iou_img[0].item(),
            "ImgIOU_1": class_iou_img[1].item(),
            "ImgRecall": mean_recall_img.item(),
            "last": 0
        }

        return result_metrics
