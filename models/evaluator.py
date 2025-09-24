import os
import numpy as np
import matplotlib.pyplot as plt
from models.networks import *
from misc.metric_tool import ConfuseMatrixMeter
from misc.logger_tool import Logger
from numpy.core.fromnumeric import choose
from utils import de_norm
import utils
import pandas as pd
import cv2
import cv2
import numpy as np
import matplotlib.pyplot as plt
import wandb

def cm2F1(confusion_matrix):
    hist = confusion_matrix
    n_class = hist.shape[0]
    tp = np.diag(hist)
    sum_a1 = hist.sum(axis=1)
    sum_a0 = hist.sum(axis=0)
    # ---------------------------------------------------------------------- #
    # 1. Accuracy & Class Accuracy
    # ---------------------------------------------------------------------- #
    acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)

    # recall
    recall = tp / (sum_a1 + np.finfo(np.float32).eps)
    mean_recall = np.nanmean(recall)

    # precision
    precision = tp / (sum_a0 + np.finfo(np.float32).eps)
    mean_precision = np.nanmean(precision)

    # F1 score
    F1 = 2 * recall * precision / (recall + precision + np.finfo(np.float32).eps)
    mean_F1 = np.nanmean(F1)
    return mean_F1, mean_precision, mean_recall

class CDEvaluator():

    def __init__(self, args, dataloader):
        self.args = args
        self.dataloader = dataloader

        self.n_class = args.n_class
        # define G
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)
        self.device = torch.device("cuda:%s" % args.gpu_ids[0] if torch.cuda.is_available() and len(args.gpu_ids)>0
                                   else "cpu")
        # print(self.device)

        # define some other vars to record the training states
        self.running_metric = ConfuseMatrixMeter(n_class=self.n_class)


        # define logger file
        logger_path = os.path.join(args.checkpoint_dir, 'log_test.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)


        #  training log
        self.epoch_acc = 0
        self.best_val_acc = 0.0
        self.best_epoch_id = 0
        self.batch_size = args.batch_size
        self.steps_per_epoch = len(dataloader)

        self.G_pred3 = None
        self.G_pred2 = None
        self.G_pred1 = None
        self.G_pred = None
        self.out2 = None

        self.pred_vis = None
        self.batch = None
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        self.checkpoint_dir = args.checkpoint_dir
        self.vis_dir = args.vis_dir

        # check and create model dir
        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)
        if os.path.exists(self.vis_dir) is False:
            os.mkdir(self.vis_dir)

        # fot vis hot



    def _load_checkpoint(self, checkpoint_name='best_ckpt.pt'):

        if os.path.exists(os.path.join(self.checkpoint_dir, checkpoint_name)):
            self.logger.write('loading last checkpoint...\n')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, checkpoint_name), map_location=self.device)
            #self.net_G = nn.DataParallel(self.net_G)   # 多卡训练的结果 单卡测试 要在这里开启

            state_dict = checkpoint['model_G_state_dict']
            state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
            self.net_G.load_state_dict(state_dict)

            #self.net_G.load_state_dict(checkpoint['model_G_state_dict'])
            self.net_G.to(self.device)

            # update some other states
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']

            self.logger.write('Eval Historical_best_acc = %.4f (at epoch %d)\n' %
                  (self.best_val_acc, self.best_epoch_id))
            self.logger.write('\n')

        else:
            raise FileNotFoundError('no such checkpoint %s' % checkpoint_name)


    def _visualize_pred(self):
        pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
        pred_vis = pred * 255
        return pred_vis


    def _update_metric(self):
        """
        update metric
        """
        target = self.batch['L'].to(self.device).detach()
        G_pred = self.G_pred.detach()
        G_pred = torch.argmax(G_pred, dim=1)

        current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=target.cpu().numpy())
        return current_score


    def _collect_running_batch_states(self):
        running_acc = self._update_metric()
        m = len(self.dataloader)

        if np.mod(self.batch_id, 100) == 1:
            message = 'Is_training: %s. [%d,%d],  running_mf1: %.5f\n' %\
                      (self.is_training, self.batch_id, m, running_acc)
            self.logger.write(message)
        

    def _collect_epoch_states(self):

        scores_dict = self.running_metric.get_scores()

        np.save(os.path.join(self.checkpoint_dir, 'scores_dict.npy'), scores_dict)

        self.epoch_acc = scores_dict['mf1']
        #wandb.log({"eval_mf1": self.epoch_acc})

        with open(os.path.join(self.checkpoint_dir, '%s.txt' % (self.epoch_acc)),
                  mode='a') as file:
            pass

        message = ''
        for k, v in scores_dict.items():
            message += '%s: %.5f ' % (k, v)
        self.logger.write('%s\n' % message)  # save the message

        self.logger.write('\n')

    def _clear_cache(self):
        self.running_metric.clear()

    #def _forward_pass(self, batch):
    def _forward_pass(self, batch, bid):    # hot2
        self.batch = batch
        img_in1 = batch['A'].to(self.device)
        img_in2 = batch['B'].to(self.device)

        if self.args.net_G == 'STINetSingle_v25' or self.args.net_G == 'STINetSingle_v25_1':
            self.G_pred2, self.G_pred_am, self.G_middle2, self.G_middle_am, self.G_low2, self.G_low_am = self.net_G(img_in1,
                                                                                                              img_in2,
                                                                                                              bid)  # hot3
            self.G_pred = self.G_pred2 + self.G_pred_am

    def eval_models(self,checkpoint_name='best_ckpt.pt'):

        self._load_checkpoint(checkpoint_name)

        ################## Eval ##################
        ##########################################
        self.logger.write('Begin evaluation...\n')
        self._clear_cache()
        self.is_training = False
        self.net_G.eval()

        # Iterate over data.
        for self.batch_id, batch in enumerate(self.dataloader, 0):
            with torch.no_grad():
                self._forward_pass(batch, self.batch_id)    # hot1
            self._collect_running_batch_states()
        self._collect_epoch_states()

    def __init__(self, args, dataloader):
        self.args = args
        self.dataloader = dataloader

        self.n_class = args.n_class
        # define G
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)
        self.device = torch.device("cuda:%s" % args.gpu_ids[0] if torch.cuda.is_available() and len(args.gpu_ids) > 0
                                   else "cpu")
        # print(self.device)

        # define some other vars to record the training states
        self.running_metric = ConfuseMatrixMeter(n_class=self.n_class)


        # define logger file
        logger_path = os.path.join(args.checkpoint_dir, 'log_test.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)

        #  training log
        self.epoch_acc = 0
        self.best_val_acc = 0.0
        self.best_epoch_id = 0
        self.batch_size = args.batch_size
        self.steps_per_epoch = len(dataloader)

        self.G_pred3 = None
        self.G_pred2 = None
        self.G_pred1 = None
        self.G_pred = None
        self.out2 = None

        self.pred_vis = None
        self.batch = None
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        self.checkpoint_dir = args.checkpoint_dir
        self.vis_dir = args.vis_dir

        # check and create model dir
        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)
        if os.path.exists(self.vis_dir) is False:
            os.mkdir(self.vis_dir)

        # from mamba
        self.acc_meter = AverageMeter()
        self.preds_all = []
        self.labels_all = []

    def _load_checkpoint(self, checkpoint_name='best_ckpt.pt'):

        if os.path.exists(os.path.join(self.checkpoint_dir, checkpoint_name)):
            self.logger.write('loading last checkpoint...\n')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, checkpoint_name), map_location=self.device)
            # self.net_G = nn.DataParallel(self.net_G)   # 多卡训练的结果 单卡测试 要在这里开启

            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])

            self.net_G.to(self.device)

            # update some other states
            #self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']

            self.logger.write('Eval Historical_best_acc = %.4f (at epoch %d)\n' %
                              (self.best_val_acc, self.best_epoch_id))
            self.logger.write('\n')

        else:
            raise FileNotFoundError('no such checkpoint %s' % checkpoint_name)

    def _visualize_pred(self):
        pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
        pred_vis = pred * 255
        return pred_vis

    def _update_metric(self):
        """
        update metric
        """
        target = self.batch['L'].to(self.device).detach()
        G_pred = self.G_pred.detach()
        # G_pred = self.G_pred
        G_pred = torch.argmax(G_pred, dim=1)

        current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=target.cpu().numpy())
        return current_score

    def _updata_metric_mamba(self):
        labels_cd = self.labels_cd.cpu().numpy()
        labels_A = self.labels_clf_t1.cpu().numpy()
        labels_B = self.labels_clf_t2.cpu().numpy()

        change_mask = torch.argmax(self.output_1, axis=1).cpu().numpy()

        preds_A = torch.argmax(self.output_semantic_t1, dim=1).cpu().numpy()
        preds_B = torch.argmax(self.output_semantic_t2, dim=1).cpu().numpy()

        preds_scd = (preds_A - 1) * 6 + preds_B
        preds_scd[change_mask == 0] = 0

        labels_scd = (labels_A - 1) * 6 + labels_B
        labels_scd[labels_cd == 0] = 0

        for (pred_scd, label_scd) in zip(preds_scd, labels_scd):
            acc_A, valid_sum_A = accuracy(pred_scd, label_scd)
            self.preds_all.append(pred_scd)
            self.labels_all.append(label_scd)
            self.acc = acc_A
            self.acc_meter.update(acc)

    def _collect_running_batch_states(self):

        ###########################################################
        ######################### 1.计算单个样本的评价指标 ###########

        # if self.batch_id < 4001:
        #     current_score = self._update_metric()
        #     # Get the scores for the current sample
        #     scores_dict = self.running_metric.get_scores()

        #     self.f1 = scores_dict['F1_1']
        #     self.iou = scores_dict['iou_1']
        #     self.pr = scores_dict['precision_1']
        #     self.recall = scores_dict['recall_1']
        #     self.logger.write('%s\n' % self.f1)

        #     self.name = f"{self.net_G}"
        #     metrics_string = f'Batch ID: {self.batch_id}, F1: {self.f1}, IOU: {self.iou}, Precision: {self.pr}, Recall: {self.recall}\n'

        #     with open('/home/wyuan/code/CDWorkflow/data_analysis/whu/icif.txt', 'a') as f:
        #     #file_path = os.path.join(self.vis_dir, 'results_300.txt')
        #     # with open(file_path, 'a') as f:
        #         f.write(metrics_string)

        #     self.running_metric.clear()

        ###########################################################
        ######################### 2.总体指标 ######################

        running_acc = self._update_metric()
        #running_acc = self._updata_metric_mamba()
        m = len(self.dataloader)

        if np.mod(self.batch_id, 100) == 1:
            message = 'Is_training: %s. [%d,%d],  running_mf1: %.5f\n' % \
                      (self.is_training, self.batch_id, m, running_acc)
            self.logger.write(message)

        ###########################################################
        ######################### 3. 可视化 ########################
        # if np.mod(self.batch_id, 1) == 0: # LEVIR, GZCD, WHU, SYSU-CD
        #     if self.batch_id < 132:
        #         vis_input = utils.make_numpy_grid(de_norm(self.batch['A']))
        #         vis_input2 = utils.make_numpy_grid(de_norm(self.batch['B']))

        #         vis_pred = utils.make_numpy_grid(self._visualize_pred())

        #         vis_gt = utils.make_numpy_grid(self.batch['L'])
        #         vis = np.concatenate([vis_input, vis_input2, vis_pred, vis_gt], axis=0)
        #         vis = np.clip(vis, a_min=0.0, a_max=1.0)
        #         file_name = os.path.join(
        #             self.vis_dir, 'eval_' + str(self.batch_id)+'.jpg')
        #         plt.imsave(file_name, vis)

        #         for i in range(self.batch_size):
        #             fig = np.clip(utils.make_numpy_grid(self._visualize_pred()[i,:,:,:]), a_min=0.0, a_max=1.0)
        #             #plt.imsave(self.vis_dir+"/predict_"+str(self.batch_id)+'_'+str(i)+'.png',fig)
        #             plt.imsave(self.vis_dir+"/"+str(self.batch_id)+'_'+str(i)+'.png',fig)

        # vis_input = np.clip(utils.make_numpy_grid(de_norm(self.batch['A'])[i,:,:,:]), a_min=0.0, a_max=1.0)
        # plt.imsave(self.vis_dir+"/t1_"+str(self.batch_id)+'_'+str(i)+'.png',vis_input)

        # vis_input2 = np.clip(utils.make_numpy_grid(de_norm(self.batch['B'])[i,:,:,:]), a_min=0.0, a_max=1.0)
        # plt.imsave(self.vis_dir+"/t2_"+str(self.batch_id)+'_'+str(i)+'.png',vis_input2)

        # vis_gt = np.clip(utils.make_numpy_grid(self.batch['L'][i,:,:,:]), a_min=0.0, a_max=1.0)
        # plt.imsave(self.vis_dir+"/gt_"+str(self.batch_id)+'_'+str(i)+'.png',vis_gt)

        ###########################################################
        ######################### 4. 03vis_hot ########################
        # if self.args.data_name == 'LEVIR':
        #     if self.batch_id == 22:
        #         savepath=r'/home/wyuan/code/CDWorkflow/sherlockWorkdir/CD2/vis/04_basehot'
        #         draw_features(8,3,(F.interpolate(self.out2, scale_factor=4, mode='bilinear')).cpu().detach().numpy(),"{}/levir_22.png".format(savepath))
        #     if self.batch_id == 41:
        #         savepath=r'/home/wyuan/code/CDWorkflow/sherlockWorkdir/CD2/vis/04_basehot'
        #         draw_features(8,3,(F.interpolate(self.out2, scale_factor=4, mode='bilinear')).cpu().detach().numpy(),"{}/levir_41.png".format(savepath))

        # if self.args.data_name == 'GZCD':
        #     if self.batch_id == 103:
        #         savepath=r'/home/wyuan/code/CDWorkflow/sherlockWorkdir/CD2/vis/04_basehot'
        #         draw_features(8,3,(F.interpolate(self.out2, scale_factor=4, mode='bilinear')).cpu().detach().numpy(),"{}/103.png".format(savepath))
        #     if self.batch_id == 118:
        #         savepath=r'/home/wyuan/code/CDWorkflow/sherlockWorkdir/CD2/vis/04_basehot'
        #         draw_features(8,3,(F.interpolate(self.out2, scale_factor=4, mode='bilinear')).cpu().detach().numpy(),"{}/118.png".format(savepath))

        # if self.args.data_name == 'WHU':
        #     if self.batch_id == 28:
        #         savepath=r'/home/wyuan/code/CDWorkflow/sherlockWorkdir/CD2/vis/04_basehot'
        #         draw_features(8,3,(F.interpolate(self.out2, scale_factor=4, mode='bilinear')).cpu().detach().numpy(),"{}/28.png".format(savepath))
        #     if self.batch_id == 129:
        #         savepath=r'/home/wyuan/code/CDWorkflow/sherlockWorkdir/CD2/vis/04_basehot'
        #         draw_features(8,3,(F.interpolate(self.out2, scale_factor=4, mode='bilinear')).cpu().detach().numpy(),"{}/129.png".format(savepath))

        # if self.args.data_name == 'SYSU-CD':
        #     if self.batch_id == 12:
        #         savepath=r'/home/wyuan/code/CDWorkflow/sherlockWorkdir/CD2/vis/04_basehot'
        #         draw_features(8,3,(F.interpolate(self.out2, scale_factor=4, mode='bilinear')).cpu().detach().numpy(),"{}/12.png".format(savepath))
        #     if self.batch_id == 97:
        #         savepath=r'/home/wyuan/code/CDWorkflow/sherlockWorkdir/CD2/vis/04_basehot'
        #         draw_features(8,3,(F.interpolate(self.out2, scale_factor=4, mode='bilinear')).cpu().detach().numpy(),"{}/97.png".format(savepath))

    def _collect_epoch_states(self):

        scores_dict = self.running_metric.get_scores()

        np.save(os.path.join(self.checkpoint_dir, 'scores_dict.npy'), scores_dict)

        self.epoch_acc = scores_dict['mf1']
        #wandb.log({"eval_mf1": self.epoch_acc})

        with open(os.path.join(self.checkpoint_dir, '%s.txt' % (self.epoch_acc)),
                  mode='a') as file:
            pass

        message = ''
        for k, v in scores_dict.items():
            message += '%s: %.5f ' % (k, v)
        self.logger.write('%s\n' % message)  # save the message

        self.logger.write('\n')

    def _clear_cache(self):
        self.running_metric.clear()

    def _forward_pass(self, batch):
        # self.batch = batch
        # img_in1 = batch['A'].to(self.device)
        # img_in2 = batch['B'].to(self.device)
        self.batch = batch
        self.pre_change_imgs, self.post_change_imgs, self.label_cd, self.label_clf_t1, self.label_clf_t2, _ = batch
        pre_change_imgs = self.pre_change_imgs.cuda()
        post_change_imgs = self.post_change_imgs.cuda()
        labels_cd = self.label_cd.cuda().long()
        labels_clf_t1 = self.label_clf_t1.cuda().long()
        labels_clf_t2 = self.label_clf_t2.cuda().long()

        if self.args.net_G == 'STINetSingle_v2':
            # if self.args.net_G == 'DMINet' or self.args.net_G == 'STIResCascadeNet' or self.args.net_G == 'STINetSingle' or self.args.net_G == 'STISingleNet' or self.args.net_G == 'STISingleNet_x' or self.args.net_G == 'STISingleBooNet' or self.args.net_G == 'STISingle4BooNet' or self.args.net_G == 'STISingleBooSwinNet' or self.args.net_G == 'STINetSingle_v2':
            self.G_pred1, self.G_pred2, self.G_middle1, self.G_middle2, self.G_low1, self.G_low2 = self.net_G(img_in1,
                                                                                                              img_in2)
        elif self.args.net_G == 'STMambaSCD':
            self.output_1, self.output_semantic_t1, self.output_semantic_t2 = self.net_G(self.pre_change_imgs, self.post_change_imgs)
            labels_cd = labels_cd.cpu().numpy()
            labels_A = labels_clf_t1.cpu().numpy()
            labels_B = labels_clf_t2.cpu().numpy()

            change_mask = torch.argmax(self.output_1, axis=1).cpu().numpy()

            preds_A = torch.argmax(self.output_semantic_t1, dim=1).cpu().numpy()
            preds_B = torch.argmax(self.output_semantic_t2, dim=1).cpu().numpy()

            preds_scd = (preds_A - 1) * 6 + preds_B
            preds_scd[change_mask == 0] = 0

            labels_scd = (labels_A - 1) * 6 + labels_B
            labels_scd[labels_cd == 0] = 0

            for (pred_scd, label_scd) in zip(preds_scd, labels_scd):
                acc_A, valid_sum_A = accuracy(pred_scd, label_scd)
                self.preds_all.append(pred_scd)
                self.labels_all.append(label_scd)
                self.acc = acc_A
                self.acc_meter.update(self.acc)


    #def eval_models(self, checkpoint_name='best_ckpt.pt'):
    def eval_models(self, checkpoint_name='last_ckpt.pt'):
        self._load_checkpoint(checkpoint_name)

        ################## Eval ##################
        ##########################################
        self.logger.write('Begin evaluation...\n')
        self._clear_cache()
        self.is_training = False
        self.net_G.eval()

        # Iterate over data.
        for self.batch_id, batch in enumerate(self.dataloader, 0):
            with torch.no_grad():
                self._forward_pass(batch)
        kappa_n0, Fscd, IoU_mean, Sek = SCDD_eval_all(self.preds_all, self.labels_all, 37)
        # print(f'Kappa coefficient rate is {kappa_n0}, F1 is {Fscd}, OA is {acc_meter.avg}, '
        #       f'mIoU is {IoU_mean}, SeK is {Sek}')
        print(f'Kappa coefficient rate is {kappa_n0}, F1 is {Fscd},'
              f'mIoU is {IoU_mean}, SeK is {Sek}')
        #return kappa_n0, Fscd, IoU_mean, Sek, self.acc_meter.avg

            #self._collect_running_batch_states()
        #self._collect_epoch_states()