import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from utils.utils import *
from model.LLMDetector import LLMDetector
from data_factory.data_loader import get_loader_segment
from einops import rearrange
from metrics.metrics import *
import warnings
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
warnings.filterwarnings('ignore')
from accelerate.utils import set_seed

def my_loss_function(p, q, input, dis='cos'):

    loss=nn.MSELoss(reduction='none')
    res=loss(p,q)
    # print(torch.sum(res,dim=-1))
    input_sim=torch.zeros((input.shape[0],input.shape[0]))
    input=rearrange(input,'b l c -> b (l c)')
    #--------instance-wise-----------------------
    if dis=='cos':
        input_sim=torch.einsum('ij,jk->ik',input,input.transpose(0,1))/torch.einsum('i,j->ij',torch.norm(input,dim=-1),torch.norm(input,dim=-1))
    input_sim=torch.clamp(input_sim,min=0)
    input_logits=torch.tril(input_sim,diagonal=-1)[:, :-1]
    input_logits+=torch.triu(input_sim,diagonal=1)[:,  1:]

    p_ins=rearrange(p,'b l c -> b (l c)')
    q_ins=rearrange(q,'b l c -> b (l c)')
    p_sim=torch.matmul(p_ins,p_ins.transpose(0,1))/torch.einsum('i,j->ij',torch.norm(p_ins,dim=-1),torch.norm(p_ins,dim=-1))
    q_sim=torch.matmul(q_ins,q_ins.transpose(0,1))/torch.einsum('i,j->ij',torch.norm(q_ins,dim=-1),torch.norm(q_ins,dim=-1))
    p_logits=torch.tril(p_sim,diagonal=-1)[:, :-1]
    p_logits+=torch.triu(p_sim,diagonal=1)[:,  1:]
    q_logits=torch.tril(q_sim,diagonal=-1)[:, :-1]
    q_logits+=torch.triu(q_sim,diagonal=1)[:,  1:]

    p_logits=-F.log_softmax(p_logits,dim=-1)
    q_logits=-F.log_softmax(q_logits,dim=-1)
    ins_loss=torch.mean(p_logits*input_logits)
    ins_loss+=torch.mean(q_logits*input_logits)

    return torch.mean(res,dim=-1),ins_loss




class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):
        kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        
        self.__dict__.update(Solver.DEFAULTS, **config)
        set_seed(self.seed)
        self.accelerator = Accelerator(kwargs_handlers=[kwargs])
        if self.mode == 'train':
            self.accelerator.print("\n\n")
            self.accelerator.print('================ Hyperparameters ===============')
            for k, v in sorted(config.items()):
                self.accelerator.print('%s: %s' % (str(k), str(v)))
            self.accelerator.print('====================  Train  ===================')

        # print("Loading dataset")
        self.train_loader = get_loader_segment(self.index, 'datasets/'+self.data_path, batch_size=self.batch_size, win_size=self.win_size, mode='train', dataset=self.dataset,step=self.step )
        # print("Train dataset loaded.")
        self.vali_loader = get_loader_segment(self.index, 'datasets/'+self.data_path, batch_size=self.batch_size, win_size=self.win_size, mode='val', dataset=self.dataset,step=self.step)
        self.test_loader = get_loader_segment(self.index, 'datasets/'+self.data_path, batch_size=self.batch_size, win_size=self.win_size, mode='test', dataset=self.dataset,step=self.step)

        self.device = torch.device(f"cuda:{self.gpu}" if torch.cuda.is_available() else "cpu")
        

        self.accelerator.print('Data loaded')
        try:
            with open(os.path.join('datasets/'+self.data_path, 'description.txt'), 'r') as f:
                self.des = f.read()
        except:
            self.des=''
        self.build_model()


    def build_model(self):
        self.model = LLMDetector(win_size=self.win_size, enc_in=self.input_c,  n_heads=self.n_heads, llm_model=self.llm_model,\
                                d_model=self.d_model, patch_size=self.patch_size, channel=self.input_c,description=self.des)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        
    def vali(self, vali_loader):
        self.model.eval()
        loss_1 = []
        vali_loader=self.accelerator.prepare(vali_loader)
        for i, (input_data, _) in enumerate(vali_loader):
            input=input_data.float()
            series, prior = self.model(input)
            series_loss,series_ins_loss  = my_loss_function(series, prior,input)
            prior_loss,prior_ins_loss = my_loss_function(prior, series,input)
            loss=torch.mean(prior_loss + series_loss)-(series_ins_loss+prior_ins_loss)
            all_losses=self.accelerator.gather(loss)
            batch_loss=0
            if all_losses.shape:
                for l in all_losses:
                    batch_loss+=l.item()
                loss_1.append(batch_loss)
            else:
                loss_1.append(all_losses.item())

        return np.average(loss_1)


    def train(self,setting):
        time_now = time.time()
        path = 'checkpoints/' + setting + '/'
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(accelerator=self.accelerator,patience=self.patience, verbose=True, dataset_name=self.data_path)
        train_steps = len(self.train_loader)
        self.model, self.optimizer, self.train_loader=self.accelerator.prepare(self.model,self.optimizer,self.train_loader)
        for epoch in range(self.num_epochs):
            iter_count = 0
            epoch_time = time.time()
            self.model.train()
            for i, (input_data, labels) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                iter_count += 1
                input=input_data.float()
                series, prior = self.model(input)
                series_loss,series_ins_loss = my_loss_function(series, prior.detach(),input)
                prior_loss, prior_ins_loss= my_loss_function(prior, series.detach(),input)
                if self.use_insloss and self.use_reploss:
                    loss = (1-self.alpha)*torch.mean(prior_loss + series_loss)-self.alpha*(series_ins_loss+prior_ins_loss)
                elif self.use_reploss:
                    loss=torch.mean(prior_loss + series_loss)
                elif self.use_insloss:
                    loss=-(series_ins_loss+prior_ins_loss)
                else:
                    assert('ths loss is none.')
                
                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    self.accelerator.print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
 
                self.accelerator.backward(loss)
                self.optimizer.step()
            with torch.no_grad():
                vali_loss= self.vali(self.test_loader)

            self.accelerator.print(
                "Epoch: {0}, Cost time: {1:.3f}s ".format(
                    epoch + 1, time.time() - epoch_time))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

            
    def test(self,setting):
        with torch.no_grad():
            path = 'checkpoints/' + setting + '/'
            self.build_model()
            self.model.load_state_dict(
                torch.load(
                    path+self.data_path+'_model.pth'))
            self.model.eval()

            test_labels = []
            attens_energy = []
            self.test_loader,self.model=self.accelerator.prepare(self.test_loader,self.model)
            for i, (input_data, labels) in enumerate(self.test_loader):
                input = input_data.float()
                series, prior = self.model(input)
                series_loss,_=my_loss_function(series, prior, input)
                prior_loss,_=my_loss_function(prior, series, input)
                loss = -series_loss - prior_loss
                
                mean=torch.mean(loss,dim=-1).unsqueeze(1)
                std=torch.std(loss,dim=-1).unsqueeze(1)
                metric=(loss-mean)/std
                attens_energy.append(metric)
                test_labels.append(labels)

            attens_energy=self.accelerator.gather(attens_energy)
            test_labels=self.accelerator.gather(test_labels)
            
            
            attens_energy=torch.stack(attens_energy).detach().cpu().numpy()
            test_labels=torch.stack(test_labels).detach().cpu().numpy()
            
            attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
            test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
            
            test_energy = np.array(attens_energy)
            test_labels = np.array(test_labels)

            if self.accelerator.is_main_process:
                np.save(path + self.dataset+ 'test_energy.npy',test_energy)
                np.save(path+self.dataset+'test_labels.npy',test_labels)

    def test1(self,setting):
        path= 'checkpoints/' + setting + '/'
        test_energy=np.load(path + self.dataset+'test_energy.npy', )
        print(test_energy.shape,self.train_loader.dataset.test.shape)
        test_labels=np.load(path+ self.dataset+'test_labels.npy')
        thresh = np.percentile(test_energy, (1 - self.anormly_ratio)*100)
        print('Anomaly Ratio:', self.anormly_ratio, (1 - self.anormly_ratio)*100)
        print("Threshold :", thresh)
        pred = (test_energy > thresh).astype(int)
        gt = test_labels.astype(int)

        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1

        pred = np.array(pred)
        gt = np.array(gt)

        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        print("Accuracy:{:0.4f}, Precision:{:0.4f}, Recall:{:0.4f}, F-score:{:0.4f} ".format(accuracy, precision, recall, f_score))
        result_path='result/'+setting+'_result.txt'
        with open(result_path, 'a') as f:
            f.write("Thresh:{:0.4f}, Anomaly Ratio:{:0.4f}. ".format(thresh,self.anormly_ratio))
            f.write("Dataset:{}, Accuracy:{:0.4f}, Precision:{:0.4f}, Recall:{:0.4f}, F-score:{:0.4f}.".format(self.dataset, accuracy, precision, recall, f_score))
            f.write('\n')


class EarlyStopping:
    def __init__(self, accelerator=None, patience=3, dataset_name='',verbose=False, delta=0, save_mode=True):
        self.accelerator = accelerator
        self.patience = patience
        self.dataset=dataset_name
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_mode = save_mode

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.accelerator is None:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            else:
                self.accelerator.print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            if self.accelerator is not None:
                self.accelerator.print(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            else:
                print(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        if self.accelerator is not None:
            model = self.accelerator.unwrap_model(model)
            torch.save(model.state_dict(), path + '/' + self.dataset+'_model.pth')
        else:
            self.accelerator.save(model.state_dict(), path + '/' + self.dataset+'_model.pth')
        self.val_loss_min = val_loss

def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr