import os
import numpy as np

import torch
import torch.nn as nn
import pandas as pd
## 네트워크 저장하기
def save(ckpt_dir, net, optim, epoch, fold_num, KF = False):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    if KF == False:

        torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
                   "%s/model_epoch%d.pth" % (ckpt_dir, epoch))
    else:
        torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
                   "%s/model_fold%d_current.pth" % (ckpt_dir,fold_num))


## 네트워크 불러오기
def load(ckpt_dir, net, optim, fold, KF = False):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch

    ckpt_lst = os.listdir(ckpt_dir)
    print(ckpt_lst)
    #print(int(''.join(filter(str.isdigit, f))))
    #ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    if KF == False:

        dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]))
        epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])
        net.load_state_dict(dict_model['net'])
        optim.load_state_dict(dict_model['optim'])
        return net, optim, epoch
    else:

        ckpt_lst2 = [i for i in ckpt_lst if str(fold) in i]

        ckpt_lst3 = [i for i in ckpt_lst2 if 'current' in i][0]

        dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst3))

        net.load_state_dict(dict_model['net'])
        optim.load_state_dict(dict_model['optim'])
        return net, optim

def best_load(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch

    ckpt_lst = os.listdir(ckpt_dir)
    print(ckpt_dir)
    best_model_name = [j for j in os.listdir(ckpt_dir) if 'best' in j][0]


    dict_model = torch.load('%s/%s' % (ckpt_dir, best_model_name ))

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(best_model_name.split('epoch')[1].split('.pth')[0])

    return net, optim, epoch


class EarlyStopping:
    """주어진 patience 이후로 validation loss가 개선되지 않으면 학습을 조기 중지"""
    def __init__(self, patience=10, verbose=False, delta=0):
        """
        Args:
            patience (int): validation loss가 개선된 후 기다리는 기간
                            Default: 7
            verbose (bool): True일 경우 각 validation loss의 개선 사항 메세지 출력
                            Default: False
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
                            Default: 0
            path (str): checkpoint저장 경로
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta


    def __call__(self, ckpt_dir,result_dir,results,val_loss, net,optim,fold, epoch,kf ):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(ckpt_dir,val_loss, net,optim, fold, epoch,kf )
            self.save_results(result_dir,results, fold, kf)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(ckpt_dir,val_loss, net,optim,fold, epoch,kf )
            self.save_results(result_dir,results, fold, kf)
            self.counter = 0

    def save_checkpoint(self, ckpt_dir,val_loss, net,optim, fold, epoch,kf):
        '''validation loss가 감소하면 모델을 저장한다.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if kf == True:
            torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},os.path.join(ckpt_dir, 'best_fold'+str(fold)+'_checkpoint.pth'))
        else:
            torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},os.path.join(ckpt_dir, 'best_checkpoint.pth'))

        self.val_loss_min = val_loss

    def save_results(self, result_dir,results, fold, kf):
        '''validation loss가 감소하면 Results을 저장한다.'''
        if kf == True:
            results.to_csv(os.path.join(result_dir,'fold'+str(fold)+'_best_result.csv'))
        else:
            results.to_csv(os.path.join(result_dir,'best_result.csv'))


class EarlyStopping_CV:
    """주어진 patience 이후로 validation loss가 개선되지 않으면 학습을 조기 중지"""
    def __init__(self, patience=10, verbose=False, delta=0):
        """
        Args:
            patience (int): validation loss가 개선된 후 기다리는 기간
                            Default: 7
            verbose (bool): True일 경우 각 validation loss의 개선 사항 메세지 출력
                            Default: False
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
                            Default: 0
            path (str): checkpoint저장 경로
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta



    def __call__(self, ckpt_dir,result_dir,results,test_results, val_loss, net, optim, fold, epoch,kf ):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(ckpt_dir,val_loss, net,optim, fold, epoch,kf )
            self.save_results(result_dir,results,test_results,   fold, kf)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(ckpt_dir,val_loss, net,optim, fold, epoch,kf )
            self.save_results(result_dir,results,test_results,  fold, kf)
            self.counter = 0

    def save_checkpoint(self, ckpt_dir,val_loss, net ,optim,  fold, epoch,kf):
        '''validation loss가 감소하면 모델을 저장한다.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if kf == True:
            torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
                       os.path.join(ckpt_dir, 'best_fold' + str(fold) + '_checkpoint.pth'))
        else:
            torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
                           os.path.join(ckpt_dir, 'best_checkpoint.pth'))


        self.val_loss_min = val_loss

    def save_results(self, result_dir,results,test_results, fold, kf):
        '''validation loss가 감소하면 Results을 저장한다.'''
        if kf == True:

            results.to_csv(os.path.join(result_dir,'val', 'fold' + str(fold) + '_best_result.csv'))
            test_results.to_csv(os.path.join(result_dir, 'test', 'fold' + str(fold) + '_best_result.csv'))

        else:
            results.to_csv(os.path.join(result_dir,'best_result.csv'))


