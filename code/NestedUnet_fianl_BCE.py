import torch

print ('Number of available devices ', torch.cuda.device_count())
# GPU 할당 변경하기
gpu_ids = 2# 원하는 GPU 번호 입력
device = torch.device(f'cuda:{gpu_ids}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)
print ('Current devices ', torch.cuda.current_device())
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
from loss_function import *
from model import *
from dataload import *
from utils import *
from metrics import *
import pandas as pd
# I expect to see RuntimeWarnings in this block

## 트레이닝 파라메터 설정하기
lr = 1e-4
batch_size = 8
num_epoch = 100

data_dir = './datasets/segmentation_final6'
ckpt_dir = './NestedUnet_seg_final_BCE/checkpoint'
log_dir = './NestedUnet_seg_final_BCE/log'
result_dir = './NestedUnet_seg_final_BCE/result'
mode = 'train'
train_continue = 'off'
kf_type = True

#device = 'cpu'
print("device: ",device)
print("learning rate: %.4e" % lr)
print("batch size: %d" % batch_size)
print("number of epoch: %d" % num_epoch)
print("data dir: %s" % data_dir)
print("ckpt dir: %s" % ckpt_dir)
print("log dir: %s" % log_dir)
print("result dir: %s" % result_dir)
print("mode: %s" % mode)
print("GPU ID : %s" % gpu_ids)
print("K-fold cross validation : %s" % str(kf_type))


testset = os.listdir(os.path.join(data_dir, 'test'))
testset = [i for i in testset if 'label' in i]


## 디렉토리 생성하기
if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir, 'png'))
    os.makedirs(os.path.join(result_dir, 'numpy'))

## 그밖에 부수적인 functions 설정하기
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean
fn_class = lambda x: 1.0 * (x > 0.5)

## Tensorboard 를 사용하기 위한 SummaryWriter 설정
writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))
# early_stopping object의 초기화

## 네트워크 학습시키기
st_epoch = 0
deepsuper = False
# TRAIN MODE
from sklearn.model_selection import KFold
k_folds = 5
KF_TYPE = True

train_continue = "off"
# TEST MODE
# TEST MODE
def test(ckpt_dir, net, optim, fold, epoch, data_name, mode):
    kf_type = True
    mode_name = mode
    net, optim= load(ckpt_dir=ckpt_dir, net=net, optim=optim, fold = fold , KF= kf_type )
    fn_loss = nn.BCELoss().to(device)################################################################################################################################################

    if mode == 'test':
        transform = transforms.Compose([Normalization(), ToTensor()])
        dataset_test = Dataset(data_dir=os.path.join(data_dir, 'test'), transform=transform)
        loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)

        # 그밖에 부수적인 variables 설정하기
        num_data_test = len(dataset_test)
        num_batch_test = np.ceil(num_data_test / batch_size)

    elif mode == 'external':
        transform = transforms.Compose([Normalization(), ToTensor()])
        dataset_test = Dataset(data_dir=os.path.join(data_dir, 'external'), transform=transform)
        loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)

        # 그밖에 부수적인 variables 설정하기
        num_data_test = len(dataset_test)
        num_batch_test = np.ceil(num_data_test / batch_size)

    with torch.no_grad():
        net.eval()
        loss_arr = []
        dice_arr = []
        iou_arr = []
        count = 0

        for batch, data in enumerate(loader_test, 1):
            # forward pass
            label = data['label'].to(device)
            input = data['input'].to(device)
            output= net(input)


            loss= fn_loss(output, label)


            # Tensorboard 저장하기
            label = fn_tonumpy(label)
            input = fn_tonumpy(input)
            output = fn_tonumpy(fn_class(output))
            Dice, IoU = metrics(output, label)
            loss_arr += [loss.item()]
            dice_arr += Dice
            iou_arr += IoU

            print(mode_name + ": BATCH %04d / %04d | LOSS %.4f" %
                  (batch, num_batch_test, np.mean(loss_arr)))

    print("AVERAGE TEST: BATCH %04d / %04d | LOSS %.4f" %
          (batch, num_batch_test, np.mean(loss_arr)))
    results = pd.DataFrame(
        {'Dice': dice_arr, 'IoU': iou_arr})
    return results


def save_best_npy(FOLD):
    transform_val = transforms.Compose([Normalization(), ToTensor()])
    dataset_val = Dataset(data_dir=os.path.join(data_dir, 'fold' + str(FOLD), 'val'), transform=transform_val)

    transform = transforms.Compose([Normalization(), ToTensor()])
    dataset_test = Dataset(data_dir=os.path.join(data_dir,'test'), transform=transform_val)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)


    # 그밖에 부수적인 variables 설정하기
    num_data_test = len(dataset_test)
    num_batch_test = np.ceil(num_data_test / batch_size)

    transform = transforms.Compose([Normalization(), ToTensor()])
    dataset_external = Dataset(data_dir=os.path.join(data_dir, 'external'), transform=transform)
    loader_external = DataLoader(dataset_external, batch_size=batch_size, shuffle=False, num_workers=0)

    # 그밖에 부수적인 variables 설정하기
    num_data_external = len(dataset_external)
    num_batch_external = np.ceil(num_data_external/ batch_size)

    ## 네트워크 생성하기
    net =NestedUNet().to(device)################################################################################################################################################
    ## Optimizer 설정하기
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=0)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)
    loader_external = DataLoader(dataset_external, batch_size=batch_size, shuffle=False, num_workers=0)

    ckpt_lst = os.listdir(ckpt_dir)
    best_model_name = [j for j in os.listdir(ckpt_dir) if 'best_fold' + str(FOLD) in j][0]

    dict_model = torch.load('%s/%s' % (ckpt_dir, best_model_name))

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])

    with torch.no_grad():
        net.eval()
        id = 0
        count = 0
        val_name = os.listdir(os.path.join(data_dir, 'fold' + str(FOLD), 'val'))
        test_name = os.listdir(os.path.join(data_dir, 'test'))
        external_name = os.listdir(os.path.join(data_dir, 'external'))

        for batch, data in enumerate(loader_val, 1):

            # forward pass
            print(data['Input_ID'])
            print(data['Label_ID'])


            label = data['label'].to(device)
            input = data['input'].to(device)
            output = net(input)

            # Tensorboard 저장하기
            label = fn_tonumpy(label)
            input = fn_tonumpy(input)
            output = fn_tonumpy(fn_class(output))

            for j in range(label.shape[0]):

                id = data['Input_ID'][j].split('_')[5]
                count += 1

                if len(str(count)) == 1:
                    str_count = '00' + str(count)
                elif len(str(count)) == 2:
                    str_count = '0' + str(count)
                elif len(str(count)) == 3:
                    str_count = str(count)

                # print(id)
                np.save(os.path.join(result_dir, 'val', 'numpy',
                                     'val_fold' + str(FOLD) + '_' + str_count + '_label_' + id),
                        label[j].squeeze())
                np.save(os.path.join(result_dir, 'val', 'numpy',
                                     'val_fold' + str(FOLD) + '_' + str_count + '_input_' + id),
                        input[j].squeeze())
                np.save(os.path.join(result_dir, 'val', 'numpy',
                                     'val_fold' + str(FOLD) + '_' + str_count + '_output_' + id),
                        output[j].squeeze())

    #######################################################
    with torch.no_grad():
        net.eval()
        id = 0
        count = 0

        for batch, data in enumerate(loader_test, 1):
            print(data['Input_ID'])
            print(data['Label_ID'])


            # forward pass
            label = data['label'].to(device)
            input = data['input'].to(device)
            output = net(input)

            # Tensorboard 저장하기
            label = fn_tonumpy(label)
            input = fn_tonumpy(input)
            output = fn_tonumpy(fn_class(output))

            for j in range(label.shape[0]):
                id = data['Input_ID'][j].split('_')[5]
                count += 1

                if len(str(count)) == 1:
                    str_count = '00' + str(count)
                elif len(str(count)) == 2:
                    str_count = '0' + str(count)
                elif len(str(count)) == 3:
                    str_count = str(count)

                # print(id)
                np.save(os.path.join(result_dir, 'test', 'numpy',
                                     'test_fold' + str(FOLD) + '_' + str_count + '_label_' + id),
                        label[j].squeeze())
                np.save(os.path.join(result_dir, 'test', 'numpy',
                                     'test_fold' + str(FOLD) + '_' + str_count + '_input_' + id),
                        input[j].squeeze())
                np.save(os.path.join(result_dir, 'test', 'numpy',
                                     'test_fold' + str(FOLD) + '_' + str_count + '_output_' + id),
                        output[j].squeeze())

    with torch.no_grad():
        net.eval()
        id = 0
        count = 0

        for batch, data in enumerate(loader_external, 1):
            print(data['Input_ID'])
            print(data['Label_ID'])


            label = data['label'].to(device)
            input = data['input'].to(device)
            output = net(input)

            # Tensorboard 저장하기
            label = fn_tonumpy(label)
            input = fn_tonumpy(input)
            output = fn_tonumpy(fn_class(output))

            for j in range(label.shape[0]):
                id = data['Input_ID'][j].split('_')[2]
                count += 1

                if len(str(count)) == 1:
                    str_count = '00' + str(count)
                elif len(str(count)) == 2:
                    str_count = '0' + str(count)
                elif len(str(count)) == 3:
                    str_count = str(count)

                # print(id)
                np.save(os.path.join(result_dir, 'external', 'numpy',
                                     'external_fold' + str(FOLD) + '_' + str_count + '_label_' + id),
                        label[j].squeeze())
                np.save(os.path.join(result_dir, 'external', 'numpy',
                                     'external_fold' + str(FOLD) + '_' + str_count + '_input_' + id),
                        input[j].squeeze())
                np.save(os.path.join(result_dir, 'external', 'numpy',
                                     'external_fold' + str(FOLD) + '_' + str_count + '_output_' + id),
                        output[j].squeeze())


if mode == 'train':
    valid_loss_list = []
    train_loss_list = []
    kfold = KFold(n_splits=k_folds, shuffle=False)
    #transform_train = transforms.Compose([ transforms.ToPILImage(),transforms.RandomAffine(degrees=(0, 0), translate=(0.25, 0.25), scale=(0.5, 1.5)),
    # transforms.RandomRotation(degrees=[-22.5, 22.5]),transforms.RandomVerticalFlip(p=0.5), transforms.RandomHorizontalFlip(p=0.5),Normalization(), ToTensor()])
    transform_train =transforms.Compose([Normalization(), ToTensor()])

    transform_val = transforms.Compose([Normalization(), ToTensor()])

    print('-------------')
    #print(dir(dataset_train))
    print('-------------')
    loss_result_list = pd.DataFrame()
    for fold in range(2,5):

        dataset_train = Dataset(data_dir=os.path.join(data_dir, 'fold'+str(fold+1),'train'), transform=transform_train)
        dataset_val = Dataset(data_dir=os.path.join(data_dir, 'fold'+str(fold+1),'val'), transform=transform_val)
        val_id = os.listdir(os.path.join(data_dir, 'fold'+str(fold+1),'val'))

        best_epoch = 0
        FOLD = fold + 1

        ## 네트워크 생성하기
        net = NestedUNet().to(device)################################################################################################################################################
        ## WEIGHT INitialization
        init_net(net, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)
        ## 손실함수 정의하기
        fn_loss = nn.BCELoss().to(device)################################################################################################################################################
        ## Optimizer 설정하기
        optim = torch.optim.Adam(net.parameters(), lr=lr)
        ## Ealry stopping
        ES_patience = 20
        # early_stopping = EarlyStopping_CV(patience=ES_patience, verbose=True)
        early_stopping = EarlyStopping_CV(patience=ES_patience, verbose=True)

        valid_loss_list = []

        ## Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', patience=5,factor=0.5,min_lr=1e-6, verbose=True)
        #train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)

        print(val_id)

        #test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False,  num_workers=0)
        loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False,  num_workers=0)

        # 그밖에 부수적인 variables 설정하기
        num_data_train = len(dataset_train)
        num_data_val = len(dataset_val)
        num_batch_train = np.ceil(num_data_train / batch_size)
        num_batch_val = np.ceil(num_data_val / batch_size)
        if train_continue == "on":
            net, optim= load(ckpt_dir=ckpt_dir, net=net, optim=optim, fold = FOLD , KF= KF_TYPE)

        for epoch in range(st_epoch + 1, num_epoch + 1):
            count = 0
            net.train()
            loss_arr = []
            for batch, data in enumerate(loader_train, 1):
                # forward pass
                label = data['label'].to(device)
                input = data['input'].to(device)
                output = net(input)

                # backward pass
                optim.zero_grad()
                loss = fn_loss( output, label)
                loss.backward()
                optim.step()
                # 손실함수 계산
                loss_arr += [loss.item()]

                print("TRAIN: K-Fold %d | EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                      (fold+1, epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))

                # Tensorboard 저장하기
                label = fn_tonumpy(label)
                input = fn_tonumpy(input)
                output = fn_tonumpy(fn_class(output))

            writer_train.add_scalar('loss', np.mean(loss_arr), epoch)


            with torch.no_grad():
                net.eval()
                loss_arr = []
                valid_loss = []
                dice_arr = []
                iou_arr = []
                results = []

                id = 0

                for batch, data in enumerate(loader_val, 1):
                    # forward pass
                    label = data['label'].to(device)
                    input = data['input'].to(device)
                    output = net(input)

                    # 손실함수 계산하기
                    loss = fn_loss(output, label)
                        # Tensorboard 저장하기
                    label = fn_tonumpy(label)
                    input = fn_tonumpy(input)
                    output = fn_tonumpy(fn_class(output))
                    Dice, IoU = metrics(output, label)
                    loss_arr += [loss.item()]
                    dice_arr += Dice
                    iou_arr += IoU

                    print("VALID:Fold %d EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                          (fold+1, epoch, num_epoch, batch, num_batch_val, np.mean(loss_arr)))



            scheduler.step(np.mean(loss_arr))
            print('Epoch-{0} lr: {1}'.format(epoch, optim.param_groups[0]['lr']))
            valid_loss = np.mean(loss_arr)
            valid_loss_list += [valid_loss]



            results = pd.DataFrame(
                {'Dice': dice_arr, 'IoU': iou_arr})
            results.to_csv(os.path.join(result_dir ,'val', 'epoch'+str(epoch)+'_result.csv'))

            print("Validation Loss :",valid_loss)
            print("Dice", np.mean(dice_arr))
            print("IoU", np.mean(iou_arr))
            writer_val.add_scalar('loss', valid_loss, epoch)
            if epoch % 1 == 0:
                save(ckpt_dir=ckpt_dir, net=net, optim=optim,  epoch=epoch, fold_num =FOLD , KF = KF_TYPE)
            # early_stopping는 validation loss가 감소하였는지 확인이 필요하며,
            # 만약 감소하였을경우 현제 모델을 checkpoint로 만든다.

            #test_result =  test(ckpt_dir, net, optim, FOLD , epoch,testset, 'test')
            #external_result =  test(ckpt_dir, net, optim, FOLD , epoch,externalset,'external')
            #early_stopping(ckpt_dir, os.path.join(result_dir), results,test_result, external_result,  valid_loss, net,optim, FOLD , epoch, kf=KF_TYPE)
            test_result =  test(ckpt_dir, net, optim, FOLD , epoch,testset, 'test')

            early_stopping(ckpt_dir, os.path.join(result_dir), results,test_result,   valid_loss, net,optim, FOLD , epoch, kf=KF_TYPE)

                # break

                ### EACH FOLD AND EPOCH
            loss_result = []
            loss_result = pd.DataFrame(data={'Fold': FOLD, 'Epoch': epoch, 'loss': valid_loss, 'LR': optim.param_groups[0]['lr'],
                                   'best epoch':best_epoch},index =[0])
            loss_result_list = loss_result_list.append(loss_result)
            loss_result_list.to_csv(os.path.join(result_dir, 'val', str(FOLD)+ 'CV_loss.csv'))
            if early_stopping.early_stop:
                #print("Early stopping")
                print("best epoch")
                best_epoch = epoch - ES_patience
                break
        save_best_npy(FOLD)






    writer_train.close()
    writer_val.close()


def cal_numeric(data, threshold):
    e = 10 ** (-5)
    #print(data)
    #print(sum(data == 0 ))
    TP = sum(data >= threshold)

    FP_TP = sum(((data) > 0))

    FN = sum(data == 0)
    FP = FP_TP - FN
    #print(FP_TP )
    #print(len(data))
    print(TP)
    print(len(data))

    Recall = round(TP/(len(data)),3)
    Precision =round(TP/(FP_TP+e),3)
    F1 =round((2*TP)/((2*TP)+FP+FN),3)
    print(Recall)
    print(Precision)

    return Recall, Precision, F1


def cal_indi_metric(df):

    Dice_indi_mean = round(np.mean(df['Dice']), 3)
    Dice_indi_std = round(np.std(df['Dice']), 3)
    Dice_indi_median = round(np.median(df['Dice']), 3)
    Dice_indi_mad = np.round(np.median(np.abs(df['Dice'] - Dice_indi_median)), 3)

    IoU_indi_mean = round(np.mean(df['IoU']), 3)
    IoU_indi_std = round(np.std(df['IoU']), 3)
    IoU_indi_median = round(np.median(df['IoU']), 3)
    IoU_indi_mad = np.round(np.median(np.abs(df['IoU'] - IoU_indi_median)), 3)

    Recall50,Precision50, F1_50 = cal_numeric(df['IoU'], 0.5)
    Recall75,Precision75, F1_75 = cal_numeric(df['IoU'], 0.75)
    Recall90,Precision90, F1_90 = cal_numeric(df['IoU'], 0.90)

    fold_result = pd.DataFrame(data = {'Dice score median':Dice_indi_median, 'Dice score mad':Dice_indi_mad,\
                         'Dice score mean':Dice_indi_mean,'Dice score std':Dice_indi_std, \
                         'IoU score median': IoU_indi_median, 'IoU score mad': IoU_indi_mad, \
                         'IoU score mean': IoU_indi_mean, 'IoU score std': IoU_indi_std, \
                         'Recall 50' : Recall50, 'Precision 50': Precision50, 'F1 50':F1_50, \
                         'Recall 75': Recall75, 'Precision 75': Precision75, 'F1 75': F1_75, \
                         'Recall 90': Recall90, 'Precision 90': Precision90, 'F1 90': F1_90
                         }, index=[0])
    return fold_result

result_val_dir = os.path.join(result_dir, 'val')
result_test_dir = os.path.join(result_dir, 'test')
result_external_dir =os.path.join(result_dir, 'external')


fold_result_total = []
k_folds = 5
def calculation(dir):
    for j in range(1,k_folds+1):
        fold_result =  pd.read_csv(os.path.join(dir, 'fold'+str(j)+'_best_result.csv'))
        fold_csv = cal_indi_metric(fold_result)
        fold_csv.to_csv(os.path.join(dir, 'fold'+str(j)+'_best_result_summary.csv'))
        if j ==1:
            fold_result_total = fold_result
        else:
            fold_result_total = fold_result_total.append(fold_result)

    print(fold_result_total)

    fold_csv = cal_indi_metric(fold_result_total)
    fold_csv.to_csv(os.path.join(dir, 'Total_best_result_summary.csv'))


calculation(result_val_dir)
calculation(result_test_dir)

