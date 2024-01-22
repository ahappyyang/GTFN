from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import numpy as np
import scipy.io as sio
import torch
import math


def chooose_train_and_test_point(train_data, test_data, true_data, num_classes):
    number_train = []
    pos_train = {}
    number_test = []
    pos_test = {}
    number_true = []
    pos_true = {}
    #-------------------------for train data------------------------------------
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(train_data==(i+1))
        number_train.append(each_class.shape[0])
        pos_train[i] = each_class

    total_pos_train = pos_train[0]
    for i in range(1, num_classes):
        total_pos_train = np.r_[total_pos_train, pos_train[i]] #(695,2)
    total_pos_train = total_pos_train.astype(int)
    #--------------------------for test data------------------------------------
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(test_data==(i+1))
        number_test.append(each_class.shape[0])
        pos_test[i] = each_class

    total_pos_test = pos_test[0]
    for i in range(1, num_classes):
        total_pos_test = np.r_[total_pos_test, pos_test[i]] #(9671,2)
    total_pos_test = total_pos_test.astype(int)
    #--------------------------for true data------------------------------------
    for i in range(num_classes+1):
        each_class = []
        each_class = np.argwhere(true_data==i)
        number_true.append(each_class.shape[0])
        pos_true[i] = each_class

    total_pos_true = pos_true[0]
    for i in range(1, num_classes+1):
        total_pos_true = np.r_[total_pos_true, pos_true[i]]
    total_pos_true = total_pos_true.astype(int)

    return total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true
#-------------------------------------------------------------------------------
# 边界拓展：镜像
def mirror_hsi(height,width,band,input_normalize,patch=5):
    padding=patch//2
    mirror_hsi=np.zeros((height+2*padding,width+2*padding,band),dtype=float)
    #中心区域
    mirror_hsi[padding:(padding+height),padding:(padding+width),:]=input_normalize
    #左边镜像
    for i in range(padding):
        mirror_hsi[padding:(height+padding),i,:]=input_normalize[:,padding-i-1,:]
    #右边镜像
    for i in range(padding):
        mirror_hsi[padding:(height+padding),width+padding+i,:]=input_normalize[:,width-1-i,:]
    #上边镜像
    for i in range(padding):
        mirror_hsi[i,:,:]=mirror_hsi[padding*2-i-1,:,:]
    #下边镜像
    for i in range(padding):
        mirror_hsi[height+padding+i,:,:]=mirror_hsi[height+padding-1-i,:,:]

    print("**************************************************")
    print("patch is : {}".format(patch))
    print("mirror_image shape : [{0},{1},{2}]".format(mirror_hsi.shape[0],mirror_hsi.shape[1],mirror_hsi.shape[2]))
    print("**************************************************")
    return mirror_hsi
#-------------------------------------------------------------------------------
# 排序取索引
def choose_top(image,cornor_index,x,y,patch,b,n_top):
    sort = image.reshape(patch * patch, b)
    sort = torch.from_numpy(sort).type(torch.FloatTensor)
    pos = (x - cornor_index[0]) * patch + (y - cornor_index[1])
    Q = torch.sum(torch.pow(sort[pos] - sort, 2), dim=1)
    _, indices = Q.topk(k=n_top, dim=0, largest=False, sorted=True)
    return indices
#-------------------------------------------------------------------------------
# 获取patch的图像数据
def gain_neighborhood_pixel(pca_image, point, i, patch, W, H,n_gcn):
    x = point[i,0]
    y = point[i,1]
    m=int((patch-1)/2)##patch奇数
    _,_,b=pca_image.shape
    if x<=m:
        if y<=m:
            temp_image = pca_image[0:patch, 0:patch, :]
            cornor_index = [0,0]
        if y>=(H-m):
            temp_image = pca_image[0:patch, H-patch:H, :]
            cornor_index = [0, H-patch]
        if y>m and y<H-m:
            temp_image = pca_image[0:patch, y-m:y+m+1, :]
            cornor_index = [0, y-m]
    if x>=(W-m):
        if y<=m:
            temp_image = pca_image[W-patch:W, 0:patch, :]
            cornor_index = [W-patch, 0]
        if y>=(H-m):
            temp_image = pca_image[W-patch:W, H-patch:H, :]
            cornor_index = [W - patch, H-patch]
        if y>m and y<H-m:
            temp_image = pca_image[W-patch:W, y-m:y+m+1, :]
            cornor_index = [W - patch, y-m]
    if x>m and x<W-m:
        if y<=m:
            temp_image = pca_image[x-m:x+m+1, 0:patch, :]
            cornor_index = [x-m, 0]
        if y>=(H-m):
            temp_image = pca_image[x-m:x+m+1, H-patch:H, :]
            cornor_index = [x - m, H-patch]
        if y>m and y<H-m:
            temp_image = pca_image[x-m:x+m+1, y-m:y+m+1, :]
            cornor_index = [x - m, y-m]
            # look11=pca_image[:,:,0]
            # look12=temp_image[:,:,0]
            # print(temp_image.shape)
    index = choose_top(temp_image, cornor_index, x, y, patch, b, n_gcn)
    return temp_image,cornor_index,index
# 汇总训练数据和测试数据
def train_and_test_data(pca_image, band, train_point, test_point, true_point, patch, w, h, n_gcn):
    x_train = np.zeros((train_point.shape[0], patch, patch, band), dtype=float)
    x_test = np.zeros((test_point.shape[0], patch, patch, band), dtype=float)
    x_true = np.zeros((true_point.shape[0], patch, patch, band), dtype=float)
    corner_train = np.zeros((train_point.shape[0], 2), dtype=int)
    corner_test = np.zeros((test_point.shape[0], 2), dtype=int)
    corner_true = np.zeros((true_point.shape[0], 2), dtype=int)
    indexs_train = torch.zeros((train_point.shape[0], n_gcn),dtype=int).cuda()
    indexs_test = torch.zeros((test_point.shape[0], n_gcn),dtype=int).cuda()
    indexs_ture = torch.zeros((true_point.shape[0], n_gcn),dtype=int).cuda()
    for i in range(train_point.shape[0]):
        x_train[i,:,:,:],corner_train[i,:],indexs_train[i]= gain_neighborhood_pixel(pca_image, train_point, i, patch, w, h,n_gcn)
    for j in range(test_point.shape[0]):
        x_test[j,:,:,:],corner_test[j,:],indexs_test[j] = gain_neighborhood_pixel(pca_image, test_point, j, patch, w, h,n_gcn)
    for k in range(true_point.shape[0]):
        x_true[k,:,:,:],corner_true[k,:],indexs_ture[k] = gain_neighborhood_pixel(pca_image, true_point, k, patch, w, h,n_gcn)
    print("x_train shape = {}, type = {}".format(x_train.shape,x_train.dtype))
    print("x_test  shape = {}, type = {}".format(x_test.shape,x_test.dtype))
    print("x_true  shape = {}, type = {}".format(x_true.shape,x_test.dtype))
    print("**************************************************")

    return x_train, x_test, x_true,corner_train,corner_test,corner_true,indexs_train,indexs_test,indexs_ture
#-------------------------------------------------------------------------------
# 标签y_train, y_test
def train_and_test_label(number_train, number_test, number_true, num_classes):
    y_train = []
    y_test = []
    y_true = []
    for i in range(num_classes):
        for j in range(number_train[i]):
            y_train.append(i)
        for k in range(number_test[i]):
            y_test.append(i)
    for i in range(num_classes):
        for j in range(number_true[i]):
            y_true.append(i)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_true = np.array(y_true)
    print("y_train: shape = {} ,type = {}".format(y_train.shape,y_train.dtype))
    print("y_test: shape = {} ,type = {}".format(y_test.shape,y_test.dtype))
    print("y_true: shape = {} ,type = {}".format(y_true.shape,y_true.dtype))
    print("**************************************************")
    return y_train, y_test, y_true
#-------------------------------------------------------------------------------
class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt
#-------------------------------------------------------------------------------
def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res, target, pred.squeeze()
#-------------------------------------------------------------------------------
# train model
def train_epoch(gcn_net,tr_net, train_loader, criterion, optimizer,optimizer2,indexs_train):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (A,batch_data, batch_target) in enumerate(train_loader):
        batch_A = A.cuda()
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()

        optimizer.zero_grad()
        optimizer2.zero_grad()
        gcn_pred = gcn_net(batch_data,batch_A,indexs_train)
        batch_pred = tr_net(gcn_pred)
        loss = criterion(batch_pred, batch_target)
        loss.backward()
        optimizer.step()
        optimizer2.step()

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
    return top1.avg, objs.avg, tar, pre
#-------------------------------------------------------------------------------
# validate model
def valid_epoch(gcn_net,tr_net, valid_loader, criterion,indexs_test):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (A,batch_data, batch_target) in enumerate(valid_loader):
        batch_A = A.cuda()
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()

        gcn_pred = gcn_net(batch_data,batch_A,indexs_test)
        batch_pred = tr_net(gcn_pred)
        loss = criterion(batch_pred, batch_target)

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
        
    return tar, pre


#-------------------------------------------------------------------------------
def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA
#-------------------------------------------------------------------------------
def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=np.float64)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA

def applyPCA(X, numComponents):

    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))

    return newX



def GET_A2(temp_image,input2,corner,patches,l,sigma=10,):
    input2=input2.cuda()
    N,h,w,_=temp_image.shape
    B = np.zeros((w * h, w * h), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            m = int(i * w + j)
            for k in range(l):
                for q in range(l):
                    n = int((i + (k - (l - 1) / 2)) * w + (j + (q - (l - 1) / 2)))
                    if 0 <= i + (k - (l - 1) / 2) < h and 0 <= (j + (q - (l - 1) / 2)) < w and m != n:
                        B[m, n] = 1

    index=np.argwhere(B == 1)
    index2 = np.where(B == 1)
    A = np.zeros((N, w * h, w * h), dtype=np.float32)

    for i in range(N):
        C = np.array(B)
        x_l=int(corner[i,0])
        x_r=int(corner[i,0]+patches)
        y_l=int(corner[i,1])
        y_r=int(corner[i,1]+patches)
        D = pdists_corner(input2[x_l:x_r,y_l:y_r,:],sigma)
        D = D.cpu().numpy()
        m= D[index2[0],index2[1]]
        C[index2[0], index2[1]] = D[index2[0],index2[1]]
        A[i,:,:] = C
    A = torch.from_numpy(A).type(torch.FloatTensor).cuda()
    return A


def pdists_corner(A,sigma=10):
    height,width, band = A.shape
    A=A.reshape(height * width, band)
    prod = torch.mm(A, A.t())
    norm = prod.diag().unsqueeze(1).expand_as(prod)
    res = (norm + norm.t() - 2 * prod).clamp(min=0)
    D =torch.exp(-res/(sigma ** 2))
    return D

def pdists(A,sigma=10):
    A=A.cuda()
    prod = torch.mm(A, A.t())
    norm = prod.diag().unsqueeze(1).expand_as(prod)
    res = (norm + norm.t() - 2 * prod).clamp(min=0)
    D =torch.exp(-res/(sigma ** 2))
    return D

def normalize(input):
    input_normalize = np.zeros(input.shape)
    for i in range(input.shape[2]):
        input_max = np.max(input[:,:,i])
        input_min = np.min(input[:,:,i])
        input_normalize[:,:,i] = (input[:,:,i]-input_min)/(input_max-input_min)
    return input_normalize

################get data######################################################################################################################
def load_dataset(Dataset):
    if Dataset == 'Indian':
        mat_data = sio.loadmat('/home/tsing/data/datasets/Indian_pines_corrected.mat')
        mat_gt = sio.loadmat('/home/tsing/data/datasets/Indian_pines_gt.mat')
        data_hsi = mat_data['indian_pines_corrected']
        gt_hsi = mat_gt['indian_pines_gt']
        TOTAL_SIZE = 10249
        VALIDATION_SPLIT = 0.97
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    if Dataset == 'PaviaU':
        uPavia = sio.loadmat('/home/tsing/data/datasets/PaviaU.mat')
        gt_uPavia = sio.loadmat('/home/tsing/data/datasets/PaviaU_gt.mat')
        data_hsi = uPavia['paviaU']
        gt_hsi = gt_uPavia['paviaU_gt']
        TOTAL_SIZE = 42776
        VALIDATION_SPLIT = 0.995
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)


    if Dataset == 'Salinas':
        SV = sio.loadmat('/home/tsing/data/datasets/Salinas_corrected.mat')
        gt_SV = sio.loadmat('/home/tsing/data/datasets/Salinas_gt.mat')
        data_hsi = SV['salinas_corrected']
        gt_hsi = gt_SV['salinas_gt']
        TOTAL_SIZE = 54129
        VALIDATION_SPLIT = 0.995
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)


    return data_hsi, gt_hsi, TOTAL_SIZE, TRAIN_SIZE, VALIDATION_SPLIT


def sampling(proportion, ground_truth,CLASSES_NUM):
    train = {}
    test = {}
    train_num = []
    test_num = []
    labels_loc = {}
    for i in range(CLASSES_NUM):
        indexes = np.argwhere(ground_truth == (i + 1))
        np.random.shuffle(indexes)
        labels_loc[i] = indexes
        if proportion != 1:
            # nb_val = max(int((1 - proportion) * len(indexes)), 3)
            if indexes.shape[0]<=60:
                nb_val = 15
            else:
                nb_val = 30
        else:
            nb_val = 0
        # print(i, nb_val, indexes[:nb_val])
        # train[i] = indexes[:-nb_val]
        # test[i] = indexes[-nb_val:]
        train_num.append(nb_val)
        test_num.append(len(indexes)-nb_val)
        train[i] = indexes[:nb_val]
        test[i] = indexes[nb_val:]
    train_indexes =train[0]
    test_indexes = test[0]
    for i in range(CLASSES_NUM-1):
        train_indexes= np.concatenate((train_indexes,train[i+1]),axis=0)
        test_indexes= np.concatenate((test_indexes,test[i+1]),axis=0)
    np.random.shuffle(train_indexes)
    np.random.shuffle(test_indexes)
    return train_indexes, test_indexes,train_num,test_num


def index_change(index,w):
    N=len(index)
    index2=np.zeros((N,2),dtype=int)
    for i in range(N):
        index2[i, 0] = index[i] // w
        index2[i, 1] = index[i] % w
    return index2
def get_label(indices,gt_hsi):
    dim_0 = indices[:, 0]
    dim_1 = indices[:, 1]
    label=gt_hsi[dim_0,dim_1]
    return label


def get_data(dataset):
    data_hsi, gt_hsi, TOTAL_SIZE, TRAIN_SIZE,VALIDATION_SPLIT = load_dataset(dataset)
    gt = gt_hsi.reshape(np.prod(gt_hsi.shape[:2]),)
    CLASSES_NUM = max(gt)
    train_indices, test_indices,train_num,test_num = sampling(VALIDATION_SPLIT, gt_hsi, CLASSES_NUM)
    _, total_indices,_,total_num = sampling(1, gt_hsi , CLASSES_NUM)
    y_train = get_label(train_indices, gt_hsi)-1
    y_test = get_label(test_indices, gt_hsi)-1
    y_true = get_label(total_indices, gt_hsi)-1
    return  data_hsi,CLASSES_NUM,train_indices,test_indices,total_indices,y_train, y_test, y_true

def metrics(best_OA2, best_AA_mean2, best_Kappa2,AA2):
    results = {}
    results["OA"] = best_OA2 * 100.0
    results['AA'] = best_AA_mean2 * 100.0
    results["Kappa"] = best_Kappa2 * 100.0
    results["class acc"] = AA2 * 100.0
    return results
def show_results(results, agregated=False):
    text = ""

    if agregated:
        accuracies = [r["OA"] for r in results]
        aa = [r['AA'] for r in results]
        kappas = [r["Kappa"] for r in results]
        class_acc = [r["class acc"] for r in results]

        class_acc_mean = np.mean(class_acc, axis=0)
        class_acc_std = np.std(class_acc, axis=0)

    else:
        accuracy = results["OA"]
        aa = results['AA']
        classacc = results["class acc"]
        kappa = results["Kappa"]

    text += "---\n"
    text += "class acc :\n"
    if agregated:
        for score, std in zip(class_acc_mean,
                                     class_acc_std):
            text += "\t{:.02f} +- {:.02f}\n".format(score, std)
    else:
        for score in classacc:
            text += "\t {:.02f}\n".format(score)
    text += "---\n"

    if agregated:
        text += ("OA: {:.02f} +- {:.02f}\n".format(np.mean(accuracies),
                                                         np.std(accuracies)))
        text += ("AA: {:.02f} +- {:.02f}\n".format(np.mean(aa),
                                                   np.std(aa)))
        text += ("Kappa: {:.02f} +- {:.02f}\n".format(np.mean(kappas),
                                                      np.std(kappas)))
    else:
        text += "OA : {:.02f}%\n".format(accuracy)
        text += "AA: {:.02f}%\n".format(aa)
        text += "Kappa: {:.02f}\n".format(kappa)

    print(text)







