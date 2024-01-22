import torch
import argparse
import torch.nn as nn
import torch.utils.data as Data
import torch.backends.cudnn as cudnn
import numpy as np
import time
import os

from vit_gcn import ViT,GCN
from functions import normalize,get_data,GET_A2,metrics,show_results,train_and_test_data,train_epoch,valid_epoch,output_metric,applyPCA


parser = argparse.ArgumentParser("HSI")
parser.add_argument('--dataset', choices=['Indian', 'PaviaU', 'Salinas'],
                    default='Indian', help='dataset to use')

parser.add_argument("--num_run", type=int, default=5)
parser.add_argument('--epoches', type=int, default=200, help='epoch number')
parser.add_argument('--patches', type=int, default=9, help='number of patches')#奇数 9
parser.add_argument('--n_gcn', type=int, default=21, help='number of related pix')
parser.add_argument('--pca_band', type=int, default=70, help='pca_components')
parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')

parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
parser.add_argument('--gpu_id', default='0', help='gpu id')
parser.add_argument('--seed', type=int, default=0, help='number of seed')
parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')#64
parser.add_argument('--test_freq', type=int, default=10, help='number of evaluation')
args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
# -------------------------------------------------------------------------------

# Parameter Setting
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
cudnn.deterministic = True
cudnn.benchmark = False
# prepare data

input, num_classes, total_pos_train, total_pos_test, total_pos_true, y_train, y_test, y_true = get_data(args.dataset)
##########得到原始图像 训练测试以及所有点坐标 每一类训练测试的个数############


input = applyPCA(input, numComponents=args.pca_band)
# normalize data by band norm
input_normalize = normalize(input)
height, width, band = input_normalize.shape  # 145*145*200
print("height={0},width={1},band={2}".format(height, width, band))
# -------------------------------------------------------------------------------
# obtain train and test data
x_train_band, x_test_band, x_true_band, corner_train, corner_test, corner_true, indexs_train, indexs_test, indexs_ture = train_and_test_data(
    input_normalize, band, total_pos_train, total_pos_test, total_pos_true, patch=args.patches, w=height, h=width,
    n_gcn=args.n_gcn)
##########得到训练测试以及所有点的光谱############



input2 = torch.from_numpy(input_normalize).type(torch.FloatTensor)

A_train = GET_A2(x_train_band, input2, corner=corner_train,patches=args.patches , l=3,sigma=10)
x_train = torch.from_numpy(x_train_band).type(torch.FloatTensor)  # [695, 200, 7, 7]
y_train = torch.from_numpy(y_train).type(torch.LongTensor)  # [695]
Label_train = Data.TensorDataset(A_train, x_train, y_train)

A_test = GET_A2(x_test_band, input2, corner=corner_test, patches=args.patches ,l=3, sigma=10)
x_test = torch.from_numpy(x_test_band).type(torch.FloatTensor)  # [9671, 200, 7, 7]
y_test = torch.from_numpy(y_test).type(torch.LongTensor)  # [9671]
Label_test = Data.TensorDataset(A_test, x_test, y_test)

x_true = torch.from_numpy(x_true_band).type(torch.FloatTensor)
y_true = torch.from_numpy(y_true).type(torch.LongTensor)
Label_true = Data.TensorDataset(x_true, y_true)

label_train_loader = Data.DataLoader(Label_train, batch_size=args.batch_size, shuffle=True)
##########训练集的光谱值及标签##########
label_test_loader = Data.DataLoader(Label_test, batch_size=args.batch_size, shuffle=True)
##########测试集的光谱值及标签##########
label_true_loader = Data.DataLoader(Label_true, batch_size=100, shuffle=False)
##########所有地物的光谱值及标签##########



results = []

best_OA2 = 0.0
best_AA_mean2 = 0.0
best_Kappa2 = 0.0
gcn_net = GCN(height, width, band, num_classes)
gcn_net = gcn_net.cuda()
# criterion
criterion = nn.CrossEntropyLoss().cuda()
# optimizer
optimizer = torch.optim.Adam(gcn_net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epoches//2, gamma=args.gamma)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epoches // 10, gamma=args.gamma)
# -------------------------------------------------------------------------------

tr_net = ViT(
    n_gcn=args.n_gcn,
    num_patches=64,
    num_classes=num_classes,
    dim=64,
    depth=5,
    heads=4,
    mlp_dim=8,
    dropout=0.1,
    emb_dropout=0.1,
)
tr_net = tr_net.cuda()
# optimizer
optimizer2 = torch.optim.Adam(tr_net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=args.epoches // 10, gamma=args.gamma)

print("start training")
tic = time.time()
for epoch in range(args.epoches):
    scheduler.step()
    scheduler2.step()
    # train model
    gcn_net.train()
    tr_net.train()
    train_acc, train_obj, tar_t, pre_t = train_epoch(gcn_net, tr_net, label_train_loader, criterion, optimizer,
                                                     optimizer2, indexs_train)
    OA1, AA_mean1, Kappa1, AA1 = output_metric(tar_t, pre_t)
    print("Epoch: {:03d} train_loss: {:.4f} train_acc: {:.4f}"
          .format(epoch + 1, train_obj, train_acc))

    if (epoch % args.test_freq == 0) | (epoch == args.epoches - 1)and epoch>=args.epoches*0.6:

        gcn_net.eval()
        tr_net.eval()
        tar_v, pre_v = valid_epoch(gcn_net, tr_net, label_test_loader, criterion, indexs_test)
        OA2, AA_mean2, Kappa2, AA2 = output_metric(tar_v, pre_v)
        if OA2 >= best_OA2 and AA_mean2 >= best_AA_mean2 and Kappa2 >= best_Kappa2:
            best_OA2 = OA2
            best_AA_mean2 = AA_mean2
            best_Kappa2 = Kappa2
            run_results = metrics(best_OA2, best_AA_mean2, best_Kappa2,AA2)
show_results(run_results,agregated=False)
results.append(run_results)
toc = time.time()










