import argparse
import os
import pandas

import numpy as np
import torch
torch.cuda.empty_cache()
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter


import utils_2
from model import Model
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print('Home device: {}'.format(device))
global top
top = np.zeros((402,2))


def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask


def nce_supervised_easy(out_1, out_2, label):
    #supervised contrastive loss baseline
    out = torch.cat([out_1, out_2], dim=0)
    label = torch.cat([label, label], dim=0)
    cost = torch.exp(2*torch.mm(out, out.t().contiguous()))
    batch = label.shape[0]
    pos_index = torch.zeros((batch, batch)).cuda()
    same_index = torch.eye(batch).cuda()
    for i in range(batch):
        ind = torch.where(label == label[i])[0]
        pos_index[i][ind] = 1
    neg_index = 1 - pos_index
    pos_index = pos_index - same_index
    pos = pos_index * cost
    neg = neg_index * cost
    neg_exp_sum = (neg.sum(1))/neg_index.sum(1)
    Nce = pos_index * (pos/(pos+(batch - 2)*neg_exp_sum.reshape(-1,1)))
    final_index = torch.where(Nce!=0)
    Nce = -((torch.log(Nce[final_index[0], final_index[1]])).mean())
    return Nce


def nce_supervised_hard(out_1, out_2, label, beta):
    out = torch.cat([out_1, out_2], dim=0)
    label = torch.cat([label, label], dim=0)
    cost = torch.exp(2*torch.mm(out, out.t().contiguous()))
    batch = label.shape[0]
    pos_index = torch.zeros((batch, batch)).cuda()
    pos_index_2 = torch.zeros((batch, batch)).cuda()
    same_index = torch.eye(batch).cuda()
    for i in range(batch):
        ind = torch.where(label == label[i])[0]
        pos_index[i][ind] = 1
        if i < batch//2:
            pos_index_2[i][i+batch//2] = 1
            pos_index_2[i+batch//2][i] = 1
    neg_index = 1 - pos_index
    pos_index = pos_index - same_index
    pos = pos_index_2 * cost
    neg = neg_index * cost
    imp = neg_index*(beta* neg.log()).exp()
    imp = imp.detach()
    neg_exp_sum = (imp*neg).sum(dim = -1) / imp.sum(dim = -1)
    Nce = pos_index_2 * (pos/(pos+(batch - 2)*neg_exp_sum.reshape(-1,1)))
    final_index = torch.where(pos_index_2!=0)
    Nce = (-torch.log(Nce[final_index[0], final_index[1]])).mean()
    return Nce

def criterion(out_1,out_2,tau_plus,batch_size,beta, estimator):
    # neg score
    out = torch.cat([out_1, out_2], dim=0)
    neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
    old_neg = neg.clone()
    mask = get_negative_mask(batch_size).to(device)
    neg = neg.masked_select(mask).view(2 * batch_size, -1)

    # pos score
    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)
        
    # negative samples similarity scoring
    if estimator=='hard':
        N = batch_size * 2 - 2
        imp = (beta* neg.log()).exp()
        reweight_neg = (imp*neg).sum(dim = -1) / imp.mean(dim = -1)
        Ng = (-tau_plus * N * pos + reweight_neg) / (1 - tau_plus)
        # constrain (optional)
        #Ng = torch.clamp(Ng, min = N * np.e**(-1 / temperature))
    elif estimator=='easy':
        Ng = neg.sum(dim=-1)
    else:
        raise Exception('Invalid estimator selected. Please use any of [hard, easy]')
            
    # contrastive loss
    loss = (- torch.log(pos / (pos + Ng) )).mean()

    return loss


def train(net, data_loader, train_optimizer, temperature, estimator, beta):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    total_loss_hucl = 0
    total_loss_ucl = 0
    total_loss_scl = 0
    total_loss_hscl = 0
    for pos_1, pos_2, target in train_bar:
        pos_1, pos_2 = pos_1.to(device,non_blocking=True), pos_2.to(device,non_blocking=True)
        feature_1, out_1 = net(pos_1)
        feature_2, out_2 = net(pos_2)
        
        #H-UCL
        loss_hucl = criterion(out_1,out_2,tau_plus,batch_size,beta, 'hard')
        #UCL
        loss_ucl = criterion(out_1,out_2,tau_plus,batch_size,beta, 'easy')
        #SCL
        loss_scl = nce_supervised_easy(out_1, out_2, target)
        #H-SCL
        loss_hscl = nce_supervised_hard(out_1, out_2, target, beta)


        train_optimizer.zero_grad()
        #Choose one unsupervised loss to for optimizing
        #For supervised case please refer to main.py
        loss_hucl.backward(retain_graph=True)
        loss_ucl.backward(retain_graph=True)
        loss_hscl.backward(retain_graph=True)
        loss_scl.backward(retain_graph=True)
        train_optimizer.step()

        total_num += batch_size
        total_loss_hucl += loss_hucl.item() * batch_size
        total_loss_ucl += loss_ucl.item() * batch_size
        total_loss_scl += loss_scl.item() * batch_size
        total_loss_hscl += loss_hscl.item() * batch_size

        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss_hucl / total_num))
    
    loss_array[epoch][0] = float(total_loss_hucl / total_num)
    loss_array[epoch][1] = float(total_loss_ucl / total_num)
    loss_array[epoch][2] = float(total_loss_hscl / total_num)
    loss_array[epoch][3] = float(total_loss_scl / total_num)
    
    np.save("ferplus_loss_beta_"+str(beta)+".npy", loss_array)
    
    return total_loss_hucl / total_num



# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, memory_data_loader, test_data_loader, beta, dataset_name, estimator):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature, out = net(data.to(device, non_blocking=True))
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        if 'cifar' in dataset_name:
            feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device) 
        elif 'ferpair_true_label' in dataset_name:
            feature_labels = torch.tensor(memory_data_loader.dataset.labels, device=feature_bank.device) 
        elif 'ferplus' in dataset_name:
            feature_labels = torch.tensor(memory_data_loader.dataset.labels, device=feature_bank.device)

        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, _, target in test_bar:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            feature, out = net(data)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1).long(), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:,:1] == target.long().unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:,:5] == target.long().unsqueeze(dim=-1)).any(dim=-1).float()).item()
            top[epoch][0] = float(total_top1 / total_num * 100)
            top[epoch][1] = float(total_top5 / total_num * 100)
            print(epoch, epochs, 'total_top1', total_top1, total_num, total_top1 / total_num * 100, 'total_top5', total_top5, total_top5 / total_num * 100)
            test_bar.set_description('KNN Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))
            

    return total_top1 / total_num * 100, total_top5 / total_num * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=32, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--tau_plus', default=0.1, type=float, help='Positive class priorx')
    parser.add_argument('--epochs', default=200, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--estimator', default='hard', type=str, help='Choose loss function')
    parser.add_argument('--dataset_name', default='ferplus', type=str, help='Choose dataset')
    parser.add_argument('--beta', default=0.5, type=float, help='beta')
    parser.add_argument('--gradient_imp', default=True, type=bool, help='gradient_imp')
    parser.add_argument('--N', default=1, type=float, help='M_view')
    parser.add_argument('--M', default=2, type=float, help='N_view')

    tb_path = 'C:/Users/Karth/Downloads/H-SCL-main/logs'
    tb_writer = SummaryWriter(log_dir=tb_path)

    # args parse
    args = parser.parse_args()
    feature_dim, temperature, k, tau_plus = args.feature_dim, args.temperature, args.k, args.tau_plus
    batch_size, epochs, estimator,beta = args.batch_size, args.epochs, args.estimator, args.beta
    M_view, N_view = args.M, args.N
    dataset_name = args.dataset_name
    gradient_imp = args.gradient_imp
    tb_path = 'C:/Users/Karth/Downloads/H-SCL-main/logs/{}_tensorboard'.format(args.dataset_name)
    
    SEED = 0
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    print(dataset_name, "estimator", estimator, beta)

    
    # data prepare
    root = './data/'
    train_data, memory_data, test_data = utils_2.get_dataset(dataset_name, M_view, N_view, root)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # model setup and optimizer config
    model = Model(feature_dim).to(device)
    model = nn.DataParallel(model)

    optimizer = Adam(model.parameters(), lr=0.001)

    # Path to checkpoint file
    checkpoint_dir = 'C:/Users/Karth/Downloads/H-SCL-main/results/ferplus'
    checkpoint_path = os.path.join(checkpoint_dir, 'ferplus_hard_32_0.5_True_paper_HSCL_final.pth')

    # Load checkpoint if resume is enabled
    if args.resume:
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(checkpoint_path), f'Error: checkpoint file not found at {checkpoint_path}'
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']

    else:
        start_epoch = 0
        loss = float('inf')

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    c = len(memory_data.classes)
    print(dataset_name, '# Classes: {}'.format(c))
    acc_1 = np.zeros((epochs,2))
    loss_array = np.zeros((epochs + 1, 4))

    # training loop 
    if not os.path.exists('./results'):
        os.mkdir('./results')
    if not os.path.exists('./results/{}'.format(dataset_name)):
        os.mkdir('./results/{}'.format(dataset_name))
        
    for epoch in range(start_epoch, epochs + 1):
        train_loss = train(model, train_loader, optimizer, temperature, estimator, beta)
        test_acc_1, test_acc_5 = test(model, memory_loader, test_loader, beta, dataset_name, estimator)
        print(float(train_loss), test_acc_1, test_acc_5)
        acc_1[epoch-1][0] = float(train_loss)
        acc_1[epoch-1][1] = test_acc_1
        
        # Log metrics to TensorBoard
        loss_values = {
            'H-UCL': loss_array[epoch][0],
            'UCL': loss_array[epoch][1],
            'H-SCL': loss_array[epoch][2],
            'SCL': loss_array[epoch][3]
            }
        tb_writer.add_scalars('Loss/beta_0.5', loss_values, epoch)
        # tb_writer.add_scalar('Accuracy/Test@1', test_acc_1, epoch)
        # tb_writer.add_scalar('Accuracy/Test@5', test_acc_5, epoch)
        
        np.save(dataset_name+"_final_acc_1_"+estimator+str(beta)+'_Seed_'+str(SEED)+str(gradient_imp), acc_1)
        
        # Save checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'loss': train_loss
            
        }
        torch.save(checkpoint, './results/{}/{}_{}_{}_{}_{}_paper_HSCL_final_ResNet18.pth'.format(dataset_name,dataset_name,estimator,batch_size,beta, gradient_imp))

    tb_writer.close()

