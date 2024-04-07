import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from model.VBGE import VBGE
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal
from model.GCN import GCN


class FVGAE(nn.Module):
    """
        VGAE AND FVGAE Module layer
    """

    def __init__(self, opt):
        super(FVGAE, self).__init__()
        self.opt = opt
        self.layer_number = opt["GNN"]
        self.encoders = []
        self.encodert = []
        self.encoderst = []
        for i in range(self.layer_number - 1):
            self.encoders.append(DGCNLayer(opt))
            self.encodert.append(DGCNLayer(opt))
            self.encoderst.append(DGCNLayer(opt))
        self.encoders.append(LastLayer(opt))
        self.encoders = nn.ModuleList(self.encoders)
        self.encodert.append(LastLayer(opt))
        self.encodert = nn.ModuleList(self.encodert)
        self.encoderst.append(LastLayer(opt))
        self.encoderst = nn.ModuleList(self.encoderst)
        self.dropout = opt["dropout"]
        self.kl_share = 0

        self.connect1 = nn.Sequential(
            nn.Linear(opt["feature_dim"]*2 , opt["feature_dim"])
        )
        self.connect11 = nn.Sequential(
            nn.Linear(opt["feature_dim"], opt["feature_dim"])
        )
        self.connect2 = nn.Sequential(
            nn.Linear(opt["feature_dim"] * 2, opt["feature_dim"])
        )
        self.connect22 = nn.Sequential(
            nn.Linear(opt["feature_dim"], opt["feature_dim"])
        )

    def forward(self, ufeas, vfeas, UV_adjs, VU_adjs,ufeat, vfeat, UV_adjt, VU_adjt):
        learn_users = ufeas
        learn_items = vfeas
        user_rets = None
        item_rets = None

        learn_usert = ufeat
        learn_itemt = vfeat
        user_rett = None
        item_rett = None

        learn_userst = (ufeas + ufeat)/2
        learn_itemst = torch.cat((vfeas, vfeat), dim=0)
        # learn_userst = (learn_users + learn_usert) / 2
        VU_adjst = torch.cat((VU_adjs, VU_adjt), dim=0)
        UV_adjst = torch.cat((UV_adjs.T, UV_adjt.T), dim=0).T

        flag=0
        for layers, layert,layerst in zip(self.encoders, self.encodert, self.encoderst):#for layer in self.encoder:

            if flag==0 or flag==1:

                learn_users = F.dropout(learn_users, self.dropout, training=self.training)
                learn_items = F.dropout(learn_items, self.dropout, training=self.training)
                learn_users, learn_items = layers(learn_users, learn_items, UV_adjs, VU_adjs)

                learn_usert = F.dropout(learn_usert, self.dropout, training=self.training)
                learn_itemt = F.dropout(learn_itemt, self.dropout, training=self.training)
                learn_usert, learn_itemt = layert(learn_usert, learn_itemt, UV_adjt, VU_adjt)

                learn_userst = F.dropout(learn_userst, self.dropout, training=self.training)
                learn_itemst = F.dropout(learn_itemst, self.dropout, training=self.training)
                learn_userst, learn_itemst = layerst(learn_userst, learn_itemst, UV_adjst, VU_adjst)

                #learn_userst = F.dropout(learn_userst, self.dropout, training=self.training)
                #learn_itemst = F.dropout(learn_itemst, self.dropout, training=self.training)
                #learn_userst, learn_itemst = layerst(learn_userst, learn_itemst, UV_adjst, VU_adjst)
            else:
                learn_users = F.dropout(learn_users, self.dropout, training=self.training)
                learn_items = F.dropout(learn_items, self.dropout, training=self.training)
                learn_users, learn_items,mean_u_s,mean_i_s,logstd_u_s,logstd_i_s= layers(learn_users, learn_items, UV_adjs, VU_adjs)

                learn_usert = F.dropout(learn_usert, self.dropout, training=self.training)
                learn_itemt = F.dropout(learn_itemt, self.dropout, training=self.training)
                learn_usert, learn_itemt,mean_u_t,mean_i_t,logstd_u_t,logstd_i_t = layert(learn_usert, learn_itemt, UV_adjt, VU_adjt)
                mean_u=(self.opt["beta_st"]*mean_u_s+mean_u_t)/(1+self.opt["beta_st"])#mean_u=(mean_u_s+mean_u_t)/2
                mean_i= torch.cat((mean_i_s, mean_i_t), dim=0)
                logstd_u = (self.opt["beta_st"]*logstd_u_s + logstd_u_t)/(1+self.opt["beta_st"])#logstd_u = (logstd_u_s + logstd_u_t)/2
                logstd_i = torch.cat((logstd_i_s, logstd_i_t), dim=0)
                learn_userst, learn_itemst = layerst.forward_user_shareconnect(learn_userst, learn_itemst, UV_adjst, VU_adjst,mean_u,mean_i,logstd_u,logstd_i)

            if user_rets is None:
                user_rets = learn_users
                item_rets = learn_items
            else:
                user_rets = torch.cat((user_rets, learn_users), dim=-1)
                item_rets = torch.cat((item_rets, learn_items), dim=-1)

            if user_rett is None:
                user_rett = learn_usert
                item_rett = learn_itemt
            else:
                user_rett = torch.cat((user_rett, learn_usert), dim=-1)
                item_rett = torch.cat((item_rett, learn_itemt), dim=-1)



            flag = flag + 1
        return learn_users, learn_items,learn_usert, learn_itemt,learn_userst, learn_itemst

    def forward_user_share(self, ufea, UV_adj, VU_adj):
        learn_user = ufea
        for layer in self.encoder[:-1]:
            learn_user = F.dropout(learn_user, self.dropout, training=self.training)
            learn_user = layer.forward_user_share(learn_user, UV_adj, VU_adj)
        mean, sigma = self.encoder[-1].forward_user_share(learn_user, UV_adj, VU_adj)
        return mean, sigma




class DGCNLayer(nn.Module):
    """
        DGCN Module layer
    """

    def __init__(self, opt):
        super(DGCNLayer, self).__init__()
        self.opt = opt
        self.dropout = opt["dropout"]
        self.gc1 = GCN(
            nfeat=opt["feature_dim"],
            nhid=opt["hidden_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )

        self.gc2 = GCN(
            nfeat=opt["feature_dim"],
            nhid=opt["hidden_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
        self.gc3 = GCN(
            nfeat=opt["hidden_dim"],  # change
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )

        self.gc4 = GCN(
            nfeat=opt["hidden_dim"],  # change
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
        self.user_union = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
        self.item_union = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])

    def forward(self, ufea, vfea, UV_adj, VU_adj):
        User_ho = self.gc1(ufea, VU_adj)
        Item_ho = self.gc2(vfea, UV_adj)
        User_ho = self.gc3(User_ho, UV_adj)
        Item_ho = self.gc4(Item_ho, VU_adj)
        User = torch.cat((User_ho, ufea), dim=1)
        Item = torch.cat((Item_ho, vfea), dim=1)
        User = self.user_union(User)
        Item = self.item_union(Item)
        return F.relu(User), F.relu(Item)

    def forward_user(self, ufea, vfea, UV_adj, VU_adj):
        User_ho = self.gc1(ufea, VU_adj)
        User_ho = self.gc3(User_ho, UV_adj)
        User = torch.cat((User_ho, ufea), dim=1)
        User = self.user_union(User)
        return F.relu(User)

    def forward_item(self, ufea, vfea, UV_adj, VU_adj):
        Item_ho = self.gc2(vfea, UV_adj)
        Item_ho = self.gc4(Item_ho, VU_adj)
        Item = torch.cat((Item_ho, vfea), dim=1)
        Item = self.item_union(Item)
        return F.relu(Item)

    def forward_user_share(self, ufea, UV_adj, VU_adj):#---
        User_ho = self.gc1(ufea, VU_adj)
        User_ho = self.gc3(User_ho, UV_adj)
        User = torch.cat((User_ho, ufea), dim=1)
        User = self.user_union(User)
        return F.relu(User)

    def forward_user_shareconnect(self, ufea,vfea, UV_adj, VU_adj):#---
        User_ho = self.gc1(ufea, VU_adj)
        User_ho = self.gc3(User_ho, UV_adj)
        User = torch.cat((User_ho, ufea), dim=1)
        User = self.user_union(User)

        Item_ho = self.gc2(vfea, UV_adj)
        Item_ho = self.gc4(Item_ho, VU_adj)
        Item = torch.cat((Item_ho, vfea), dim=1)
        Item = self.item_union(Item)
        return F.relu(User),F.relu(Item)


class LastLayer(nn.Module):
    """
        DGCN Module layer
    """

    def __init__(self, opt):
        super(LastLayer, self).__init__()
        self.opt = opt
        self.dropout = opt["dropout"]
        self.gc1 = GCN(
            nfeat=opt["feature_dim"],
            nhid=opt["hidden_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )

        self.gc2 = GCN(
            nfeat=opt["feature_dim"],
            nhid=opt["hidden_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
        self.gc3_mean = GCN(
            nfeat=opt["hidden_dim"],  # change
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
        self.gc3_logstd = GCN(
            nfeat=opt["hidden_dim"],  # change
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )

        self.gc4_mean = GCN(
            nfeat=opt["hidden_dim"],  # change
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
        self.gc4_logstd = GCN(
            nfeat=opt["hidden_dim"],  # change
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
        self.user_union_mean = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
        self.user_union_logstd = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
        self.item_union_mean = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
        self.item_union_logstd = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])

        self.user_union_mean_con= nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
        self.user_union_logstd_con= nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
        self.item_union_mean_con= nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
        self.item_union_logstd_con= nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
        self.kld_loss =0

    def _kld_gauss(self, mu_1, logsigma_1, mu_2, logsigma_2):
        """Using std to compute KLD"""
        sigma_1 = torch.exp(0.1 + 0.9 * F.softplus(torch.clamp_max(logsigma_1, 0.4)))
        sigma_2 = torch.exp(0.1 + 0.9 * F.softplus(torch.clamp_max(logsigma_2, 0.4)))
        q_target = Normal(mu_1, sigma_1)
        q_context = Normal(mu_2, sigma_2)
        kl = kl_divergence(q_target, q_context).mean(dim=0).sum()
        return kl

    def reparameters(self, mean, logstd):
        sigma = torch.exp(0.1 + 0.9 * F.softplus(torch.clamp_max(logstd, 0.4)))
        gaussian_noise = torch.randn(mean.size(0), self.opt["hidden_dim"]).cuda(mean.device)
        if self.gc1.training:
            self.sigma = sigma
            sampled_z = gaussian_noise * sigma + mean
        else:
            sampled_z = mean
        kld_loss = self._kld_gauss(mean, logstd, torch.zeros_like(mean), torch.ones_like(logstd))
        return sampled_z, kld_loss

    def forward(self, ufea, vfea, UV_adj, VU_adj):
        item, item_kld,mean_i,logstd_i = self.forward_item(ufea, vfea, UV_adj, VU_adj)
        user, user_kld ,mean_u,logstd_u= self.forward_user(ufea, vfea, UV_adj, VU_adj)

        self.kld_loss = self.opt["beta"] * user_kld + item_kld

        return user, item,mean_u,mean_i,logstd_u,logstd_i

    def forward_user(self, ufea, vfea, UV_adj, VU_adj):
        User_ho = self.gc1(ufea, VU_adj)
        User_ho_mean = self.gc3_mean(User_ho, UV_adj)
        User_ho_logstd = self.gc3_logstd(User_ho, UV_adj)
        User_ho_mean = torch.cat((User_ho_mean, ufea), dim=1)
        User_ho_mean = self.user_union_mean(User_ho_mean)

        User_ho_logstd = torch.cat((User_ho_logstd, ufea), dim=1)
        User_ho_logstd = self.user_union_logstd(User_ho_logstd)

        user, kld_loss = self.reparameters(User_ho_mean, User_ho_logstd)
        return user, kld_loss,User_ho_mean, User_ho_logstd

    def forward_item(self, ufea, vfea, UV_adj, VU_adj):
        Item_ho = self.gc2(vfea, UV_adj)

        Item_ho_mean = self.gc4_mean(Item_ho, VU_adj)
        Item_ho_logstd = self.gc4_logstd(Item_ho, VU_adj)
        Item_ho_mean = torch.cat((Item_ho_mean, vfea), dim=1)
        Item_ho_mean = self.item_union_mean(Item_ho_mean)

        Item_ho_logstd = torch.cat((Item_ho_logstd, vfea), dim=1)
        Item_ho_logstd = self.item_union_logstd(Item_ho_logstd)

        item, kld_loss = self.reparameters(Item_ho_mean, Item_ho_logstd)
        return item, kld_loss,Item_ho_mean, Item_ho_logstd

    def forward_user_share(self, ufea, UV_adj, VU_adj):#---
        User_ho = self.gc1(ufea, VU_adj)
        User_ho_mean = self.gc3_mean(User_ho, UV_adj)
        User_ho_logstd = self.gc3_logstd(User_ho, UV_adj)
        User_ho_mean = torch.cat((User_ho_mean, ufea), dim=1)
        User_ho_mean = self.user_union_mean(User_ho_mean)

        User_ho_logstd = torch.cat((User_ho_logstd, ufea), dim=1)
        User_ho_logstd = self.user_union_logstd(User_ho_logstd)

        return User_ho_mean, User_ho_logstd

    def forward_user_shareconnect(self, ufea, vfea,UV_adj, VU_adj,mean_u,mean_i,lg_u,lg_i):#---
        User_ho = self.gc1(ufea, VU_adj)
        User_ho_mean = self.gc3_mean(User_ho, UV_adj)
        User_ho_logstd = self.gc3_logstd(User_ho, UV_adj)
        User_ho_mean = torch.cat((User_ho_mean, ufea), dim=1)
        User_ho_mean = self.user_union_mean(User_ho_mean)

        User_ho_logstd = torch.cat((User_ho_logstd, ufea), dim=1)
        User_ho_logstd = self.user_union_logstd(User_ho_logstd)
        mean_u_st=torch.cat((User_ho_mean,mean_u),dim=1)
        mean_u_st=self.user_union_mean_con(mean_u_st)

        logstd_u_st=torch.cat((User_ho_logstd,lg_u),dim=1)
        logstd_u_st = self.user_union_logstd_con(logstd_u_st)#logstd_u_st = self.user_union_logstd(logstd_u_st)

        user, u_kld_loss = self.reparameters(mean_u_st, logstd_u_st)

        Item_ho = self.gc2(vfea, UV_adj)

        Item_ho_mean = self.gc4_mean(Item_ho, VU_adj)
        Item_ho_logstd = self.gc4_logstd(Item_ho, VU_adj)
        Item_ho_mean = torch.cat((Item_ho_mean, vfea), dim=1)
        Item_ho_mean = self.item_union_mean(Item_ho_mean)

        Item_ho_logstd = torch.cat((Item_ho_logstd, vfea), dim=1)
        Item_ho_logstd = self.item_union_logstd(Item_ho_logstd)

        mean_i_st=torch.cat((Item_ho_mean, mean_i), dim=1)
        mean_i_st = self.item_union_mean_con(mean_i_st)

        logstd_i_st =torch.cat((Item_ho_logstd, lg_i), dim=1)
        logstd_i_st = self.item_union_logstd_con(logstd_i_st)
        item, i_kld_loss = self.reparameters(mean_i_st, logstd_i_st)

        self.kld_loss = self.opt["beta"] * u_kld_loss + i_kld_loss

        return user, item




