import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal
from model.GCN import GCN
from model.FVGAE import FVGAE



class d2c2r(nn.Module):
    def __init__(self, opt):
        super(d2c2r, self).__init__()
        self.opt=opt
        self.st_GNN = FVGAE(opt)
        self.source_share_GNN = FVGAE(opt)
        self.target_share_GNN = FVGAE(opt)

        self.criterion = nn.BCEWithLogitsLoss()

        self.dropout = opt["dropout"]


        self.source_user_embedding = nn.Embedding(opt["source_user_num"], opt["feature_dim"])
        self.target_user_embedding = nn.Embedding(opt["target_user_num"], opt["feature_dim"])
        self.source_item_embedding = nn.Embedding(opt["source_item_num"], opt["feature_dim"])
        self.target_item_embedding = nn.Embedding(opt["target_item_num"], opt["feature_dim"])
        self.source_user_embedding_share = nn.Embedding(opt["source_user_num"], opt["feature_dim"])
        self.target_user_embedding_share = nn.Embedding(opt["target_user_num"], opt["feature_dim"])

        self.share_mean = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
        self.share_sigma = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])

        # self.shared_user = torch.arange(0, self.opt["shared_user"], 1)
        self.source_user_index = torch.arange(0, self.opt["source_user_num"], 1)
        self.target_user_index = torch.arange(0, self.opt["target_user_num"], 1)
        self.source_item_index = torch.arange(0, self.opt["source_item_num"], 1)
        self.target_item_index = torch.arange(0, self.opt["target_item_num"], 1)

        if self.opt["cuda"]:
            self.criterion.cuda()
            # self.shared_user = self.shared_user.cuda()
            self.source_user_index = self.source_user_index.cuda()
            self.target_user_index = self.target_user_index.cuda()
            self.source_item_index = self.source_item_index.cuda()
            self.target_item_index = self.target_item_index.cuda()


        ##########
        self.hidden = 16  # dim 16
        self.layers = 3 - 1#2-1
        self.net_in_s = nn.Sequential(
            nn.Linear(opt["feature_dim"]*4, self.hidden),
            nn.ReLU(),
        )
        self.net_hid_s = nn.Sequential(
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
        )
        self.net_out_s = nn.Sequential(
            nn.Linear(self.hidden, opt["feature_dim"]),
        )

        self.net_in_t = nn.Sequential(
            nn.Linear(opt["feature_dim"] *4, self.hidden),
            nn.ReLU(),
        )
        self.net_hid_t = nn.Sequential(
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
        )
        self.net_out_t = nn.Sequential(
            nn.Linear(self.hidden, opt["feature_dim"] ),
        )


        ##########~~~~~~~~~~~~~~~~
        self.net_in_iis = nn.Sequential(
            nn.Linear(opt["feature_dim"]*2, self.hidden),
            nn.ReLU(),
        )
        self.net_hid_iis = nn.Sequential(
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
        )
        self.net_out_iis = nn.Sequential(
            nn.Linear(self.hidden, opt["feature_dim"]),
        )

        self.net_in_iit = nn.Sequential(
            nn.Linear(opt["feature_dim"]*2, self.hidden),
            nn.ReLU(),
        )
        self.net_hid_iit = nn.Sequential(
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
        )
        self.net_out_iit = nn.Sequential(
            nn.Linear(self.hidden, opt["feature_dim"]),
        )
        #######


        self.W_q = nn.Linear(opt["feature_dim"], 32)
        self.W_k = nn.Linear(opt["feature_dim"], 32)
        self.W_v = nn.Linear(opt["feature_dim"]*opt["beta_cat"], 32)
        self.W_o = nn.Linear(32, opt["feature_dim"]*opt["beta_cat"])

        self.W_qt= nn.Linear(opt["feature_dim"] , 32)
        self.W_kt = nn.Linear(opt["feature_dim"] , 32)
        self.W_vt = nn.Linear(opt["feature_dim"] * opt["beta_cat"], 32)
        self.W_ot = nn.Linear(32, opt["feature_dim"] * opt["beta_cat"])



    def calculate_uis(self,e_u, e_i):
        u, d = e_u.size()

        e_u_c = e_u.view(u, 1, d)
        e_i_c = e_i.view(u, 1, d)
        catcat = torch.cat((e_u_c, e_i_c), dim=1)

        Q = self.W_q(e_i_c)
        K = self.W_k(catcat)

        attention_scores = Q.matmul(K.transpose(1, 2)) / (d ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)

        attended_e_i = attention_weights[:, :, 0:1] * e_i_c
        # Ablation1
        # attended_e_i =e_i_c

        diff_tensor = e_u_c - attended_e_i

        elementwise_product = e_u_c * attended_e_i

        # Ablation2
        # diff_tensor = e_u_c
        # elementwise_product = attended_e_i

        concatenated_tensor = torch.cat([e_u_c, attended_e_i], dim=-1)
        h_ui = torch.cat([concatenated_tensor, diff_tensor, elementwise_product], dim=-1)

        output = h_ui.view(u, -1)


        return output

    def calculate_uit(self,e_u, e_i):
        u, d = e_u.size()

        e_u_c = e_u.view(u, 1, d)
        e_i_c = e_i.view(u, 1, d)
        catcat = torch.cat((e_u_c, e_i_c), dim=1)

        Q = self.W_qt(e_i_c)
        K = self.W_kt(catcat)

        attention_scores = Q.matmul(K.transpose(1, 2)) / (d ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)


        attended_e_i = attention_weights[:, :, 0:1] * e_i_c
        # Ablation1
        # attended_e_i = e_i_c

        diff_tensor = e_u_c - attended_e_i

        elementwise_product = e_u_c * attended_e_i

        # Ablation2
        # diff_tensor = e_u_c
        # elementwise_product = attended_e_i

        concatenated_tensor = torch.cat([e_u_c, attended_e_i], dim=-1)
        h_ui = torch.cat([concatenated_tensor, diff_tensor, elementwise_product], dim=-1)

        output = h_ui.view(u, -1)

        return output


    def source_predict_nn(self, user_embedding, item_embedding):
        fea = torch.cat((user_embedding, item_embedding), dim=-1)
        out = self.source_predict_1(fea)
        out = F.relu(out)
        out = self.source_predict_2(out)
        out = torch.sigmoid(out)
        return out

    def target_predict_nn(self, user_embedding, item_embedding):
        fea = torch.cat((user_embedding, item_embedding), dim=-1)
        out = self.target_predict_1(fea)
        out = F.relu(out)
        out = self.target_predict_2(out)
        out = torch.sigmoid(out)
        return out

    def source_predict_dot(self, user_embedding, item_embedding):
        output = (user_embedding * item_embedding).sum(dim=-1)
        # return torch.sigmoid(output)
        return output

    def target_predict_dot(self, user_embedding, item_embedding):
        output = (user_embedding* item_embedding).sum(dim=-1)
        # return torch.sigmoid(output)
        return output

    def my_index_select(self, memory, index):
        tmp = list(index.size()) + [-1]
        index = index.view(-1)
        ans = torch.index_select(memory, 0, index)
        ans = ans.view(tmp)
        return ans

    def HingeLoss(self, pos, neg):
        pos = F.sigmoid(pos)
        neg = F.sigmoid(neg)
        gamma = torch.tensor(self.opt["margin"])
        if self.opt["cuda"]:
            gamma = gamma.cuda()
        return F.relu(gamma - pos + neg).mean()

    def mapping_s(self,A):
        x = self.net_in_s(A)
        for _ in range(self.layers):
            x = self.net_hid_s(x)
        x = self.net_out_s(x)
        return x
    def mapping_t(self,A):
        x = self.net_in_t(A)
        for _ in range(self.layers):
            x = self.net_hid_t(x)
        x = self.net_out_t(x)
        return x

    def mapping_iis(self,A):
        x = self.net_in_iis(A)
        for _ in range(self.layers):
            x = self.net_hid_iis(x)
        x = self.net_out_iis(x)
        return x
    def mapping_iit(self,A):
        x = self.net_in_iit(A)
        for _ in range(self.layers):
            x = self.net_hid_iit(x)
        x = self.net_out_iit(x)
        return x


    def contrastive_loss(self,A,B):
        shared_user, _ = A.size()
        A_abs = A.norm(dim=1)
        B_abs = B.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', A, B) / torch.einsum('i,j->ij', A_abs, B_abs)
        sim_matrix = torch.exp(sim_matrix / 0.1)
        pos_sim = sim_matrix[range(shared_user), range(shared_user)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
        return loss

    def forward(self, source_UV, source_VU, target_UV, target_VU):
        source_user = self.source_user_embedding(self.source_user_index)#self.source_user_embedding(self.source_user_index)
        target_user = self.target_user_embedding(self.target_user_index)
        source_item = self.source_item_embedding(self.source_item_index)
        target_item = self.target_item_embedding(self.target_item_index)

        source_learn_user, source_learn_item,target_learn_user, target_learn_item,connect_learn_user, connect_learn_item= self.st_GNN(source_user, source_item, source_UV, source_VU,target_user, target_item, target_UV, target_VU)

        if self.source_user_embedding.training:

            range1 = torch.randperm(self.opt["shared_user"])
            range2 = torch.randperm(self.opt["target_user_num"] - self.opt["target_shared_user"]) + self.opt["target_shared_user"]
            combined_sequence = torch.cat((range1, range2), dim=0)

            per_stable = combined_sequence[:self.opt["user_batch_size"]].cuda()
            chose_s=self.my_index_select(source_learn_user, per_stable)
            chose_t = self.my_index_select(target_learn_user, per_stable)
            chose_st = self.my_index_select(connect_learn_user, per_stable)

            source_chose_map_target = self.mapping_s(self.calculate_uis(chose_s,chose_st))
            target_chose_map_source = self.mapping_t(self.calculate_uit(chose_t,chose_st))
            # self.contrastive_loss(source_chose_map_target,target_chose_map_source)
            self.critic_loss = self.contrastive_loss(source_chose_map_target, target_chose_map_source)\
                               +self.contrastive_loss(source_chose_map_target, chose_t) + self.contrastive_loss(target_chose_map_source, chose_s)

            source_map_target=self.mapping_s(self.calculate_uis(source_learn_user,connect_learn_user))
            target_map_source = self.mapping_t(self.calculate_uit(target_learn_user,connect_learn_user))

            connect_learn_item_s = connect_learn_item[:self.opt["source_item_num"]]
            connect_learn_item_t = connect_learn_item[self.opt["source_item_num"]:]
            source_item=self.mapping_iis(torch.cat((source_learn_item,connect_learn_item_s),dim=1))
            target_item=self.mapping_iit(torch.cat((target_learn_item,connect_learn_item_t),dim=1))


            return target_map_source, source_item, source_map_target, target_item, source_learn_user, source_learn_item,target_learn_user, target_learn_item,connect_learn_user, connect_learn_item

        else:
            source_map_target = self.mapping_s(self.calculate_uis(source_learn_user,connect_learn_user))
            target_map_source = self.mapping_t(self.calculate_uit(target_learn_user,connect_learn_user))

            connect_learn_item_s = connect_learn_item[:self.opt["source_item_num"]]
            connect_learn_item_t = connect_learn_item[self.opt["source_item_num"]:]
            source_item = self.mapping_iis(torch.cat((source_learn_item, connect_learn_item_s), dim=1))
            target_item = self.mapping_iit(torch.cat((target_learn_item, connect_learn_item_t), dim=1))

            return target_map_source, source_item, source_map_target, target_item
