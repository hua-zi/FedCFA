import sys
sys.path.append('../')
from fedlab.contrib.algorithm.basic_server import SyncServerHandler
from fedlab.contrib.algorithm.basic_client import SGDClientTrainer, SGDSerialClientTrainer
from fedlab.utils.aggregator import Aggregators
from fedlab.utils.serialization import SerializationTool
from fedlab.utils import Logger
from copy import deepcopy
import torch
import torch.nn.functional as F

from typing import List
import copy
import torch.autograd as autograd

##################
#
#      Server
#
##################


class FedCFServerHandler(SyncServerHandler):
    """FedCF server handler."""
    def __init__(self,                 
                 model: torch.nn.Module,
                 global_round: int,
                 sample_ratio: float,
                 cuda: bool = False,
                 device: str=None,
                 logger: Logger = None,
                 Xg = None,
                 Yg = None):
        super(FedCFServerHandler, self).__init__(model, global_round, sample_ratio, cuda, device, logger)

        self.Xg = Xg
        self.Yg = Yg

    @property
    def downlink_package(self) -> List[torch.Tensor]:
        """Property for manager layer. Server manager will call this property when activates clients."""
        return [self.model_parameters, self.Xg, self.Yg]

    def global_update(self, buffer):
        parameters_list = [ele[0] for ele in buffer]
        weights = [ele[1] for ele in buffer]
        serialized_parameters = Aggregators.fedavg_aggregate(
            parameters_list, weights)
        SerializationTool.deserialize_model(self._model, serialized_parameters)
        self.Xg = [ele[-2] for ele in buffer]
        self.Yg = [ele[-1] for ele in buffer]


##################
#
#      Client
#
##################


class FedCFSerialClientTrainer(SGDSerialClientTrainer):
    """Federated client with local SGD solver."""
    def __init__(self, model, num_clients, cuda=False, device=None, logger=None, personal=False, topk=None, num_classes=10, fedcfa_rate=None):
        super().__init__(model, num_clients, cuda, device, logger, personal)
        self.num_classes = num_classes
        self.topk = topk
        self.fedcfa_rate = fedcfa_rate
    
    @property
    def uplink_package(self):
        package = deepcopy(self.cache)
        self.cache = []
        return package

    def local_process(self, payload, id_list):
        model_parameters = payload[0]
        Xg, Yg = payload[-2], payload[-1]   # list of tensors 
        if Xg != None:
            Xg = torch.cat(Xg, dim=0)   # tensors, torch.Size([num, 3,32,32]
            Yg = torch.cat(Yg, dim=0)   # tensors, torch.Size([num, num_classes]) 
        for id in id_list:
            data_loader = self.dataset.get_dataloader(id, self.batch_size)
            pack = self.train(model_parameters, data_loader, Xg, Yg)
            self.cache.append(pack)

    def calculate_mean_data(self, train_loader, mean_batch: int):
        data, label = [], []
        for X, y in train_loader:
            data.append(X)
            label.append(y)
        data = torch.cat(data, dim=0)
        label = torch.cat(label, dim=0)

        random_ids = torch.randperm(len(data))
        data, label = data[random_ids], label[random_ids]
        data = torch.split(data, mean_batch)
        label = torch.split(label, mean_batch)

        self.Xmean, self.ymean = [], []
        for d, l in zip(data, label):
            self.Xmean.append(torch.mean(d, dim=0))
            self.ymean.append(torch.mean(F.one_hot(l, num_classes=self.num_classes).to(dtype=torch.float32), dim=0))
        self.Xmean = torch.stack(self.Xmean, dim=0)
        self.ymean = torch.stack(self.ymean, dim=0)
        return self.Xmean, self.ymean
  
    def train(self, model_parameters, train_loader, Xg, Yg):
        self.set_model(model_parameters)
        model_g = copy.deepcopy(self.model)
        self._model.train()

        data_size = 0
        for _ in range(self.epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                if self.cuda:
                    data = data.cuda(self.device)
                    target = target.cuda(self.device)

                output, latent = self.model(data)   

                loss_cls = self.criterion(output, target)

                if Xg == None:
                    loss = loss_cls
                else:
                    data_g = latent.clone().detach()
                    target_g = copy.deepcopy(target)
                    data_g.requires_grad_()
                    out_g = torch.sum(model_g.decoder(data_g) * F.one_hot(target_g, num_classes=self.num_classes).to(dtype=torch.float32))/len(data_g)
                    gradients_g = autograd.grad(outputs=out_g, inputs=data_g)[0]
                    mask = torch.mean(gradients_g,dim=(-1,-2)).view(gradients_g.shape[0],-1)

                    mask_p = torch.ones_like(mask,device=self.device)
                    mask_p.scatter_(-1,mask.topk(k=self.topk, dim=-1, largest=False).indices,0)
                    mask_p = mask_p[...,None,None].repeat(1,1,data_g.shape[-2],data_g.shape[-1])
                    
                    mask_n = torch.ones_like(mask,device=self.device)
                    mask_n.scatter_(-1,mask.topk(k=self.topk, dim=-1).indices,0)
                    mask_n = mask_n[...,None,None].repeat(1,1,data_g.shape[-2],data_g.shape[-1])
                    
                    random_ids = torch.randint(0,len(Xg),data.shape[:1])
                    Xg_sel, Yg_sel = Xg[random_ids].cuda(self.device), Yg[random_ids].cuda(self.device)
                    lat_sel = self.model.encoder(Xg_sel)

                    data_p = latent.clone().detach()
                    lat_p = lat_sel.clone().detach()
                    data_p = mask_p * data_p + (1-mask_p) * lat_p
                    target_p = copy.deepcopy(target)
                    target_p = F.one_hot(target_p, num_classes=self.num_classes).to(dtype=torch.float32)
                    # target_p = torch.mean(mask_p) * target_p + torch.mean(1-mask_p) * Yg_sel
                    output_p = self.model.decoder(data_p)
                    loss_p = self.criterion(output_p, target_p)

                    data_n = latent.clone().detach()
                    lat_n = lat_sel.clone().detach()
                    data_n = mask_n * data_n + (1-mask_n) * lat_n
                    target_n = copy.deepcopy(target)
                    target_n = F.one_hot(target_n, num_classes=self.num_classes).to(dtype=torch.float32)
                    target_n = torch.mean(mask_n) * target_n + torch.mean(1-mask_n) * Yg_sel
                    output_n = self.model.decoder(data_n)
                    loss_n = self.criterion(output_n, target_n)

                    loss = self.fedcfa_rate[0]*loss_cls + self.fedcfa_rate[1]*loss_p + self.fedcfa_rate[2]*loss_n

                data_size += len(target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self._model.eval()
        Xc, Yc = self.calculate_mean_data(train_loader, 128)

        return [self.model_parameters, data_size, Xc, Yc]