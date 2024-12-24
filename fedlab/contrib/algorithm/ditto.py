import torch
import tqdm
from tqdm import *
from copy import deepcopy

from .basic_server import SyncServerHandler
from .basic_client import SGDClientTrainer, SGDSerialClientTrainer
# from ...utils.serialization import SerializationTool
from ...utils import Aggregators, SerializationTool

##################
#
#      Server
#
##################


class DittoServerHandler(SyncServerHandler):
    """Ditto server acts the same as fedavg server."""
    None


##################
#
#      Client
#
##################


class DittoSerialClientTrainer(SGDSerialClientTrainer):
    def __init__(self, model, num, cuda=False, device=None, logger=None, personal=True) -> None:
        super().__init__(model, num, cuda, device, logger, personal)
        self.ditto_gmodels = []
        print('------ditto client')

    def setup_dataset(self, dataset):
        return super().setup_dataset(dataset)
    
    def setup_optim(self, epochs, batch_size, lr):
        return super().setup_optim(epochs, batch_size, lr)

    def local_process(self, payload, id_list):
        global_model = payload[0]
        # for id in tqdm(id_list):
        for id in id_list:
            # self._LOGGER.info("Local process is running. Training client {}".format(id))
            train_loader = self.dataset.get_dataloader(id, batch_size=self.batch_size)
            # self.parameters[id], glb_model  = self._train_alone(global_model, self.local_models[id], train_loader)
            self.parameters[id], glb_model  = self.train(global_model, self.parameters[id], train_loader) # 每个客户端只获取自己的模型参数，更新自己和全局的模型参数
            self.ditto_gmodels.append(deepcopy(glb_model)) # 记录每一个客户更新的全局模型

    @property
    def uplink_package(self):
        ditto_gmodels = deepcopy(self.ditto_gmodels)
        self.ditto_gmodels = []
        return [[parameter] for parameter in ditto_gmodels] # 转为列表的列表

    def train(self, global_model_parameters, local_model_parameters, train_loader):
        criterion = torch.nn.CrossEntropyLoss()
        SerializationTool.deserialize_model(self._model, global_model_parameters)
        self._model.train()
        for ep in range(self.epochs):
            for data, label in train_loader:
                if self.cuda:
                    data, label = data.cuda(self.device), label.cuda(self.device)

                preds = self._model(data)
                loss = criterion(preds,label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step() # 在本地下载全局参数并训练

        updated_glb_models = deepcopy(self.model_parameters) # 第一次训练的模型参数，用来更新全局参数

        frz_model = deepcopy(self._model) 
        SerializationTool.deserialize_model(frz_model, global_model_parameters) # 一个使用全局参数的模型

        SerializationTool.deserialize_model(self._model, local_model_parameters) # 使用本地参数的模型，在第二次训练时更新
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self._model.parameters(), lr=self.lr)

        self._model.train()
        frz_model.eval()
        for ep in range(self.epochs):
            for data, label in train_loader:
                if self.cuda:
                    data, label = data.cuda(self.device), label.cuda(self.device)

                preds = self._model(data)
                l1 = criterion(preds,label)
                l2 = 0.0 # 再次训练本地参数，并得到与训练前全局参数的距离，作为损失
                for w0, w in zip(frz_model.parameters(), self._model.parameters()):
                    l2 += torch.sum(torch.pow(w - w0, 2))

                # loss = l1 + 0.5 * self.args.mu * l2
                loss = l1 + 0.5 * 0.1 * l2
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        # 全局影响本地，本地不影响全局，所以全局模型和fedavg一样
        updated_glb_models = Aggregators.fedavg_aggregate([self.model_parameters, updated_glb_models])
        return self.model_parameters, updated_glb_models # 第二次训练参数（本地用），第一次训练参数（全局用）
