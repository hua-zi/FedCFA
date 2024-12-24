import os

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from fedlab.leaf.dataloader import get_LEAF_all_test_dataloader

import logging
from pathlib import Path
from fedlab.leaf.pickle_dataset import PickleDataset

BASE_DIR = Path(__file__).resolve().parents[2]

class PartitionLEAF:
    """
     Args:
        dataset (str):  dataset name string to get dataloader
        client_id (int, optional): assigned client_id to get dataloader for this client. Defaults to 0
        batch_size (int, optional): the number of batch size for dataloader. Defaults to 128
        data_root (str): path for data saving root.
                        Default to None and will be modified to the datasets folder in FedLab: "fedlab-benchmarks/datasets"
        pickle_root (str): path for pickle dataset file saving root.
                        Default to None and will be modified to Path(__file__).parent / "pickle_datasets"
    """
    def __init__(self,
                 dataname,
                 data_root, 
                 pickle_root,
                 num_clients=0, 
                 shuffle=False,
                 transform=None,
                 target_transform=None) -> None:
        self.dataname = dataname
        self.data_root = os.path.expanduser(data_root)
        self.pickle_root = os.path.expanduser(pickle_root)
        self.num_clients = num_clients
        self.shuffle = shuffle
        # self.transform = transform
        # self.targt_transform = target_transform


    def get_dataset(self, cid, type="train"):
        """Load subdataset for client with client ID ``cid`` from local file.

        Args:
             cid (int): client id
             type (str, optional): Dataset type, can be ``"train"``, ``"val"`` or ``"test"``. Default as ``"train"``.

        Returns:
            Dataset
        """
         # Need to run leaf/gen_pickle_dataset.sh to generate pickle files for this dataset firstly
        pdataset = PickleDataset(dataset_name=self.dataname, data_root=self.data_root, pickle_root=self.pickle_root)
        try:
            if type == "train":
                dataset = pdataset.get_dataset_pickle(dataset_type="train", client_id=cid)
            else:
                dataset = pdataset.get_dataset_pickle(dataset_type="test", client_id=cid)
        except FileNotFoundError:
            logging.error(f"""
                            No built PickleDataset json file for {self.dataname}-client {cid} in {pdataset.pickle_root.resolve()}
                            Please run `{BASE_DIR / 'leaf/gen_pickle_dataset.sh'} to generate {self.dataname} pickle files firstly!` 
                            """)
            
        return dataset

    def get_dataloader(self, cid, batch_size=None, type="train"):
        """Return dataload for client with client ID ``cid``.

        Args:
            cid (int): client id
            batch_size (int, optional): batch size in DataLoader.
            type (str, optional): Dataset type, can be ``"train"``, ``"val"`` or ``"test"``. Default as ``"train"``.
        """
        dataset = self.get_dataset(cid, type)
        if type=="train":
            data_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                drop_last=self.shuffle)  # avoid train dataloader size 0
        else:
            data_loader = DataLoader(
                dataset,
                batch_size=len(dataset),
                shuffle=self.shuffle)
            
        return data_loader
    
    def get_all_test_dataloader(self, batch_size=128):
        return get_LEAF_all_test_dataloader(dataset=self.dataname,
                                            batch_size=batch_size,
                                            data_root=self.data_root,
                                            pickle_root=self.pickle_root)




