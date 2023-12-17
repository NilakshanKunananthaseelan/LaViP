import os
import random
import pickle

from mm_dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from mm_dassl.utils import mkdir_if_missing, listdir_nohidden, write_json, read_json

from .oxford_pets import OxfordPets


@DATASET_REGISTRY.register()
class PCAM(DatasetBase):
    dataset_dir = 'pcam'

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = self.dataset_dir
        self.train_image_dir = os.path.join(self.dataset_dir, 'train')
        self.test_image_dir = os.path.join(self.dataset_dir, 'test')
        self.val_image_dir = os.path.join(self.dataset_dir, 'val')

        num_shots = cfg.DATASET.NUM_SHOTS
        self.split_fewshot_dir = os.path.join(self.dataset_dir, 'split_fewshot')
        if num_shots >= 1:mkdir_if_missing(self.split_fewshot_dir)
        train = self.read_data(self.train_image_dir)
        test = self.read_data(self.test_image_dir)
        val = self.read_data(self.train_image_dir)

        
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")

            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}

                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)

        super().__init__(train_x=train, val=val, test=test)

    def read_data(self,  split_dir):
        split_dir = os.path.join(self.image_dir, split_dir)
        folders = sorted(f.name for f in os.scandir(split_dir) if f.is_dir())
        items = []
        

        classnames = {'negative': 'lymph node', 'positive': 'lymph node containing metastatic tumor tissue'}
        # classnames = {'negative': 'not containing', 'positive': 'containing'}

        for label, folder in enumerate(folders):
            
            imnames = listdir_nohidden(os.path.join(split_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(split_dir, folder, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items
