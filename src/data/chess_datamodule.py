from typing import Any, Dict, Optional, Tuple

import os
import torch
import numpy as np
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, TensorDataset
from src.data.components.gen_data import ChessDataLoader, ChessDataGenerator


class ChessDataModule(LightningDataModule):
    """`LightningDataModule` for the MNIST dataset.

    The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples.
    It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a
    fixed-size image. The original black and white images from NIST were size normalized to fit in a 20x20 pixel box
    while preserving their aspect ratio. The resulting images contain grey levels as a result of the anti-aliasing
    technique used by the normalization algorithm. the images were centered in a 28x28 image by computing the center of
    mass of the pixels, and translating the image so as to position this point at the center of the 28x28 field.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        dataset: list = None,
        train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        gen_data: bool = False,
        case_nums: int = 5000,
        force_parse_data: bool = False,
    ) -> None:
        """Initialize a `ChessDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param dataset: The dataset. Defaults to `None`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        :param gen_data: Whether to generate data. Defaults to `False`.
        :param case_nums: The number of cases to generate. Defaults to `5000`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        self.hparams.train_dataset = self.hparams.dataset.train.data_path
        self.hparams.val_dataset = self.hparams.dataset.validation.data_path
        self.hparams.test_dataset = self.hparams.dataset.test.data_path
        if self.hparams.gen_data:
            ChessDataGenerator().generate_data(self.hparams.dataset.train.case_nums, self.hparams.train_dataset)
            ChessDataGenerator().generate_data(self.hparams.dataset.validation.case_nums, self.hparams.val_dataset)
            ChessDataGenerator().generate_data(self.hparams.dataset.test.case_nums, self.hparams.test_dataset)
        else:
            if (
                not os.path.exists(self.hparams.train_dataset)
                or not os.path.exists(self.hparams.val_dataset)
                or not os.path.exists(self.hparams.test_dataset)
            ):
                ChessDataGenerator().convert_data(
                    self.hparams.dataset.raw_data.data_path,
                    self.hparams.train_dataset,
                    self.hparams.val_dataset,
                )
                ChessDataGenerator().generate_data(30, self.hparams.test_dataset)

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        if not self.data_train and not self.data_val and not self.data_test:
            (
                train_data,
                train_labels,
                train_fens,
                train_stockfish_evals,
            ) = ChessDataLoader().load_data(self.hparams.train_dataset)
            val_data, val_labels, val_fens, val_stockfish_evals = ChessDataLoader().load_data(self.hparams.val_dataset)
            test_data, test_labels, test_fens, test_stockfish_evals = ChessDataLoader().load_data(
                self.hparams.test_dataset
            )

            train_data = torch.tensor(train_data).float()
            train_labels = torch.tensor(train_labels).float().view(-1, 1)
            train_stockfish_evals = torch.tensor(train_stockfish_evals).float().view(-1, 1)

            val_data = torch.tensor(val_data).float()
            val_labels = torch.tensor(val_labels).float().view(-1, 1)
            val_stockfish_evals = torch.tensor(val_stockfish_evals).float().view(-1, 1)

            test_data = torch.tensor(test_data).float()
            test_labels = torch.tensor(test_labels).float().view(-1, 1)
            test_stockfish_evals = torch.tensor(test_stockfish_evals).float().view(-1, 1)

            self.data_train = TensorDataset(train_data, train_labels, train_stockfish_evals)
            self.data_val = TensorDataset(val_data, val_labels, val_stockfish_evals)
            self.data_test = TensorDataset(test_data, test_labels, test_stockfish_evals)

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = ChessDataModule()
