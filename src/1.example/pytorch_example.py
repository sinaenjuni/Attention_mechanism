import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, Normalize

import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy

from torchmetrics import functional as FM


# 하나의 데이터 샘플을 만드는 역할
class MNISTDataset(Dataset):
    """
    a single example:
        python object --> Tensor
    """

    def __init__(self, train=True):
        transforms = Compose([ToTensor(),
                              Normalize([0.5], [0.5])])
        if train:
            self.data = MNIST(root="./", train=True, download=True, transform=transforms)
        else:
            self.data = MNIST(root='./', train=False, download=True, transform=transforms)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx][0]
        image = torch.flatten(image)
        label = self.data[idx][1]

        return [image, label]

# dataset = MNISTDataset(True)
# print(dataset[0][0].size())


# 하나의 데이터 셋들을 배치 단위로 만드는 역할
class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.save_hyperparameters()
        # save_hyperparameters()를 해야 log에 parameter들이 기록되고,
        # 동시에 self.hparams로 변수 접근이 가능하다.

        ## prepare_data(), setup() 이라는 특별한 함수가 있음, 이건 알아서 찾아보기

        self.batch_size = batch_size
        # self.batch_size = self.hparams.batch_size

        self.all_train_dataset = MNISTDataset(True)
        self.all_test_dataset = MNISTDataset(False)

        N = len(self.all_train_dataset)
        tr = int(N * 0.8)
        va = N - tr
        self.train_dataset, self.valid_dataset = random_split(self.all_train_dataset, [tr, va])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=4, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.all_test_dataset, batch_size=self.batch_size, num_workers=4, persistent_workers=True)
    # def teardown(self):
    #     ...

# data_module = MNISTDataModule().train_dataloader()

# for i in data_module:
#     print(i)


# 실제 신경망 모델을 구현하는 부분
# 아래에 구현되어 있는 함수들은 모두 pl에 정의되어 있는 함수들로 기능에 따라 자동으로 매핑되기 때문에
# 구현 해주어야 한다.
class MLP_MNIST_Classifier(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        """
        self 다음에 오는 모든 argument들을 저장
        self.hparams.learning_rate 를 통해서 참조가 가능
        또한 check point에 자동을 저장된다.
        """

        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def training_step(self, batch, batch_idx):
        image, label = batch
        label_logits = self(image)  # forward를 암시적으로 호출
        loss = self.criterion(label_logits, label.long())
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)  # tensorboard와 자동으로 연결
        return loss  # 반드시 loss가 리턴

    def validation_step(self, batch, batch_idx):
        image, label = batch
        label_logit = self(image)
        loss = self.criterion(label_logit, label.long())
        prob = F.softmax(label_logit, dim=-1)
        acc = FM.accuracy(prob, label)

        metrics = {'val_acc': acc, 'val_loss': loss}
        self.log_dict(metrics)
        return metrics

    def validation_step_end(self, val_step_outputs):
        val_acc = val_step_outputs['val_acc'].cpu()
        val_loss = val_step_outputs['val_loss'].cpu()

        self.log('validation_acc', val_acc, prog_bar=True)
        self.log('validation_loss', val_loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        image, label = batch
        label_logits = self(image)
        loss = self.criterion(label_logits, label.long())
        prob = F.softmax(label_logits, dim=-1)
        acc = FM.accuracy(prob, label)
        metrics = {'test_acc': acc, 'test_loss': loss}
        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.hparams.learning_rate)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("MLP_MNIST_Classifier")
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parent_parser


from argparse import ArgumentParser
from pytorch_lightning.callbacks import EarlyStopping


def cli_main():
    pl.seed_everything(1234)
    # 다른 환경에서도 동일한 성능을 보장하기 위한 random seed 초기화
    # 맹신하지 말것!

    parser = ArgumentParser()
    parser.add_argument("--batch_size", default=200, type=int)
    # 학습에 사용되는 파라미터 파싱

    parser = MLP_MNIST_Classifier.add_model_specific_args(parser)
    # model에 있는 hyper-parameter를 parser에 등록하는 과정
    # 모델에 정의된 hyper-parameter를 가져오는 과정이다.

    # parser = pl.Trainer.add_argparse_args(parser)
    # pytorch lightning에 hyper-parameter 전달

    args = parser.parse_args('')

    dm = MNISTDataModule.from_argparse_args(args)
    # 데이터 모듈에 파라미터 전달
    print(dm.batch_size)
    model = MLP_MNIST_Classifier(args.learning_rate)
    # 학습 모델 정의

    trainer = pl.Trainer(max_epochs=1,
                         strategy=DDPStrategy(find_unused_parameters=False),
                         accelerator='gpu',
                         callbacks=[EarlyStopping(monitor='val_loss')],
                         gpus=-1
                         )  # 0이면 no gpu, 1이상이면 해당 개수만큼의 gpu사용
    # 학습 과정에서 사용되는 파라미터들 정의

    trainer.fit(model, datamodule=dm)
    # 학습 시작

    result = trainer.test(model, dataloaders=dm.test_dataloader())
    # 테스트 시작

    print(result)


if __name__ == '__main__':
    cli_main()

