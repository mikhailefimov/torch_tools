import io
import random
from unittest import TestCase, mock

import numpy as np
import torch
import torchvision
from torch.backends import cudnn

from tests.random_image_dataset import RandomImageDataset
from tools.model_faster_rcnn import ModelRCNN
from tools.trainer import Trainer


class TimeMock:
    _mocked_timer = 1

    @staticmethod
    def mock_time():
        TimeMock._mocked_timer += 1
        return TimeMock._mocked_timer


def reseed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@mock.patch('time.monotonic', mock.MagicMock(side_effect=TimeMock.mock_time))
class TestFasterRCNN(TestCase):

    def test_faster_rcnn(self):
        reseed()
        NCLASSES = 2
        dataset = RandomImageDataset()
        if torch.cuda.is_available() and torch.cuda.get_device_properties("cuda:0").total_memory > 8 * (1024 ** 3):
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False,
                                                                     num_classes=NCLASSES).to(device)

        trainer = Trainer(ModelRCNN(model, device),
                          lambda model: torch.optim.SGD(model.model.parameters(), lr=3e-4),
                          train_dataset=dataset, val_split=1, batch_size=2, outdir=None)
        trainer.train(2, start_lr=1e-2, end_lr=1e-3)

        # Check logged results

        with io.StringIO() as out:
            trainer.epoch_logger.print_logs(file=out)
            trainer.batch_logger.print_logs(file=out)
            s = out.getvalue()
        self.assertEqual(s, (
            "--------------------------------------------------------------\n"
            "Training epochs 0=>2  {'start_lr': 0.01, 'end_lr': 0.001}\n"
            "--------------------------------------------------------------\n"
            "Train epoch	[1/2]	(0:00:04<=0:00:04s)	LR:0.010000, loss_cls:0.3209, loss_box:0.0150, loss_obj:0.4337, loss_rpn_box:0.1177, mAP:0.000\n"
            "Train epoch	[2/2]	(0:00:08<=0:00:00s)	LR:0.001000, loss_cls:0.2618, loss_box:0.0674, loss_obj:0.0696, loss_rpn_box:0.1623, mAP:0.250\n"
            "Training epochs 0=>2  {'start_lr': 0.01, 'end_lr': 0.001} Total time: 0:00:08 (4.0000 s / it)\n"
            "batch	[1/2]	(0:00:01<=0:00:01s)	loss_cls:0.5228, loss_box:0.0154, loss_obj:0.6656, loss_rpn_box:0.0206\n"
            "batch	[2/2]	(0:00:02<=0:00:00s)	loss_cls:0.1189, loss_box:0.0145, loss_obj:0.2019, loss_rpn_box:0.2149\n"
            " Total time: 0:00:02 (1.0000 s / it)\n"
            "batch	[1/2]	(0:00:01<=0:00:01s)	loss_cls:0.2161, loss_box:0.0034, loss_obj:0.0581, loss_rpn_box:0.2668\n"
            "batch	[2/2]	(0:00:02<=0:00:00s)	loss_cls:0.3076, loss_box:0.1313, loss_obj:0.0811, loss_rpn_box:0.0578\n"
            " Total time: 0:00:02 (1.0000 s / it)\n"))
