import argparse
from typing import Tuple
import datetime
import os
import sys
import numpy as np
import h5py

from pyparsing import Opt
from torch.utils.tensorboard.writer import SummaryWriter
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from src.models.DLPT.dataload import MRIDataset
from src.models.DLPT.model import MRIModel


from src.utils.logconf import logging, formatter, root_logger
from src.utils.util import enumerateWithEstimate


CWD = os.getcwd()
BREAST_MRI_DIR = CWD[:CWD.find('breast_mri')+10]
MODEL_PATH = BREAST_MRI_DIR + '/models/DLPT/'

print(f"{BREAST_MRI_DIR=}")


HDF_PATH = BREAST_MRI_DIR  + '/data/processed/hdf5/'
PERSONAL_HDF_PATH = BREAST_MRI_DIR  + '/personal/data/processed/hdf5/'

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)


# Used for computeBatchLoss and logMetrics to index into metrics_t/metrics_a
METRICS_LABEL_NDX=0
METRICS_PRED_NDX=1
METRICS_LOSS_NDX=2
METRICS_SIZE = 3


class MRITraining:
    def __init__(self, sys_argv = None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--num-workers',
            help='Number of worker processes for background data loading',
            default=8,
            type=int,
        )
        parser.add_argument('--batch-size',
            help='Batch size to use for training',
            default=32,
            type=int,
        )
        parser.add_argument('--epochs',
            help='Number of epochs to train for',
            default=500,
            type=int,
        )

        parser.add_argument('--tb-prefix',
            default='DLPT_model',
            help="Data prefix to use for Tensorboard run. Defaults to chapter.",
        )

        parser.add_argument('--split',
            default=None,
            help="The filename for the .json containing the split to use for training and validation.",
        )

        parser.add_argument('--debug',
            help="Run in debug mode. Only a subset of the data will be used.",
            action='store_true',
            default=False,
        )

        parser.add_argument('--size',
            help="The size of the cube used as sample.",
            default=32,
            type=int,
        )

        parser.add_argument('--balanced',
            help="Ratio of positive to negative samples.",
            default=1,
            type=int,
        )

        parser.add_argument('--augmented',
            help="Augment the training data.",
            action='store_true',
            default=False,
        )

        parser.add_argument('--augment-flip',
            help="Augment the training data by randomly flipping the data left-right, up-down, and front-back.",
            type=bool,
            default=True,
        )

        parser.add_argument('--augment-offset',
            help="Augment the training data by randomly offsetting the data slightly along the X and Y axes.",
            action='store_true',
            default=False,
        )

        parser.add_argument('--augment-scale',
            help="Augment the training data by randomly increasing or decreasing the size of the candidate.",
            action='store_true',
            default=False,
        )

        parser.add_argument('--augment-rotate',
            help="Augment the training data by randomly rotating the data around the head-foot axis.",
            action='store_true',
            default=False,
        )

        parser.add_argument('--augment-noise',
            help="Augment the training data by randomly adding noise to the data.",
            action='store_true',
            default=False,
        )
        parser.add_argument('--save-model',
            help="Save the model after training.",
            action='store_true',
            default=False,
        )

        parser.add_argument('--personal',
            help="Use data from personal folder",
            action='store_true',
        )

        parser.add_argument('--lr',
            help="Learning rate.",
            default=1e-5,
            type=float,
        )

        parser.add_argument('--wd',
            help="Weight decay.",
            default=None,
            type=float,
        )

        parser.add_argument('--model-path',
            help="Path to the model to be loaded.",
            default=None,
            type=str,
        )

        parser.add_argument('--weight',
            help="Weight of the positive class.",
            default=1.,
            type=float,
        )

        parser.add_argument('comment',
            help="Comment suffix for Tensorboard run.",
            nargs='?',
            default='dwlpt',
        )

        

        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        self.trn_writer = None
        self.val_writer = None
        self.totalTrainingSamples_count = 0

        self.augmentation_dict = {}
        if self.cli_args.augmented or self.cli_args.augment_flip:
            self.augmentation_dict['flip'] = True
        if self.cli_args.augmented or self.cli_args.augment_offset:
            self.augmentation_dict['offset'] = 0.1
        if self.cli_args.augmented or self.cli_args.augment_scale:
            self.augmentation_dict['scale'] = 0.2
        if self.cli_args.augmented or self.cli_args.augment_rotate:
            self.augmentation_dict['rotate'] = True
        if self.cli_args.augmented or self.cli_args.augment_noise:
            self.cli_args.augment_noise = 0.05
            self.augmentation_dict['noise'] = 0.05


        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.use_model = self.cli_args.model_path is not None
        if self.use_model:
            self.model_path = MODEL_PATH + self.cli_args.tb_prefix + '/' + self.cli_args.model_path

        self.initLog()

        self.model = self.initModel()
        self.optimizer = self.initOptimizer()

    def initModel(self):
        model = MRIModel(cube_size=self.cli_args.size)
        if self.use_model:
            log.info(f'Loading model from {self.model_path}')
            model.load_state_dict(torch.load(self.model_path))
        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)

        return model

    def initOptimizer(self):
        Optimizer = 'Adam'
        weight_decay = self.cli_args.wd if self.cli_args.wd is not None else 0
        lr = self.cli_args.lr
        if Optimizer == 'SGD':
            momentum = 0.99
            log.info(f'Using {Optimizer} optimizer with lr={lr}, momentum={momentum}, weight_decay={weight_decay}')
            return SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif Optimizer == 'Adam':
            log.info(f'Using {Optimizer} optimizer with lr={lr}')
            return Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
    


    def initTrainDl(self, dataset_hdf):
        train_ds = MRIDataset(
            dataset_hdf,
            val_ratio_int=5,
            isValSet_bool=False,
            ratio_int=int(self.cli_args.balanced),
            augmentation_dict=self.augmentation_dict,
            split_name=self.cli_args.split,
        )

        self.totalPosTrain = len(train_ds.pos_list)

        if not train_ds[0][0].shape[-1] == self.cli_args.size:
            raise ValueError(
                f"Expected cube size {self.cli_args.size} but got {train_ds[0][0].shape[-1]}"
            )

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return train_dl

    def initValDl(self, dataset_hdf):
        val_ds = MRIDataset(
            dataset_hdf,
            val_ratio_int=5,
            isValSet_bool=True,
            split_name=self.cli_args.split,
        )

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return val_dl
    
    def doTraining(self, epoch_ndx, train_dl):
        self.model.train()
        train_dl.dataset.shuffleSamples()
        trnMetrics_g = torch.zeros(
            METRICS_SIZE,
            len(train_dl.dataset),
            device=self.device,
        )

        batch_iter = enumerateWithEstimate(
            train_dl,
            "E{} Training".format(epoch_ndx),
            start_ndx=2*train_dl.num_workers,
        )
        for batch_ndx, batch_tup in batch_iter:
            self.optimizer.zero_grad()

            loss_var = self.computeBatchLoss(
                batch_ndx,
                batch_tup,
                train_dl.batch_size,
                trnMetrics_g
            )

            loss_var.backward()
            self.optimizer.step()

            # This is for adding the model graph to TensorBoard.
            if epoch_ndx == 1 and batch_ndx == 0:
                with torch.no_grad():
                    model = MRIModel()
                    assert self.trn_writer is not None
                    self.trn_writer.add_graph(model, batch_tup[0], verbose=False)

        self.totalTrainingSamples_count += len(train_dl.dataset)

        return trnMetrics_g.to('cpu')
    
    def doValidation(self, epoch_ndx, val_dl):
        with torch.no_grad():
            self.model.eval()
            valMetrics_g = torch.zeros(
                METRICS_SIZE,
                len(val_dl.dataset),
                device=self.device,
            )

            batch_iter = enumerateWithEstimate(
                val_dl,
                "E{} Validation ".format(epoch_ndx),
                start_ndx=2*val_dl.num_workers,
            )
            for batch_ndx, batch_tup in batch_iter:
                self.computeBatchLoss(
                    batch_ndx, batch_tup, val_dl.batch_size, valMetrics_g)

        return valMetrics_g.to('cpu')

    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_g):
        input_t, label_t = batch_tup

        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)

        logits_g, probability_g = self.model(input_g)

        loss_func = nn.CrossEntropyLoss(reduction='none', weight=torch.tensor([1., self.cli_args.weight]).to(self.device))

        loss_g = loss_func(
            logits_g,
            label_g[:,1],
        )
        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + label_t.size(0)

        metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = \
            label_g[:,1].detach()
        metrics_g[METRICS_PRED_NDX, start_ndx:end_ndx] = \
            probability_g[:,1].detach()
        metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = \
            loss_g.detach()

        return loss_g.mean()

    def initTensorboardWriters(self):
        if self.trn_writer is None:
            log_dir = os.path.join('src/runs', self.cli_args.tb_prefix, self.time_str)
            print(f"{log_dir=}")
            self.trn_writer = SummaryWriter(
                log_dir=log_dir + '-trn_cls-' + self.cli_args.comment)
            self.val_writer = SummaryWriter(
                log_dir=log_dir + '-val_cls-' + self.cli_args.comment)

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        filename = 'cubes16stride8' if self.cli_args.size == 16 else 'cubes32stride16'

        if self.cli_args.personal:
            dataset_hdf = h5py.File(PERSONAL_HDF_PATH + f'{filename}.hdf5', 'r')
        else:
            dataset_hdf = h5py.File(HDF_PATH + f'{filename}.hdf5', 'r')

        if self.cli_args.debug:
            tumors_hdf = dataset_hdf['tumors']
            nontumors_hdf = dataset_hdf['nontumors']

            assert isinstance(tumors_hdf, h5py.Group)
            assert isinstance(nontumors_hdf, h5py.Group)

            log.debug('Using only 10 exams')

            nontumors = {key: nontumors_hdf[key] for key in list(nontumors_hdf.keys())[:10]}
            tumors = {key: tumors_hdf[key] for key in list(tumors_hdf.keys())[:10]}
            
            dataset_debug = {'tumors': tumors, 'nontumors': nontumors}
            train_dl = self.initTrainDl(dataset_debug)
            val_dl = self.initValDl(dataset_debug)

        else:
            train_dl = self.initTrainDl(dataset_hdf)
            val_dl = self.initValDl(dataset_hdf)
            
        self.initTensorboardWriters()
        precision, recall = 0, 0

        for epoch_ndx in range(1, self.cli_args.epochs + 1):

            log.info("Epoch {} of {}, {}/{} batches of size {}*{}".format(
                epoch_ndx,
                self.cli_args.epochs,
                len(train_dl),
                len(val_dl),
                self.cli_args.batch_size,
                (torch.cuda.device_count() if self.use_cuda else 1),
            ))

            trnMetrics_t = self.doTraining(epoch_ndx, train_dl)
            self.logMetrics(epoch_ndx, 'trn', trnMetrics_t)

            valMetrics_t = self.doValidation(epoch_ndx, val_dl)
            new_precision, new_recall = self.logMetrics(epoch_ndx, 'val', valMetrics_t)
            
            # # Save the model if the precision and recall are better than the previous best
            # if new_precision > min(precision, 0.975) and new_recall > recall: # 0.975 corresponds to 3/149 false negatives
            #     precision = new_precision
            #     recall = new_recall

            if self.cli_args.save_model:
                model_dir = os.path.join(BREAST_MRI_DIR, f'models/DLPT/{self.cli_args.tb_prefix}/{self.time_str}-{self.cli_args.comment}')
                os.makedirs(model_dir, exist_ok=True)
                filename  = os.path.join(model_dir, f"epoch{epoch_ndx}" +'.pt')
                torch.save(self.model.state_dict(), filename)


        dataset_hdf.close()

        if hasattr(self, 'trn_writer'):
            assert self.trn_writer is not None and self.val_writer is not None
            self.trn_writer.close()
            self.val_writer.close()
    
    def logMetrics(
            self,
            epoch_ndx,
            mode_str,
            metrics_t,
            classificationThreshold=0.5,
    ) -> Tuple[float, float]:
        log.info("E{} {}".format(
            epoch_ndx,
            type(self).__name__,
        ))

        negLabel_mask = metrics_t[METRICS_LABEL_NDX] <= classificationThreshold
        negPred_mask = metrics_t[METRICS_PRED_NDX] <= classificationThreshold

        posLabel_mask = ~negLabel_mask
        posPred_mask = ~negPred_mask

        neg_count = int(negLabel_mask.sum())
        pos_count = int(posLabel_mask.sum())

        neg_correct = int((negLabel_mask & negPred_mask).sum())
        pos_correct = int((posLabel_mask & posPred_mask).sum())

        trueNeg_count = neg_correct = int((negLabel_mask & negPred_mask).sum())
        truePos_count = pos_correct = int((posLabel_mask & posPred_mask).sum())

        falsePos_count = neg_count - neg_correct
        falseNeg_count = pos_count - pos_correct

        metrics_dict = {}
        metrics_dict['loss/all'] = \
            metrics_t[METRICS_LOSS_NDX].mean()
        metrics_dict['loss/neg'] = \
            metrics_t[METRICS_LOSS_NDX, negLabel_mask].mean()
        metrics_dict['loss/pos'] = \
            metrics_t[METRICS_LOSS_NDX, posLabel_mask].mean()

        metrics_dict['correct/all'] = (pos_correct + neg_correct) \
            / np.float32(metrics_t.shape[1]) * 100
        metrics_dict['correct/neg'] = neg_correct / np.float32(neg_count) * 100
        metrics_dict['correct/pos'] = pos_correct / np.float32(pos_count) * 100

        if mode_str == 'trn':
            truePos_count = int(truePos_count * self.totalPosTrain / pos_count)
            falseNeg_count = int(falseNeg_count * self.totalPosTrain / pos_count)
            pos_count = self.totalPosTrain
            pos_correct = truePos_count

        precision = metrics_dict['pr/precision'] = \
            truePos_count / np.float32(truePos_count + falsePos_count)
        recall    = metrics_dict['pr/recall'] = \
            truePos_count / np.float32(truePos_count + falseNeg_count - 1e-7)

        metrics_dict['pr/f1_score'] = \
            2 * (precision * recall) / (precision + recall)

        
        log.info(
            ("E{} {:8} {loss/all:.4f} loss, "
                 + "{correct/all:-5.1f}% correct, "
                 + "{pr/precision:.4f} precision, "
                 + "{pr/recall:.4f} recall, "
                 + "{pr/f1_score:.4f} f1 score"
            ).format(
                epoch_ndx,
                mode_str,
                **metrics_dict,
            )
        )
        log.info(
            ("E{} {:8} {loss/neg:.4f} loss, "
                 + "{correct/neg:-5.1f}% correct ({neg_correct:} of {neg_count:})"
            ).format(
                epoch_ndx,
                mode_str + '_neg',
                neg_correct=neg_correct,
                neg_count=neg_count,
                **metrics_dict,
            )
        )
        log.info(
            ("E{} {:8} {loss/pos:.4f} loss, "
                 + "{correct/pos:-5.1f}% correct ({pos_correct:} of {pos_count:})"
            ).format(
                epoch_ndx,
                mode_str + '_pos',
                pos_correct=pos_correct,
                pos_count=pos_count,
                **metrics_dict,
            )
        )

        writer = getattr(self, mode_str + '_writer')

        for key, value in metrics_dict.items():
            writer.add_scalar(key, value, self.totalTrainingSamples_count)

        writer.add_pr_curve(
            'pr',
            metrics_t[METRICS_LABEL_NDX],
            metrics_t[METRICS_PRED_NDX],
            self.totalTrainingSamples_count,
        )

        bins = [x/50.0 for x in range(51)]

        negHist_mask = negLabel_mask & (metrics_t[METRICS_PRED_NDX] > 0.01)
        posHist_mask = posLabel_mask & (metrics_t[METRICS_PRED_NDX] < 0.99)

        #if negHist_mask.any():
        #    writer.add_histogram(
        #        'is_neg',
        #        metrics_t[METRICS_PRED_NDX, negHist_mask],
        #        self.totalTrainingSamples_count,
        #        bins=bins,
        #    )
        #if posHist_mask.any():
        #    writer.add_histogram(
        #        'is_pos',
        #        metrics_t[METRICS_PRED_NDX, posHist_mask],
        #        self.totalTrainingSamples_count,
        #        bins=bins,
        #    )

        assert isinstance(precision, float)
        assert isinstance(recall, float)

        return precision, recall

    def initLog(self):
        log_dir = BREAST_MRI_DIR + '/src/logs/' + self.cli_args.tb_prefix
        os.makedirs(log_dir, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(log_dir, self.time_str + '-' + self.cli_args.comment + '.log'), mode='w')
        fileHandler.setFormatter(formatter)
        fileHandler.setLevel(logging.INFO)

        root_logger.addHandler(fileHandler)

if __name__ == '__main__':
    MRITraining().main()

    ## debug mode
    #MRITraining(['--size 16']).main()
