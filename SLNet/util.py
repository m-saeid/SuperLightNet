import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder.encodings import *



import pandas as pd
import os

def log_experiment(results, excel_path='experiment_results.xlsx'):
    df_new = pd.DataFrame([results])
    # If file exists, append the new results to it
    if os.path.exists(excel_path):
        df_existing = pd.read_excel(excel_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_excel(excel_path, index=False)
    else:
        df_new.to_excel(excel_path, index=False)
'''
# Example usage at end of training:
results = {
    'Task': 'Classification (ModelNet40)',
    'Point_Number': 4096,
    'Embed': [9,16,'mlp']}#args.embed,
    #'Epochs': args.epoch,
    #'Batch_Size': args.batch_size,
    #'Learning_Rate': args.learning_rate,
    #'Checkpoint_Path': args.checkpoint,
    #'Accuracy': final_accuracy,  # replace with your computed metric
    #'Model_Params': model_params,  # total number of parameters (e.g., sum(p.numel() for p in model.parameters())
    #'Message': args.msg
#}
log_experiment(results)

'''


################# Classification ################

class Classifier(nn.Module):
    def __init__(self, last_channel, classifier_mode, num_cls):
        super(Classifier, self).__init__()

        sigma = 0.3
        alpha = 100.0
        beta = 1.0

        if last_channel > 128:
            if classifier_mode == "mlp":
                l1 = nn.Linear(last_channel, 256)
                l2 = nn.Linear(256, 128)
                l3 = nn.Linear(128, num_cls)
            self.classifier = nn.Sequential(
                l1,
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                l2,
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                l3
            )
        else:
            if classifier_mode == "mlp":
                l1 = nn.Linear(last_channel, 128)
                l2 = nn.Linear(128, 64)
                l3 = nn.Linear(64, num_cls)
            elif classifier_mode == 'fourier':
                l1 = FourierPositionalEncoding(input_dim=last_channel, output_dim=128, num_frequencies=16, scale=1.0)
                l2 = FourierPositionalEncoding(input_dim=128, output_dim=64, num_frequencies=16, scale=1.0)
                l3 = FourierPositionalEncoding(input_dim=64, output_dim=num_cls, num_frequencies=16, scale=1.0)
            elif classifier_mode == 'scaled_fourier':
                l1 = ScaledFourierPositionalEncoding(input_dim=last_channel, output_dim=128, alpha=alpha, beta=beta)
                l2 = ScaledFourierPositionalEncoding(input_dim=128, output_dim=64, alpha=alpha, beta=beta)
                l3 = ScaledFourierPositionalEncoding(input_dim=64, output_dim=num_cls, alpha=alpha, beta=beta)
            elif classifier_mode == 'gaussian':
                l1 = GaussianPositionalEncoding(input_dim=last_channel, output_dim=128, sigma=sigma)
                l2 = GaussianPositionalEncoding(input_dim=128, output_dim=64, sigma=sigma)
                l3 = GaussianPositionalEncoding(input_dim=64, output_dim=num_cls, sigma=sigma)
            elif classifier_mode == 'harmonic':
                l1 = HarmonicPositionalEncoding(input_dim=last_channel, output_dim=128, num_frequencies=4)
                l2 = HarmonicPositionalEncoding(input_dim=128, output_dim=64, num_frequencies=4)
                l3 = HarmonicPositionalEncoding(input_dim=64, output_dim=num_cls, num_frequencies=4)
            elif classifier_mode == 'mlp2':
                l1 = MLPPositionalEncoding(input_dim=last_channel, output_dim=128, hidden_dim=64)
                l2 = MLPPositionalEncoding(input_dim=128, output_dim=64, hidden_dim=64)
                l3 = MLPPositionalEncoding(input_dim=64, output_dim=num_cls, hidden_dim=64)
            #elif classifier_mode == 'learnable':
            #    l1 = LearnablePositionalEmbedding(num_points, output_dim)
            #    l2 = LearnablePositionalEmbedding(num_points, output_dim)
            #    l3 = LearnablePositionalEmbedding(num_points, output_dim)
            elif classifier_mode == 'relative':
                l1 = RelativePositionalEncoding(input_dim=last_channel, output_dim=128, hidden_dim=64)
                l2 = RelativePositionalEncoding(input_dim=128, output_dim=64, hidden_dim=64)
                l3 = RelativePositionalEncoding(input_dim=64, output_dim=num_cls, hidden_dim=64)
            elif classifier_mode == 'linear_coord':
                l1 = LinearCoordinateEmbedding(input_dim=last_channel, output_dim=128)
                l2 = LinearCoordinateEmbedding(input_dim=128, output_dim=64)
                l3 = LinearCoordinateEmbedding(input_dim=64, output_dim=num_cls)
            else:
                raise Exception(f"classifier_mode!!! {classifier_mode}")
            
            self.classifier = nn.Sequential(
                l1,
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                l2,
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                l3
            )
    def forward(self, x):
        return self.classifier(x)
    

########################### Segmentation #################################
import numpy as np
import torch
import torch.nn.functional as F


def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1) # gold is the groudtruth label in the dataloader

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)  # the number of feature_dim of the ouput, which is output channels

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss


# create a file and write the text into it:
class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda(non_blocking=True)
    return new_y


def compute_overall_iou(pred, target, num_classes):
    shape_ious = []
    pred = pred.max(dim=2)[1]    # (batch_size, num_points)  the pred_class_idx of each point in each sample
    pred_np = pred.cpu().data.numpy()

    target_np = target.cpu().data.numpy()
    for shape_idx in range(pred.size(0)):   # sample_idx
        part_ious = []
        for part in range(num_classes):   # class_idx! no matter which category, only consider all part_classes of all categories, check all 50 classes
            # for target, each point has a class no matter which category owns this point! also 50 classes!!!
            # only return 1 when both belongs to this class, which means correct:
            I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
            # always return 1 when either is belongs to this class:
            U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))

            F = np.sum(target_np[shape_idx] == part)

            if F != 0:
                iou = I / float(U)    #  iou across all points for this class
                part_ious.append(iou)   #  append the iou of this class
        shape_ious.append(np.mean(part_ious))   # each time append an average iou across all classes of this sample (sample_level!)
    return shape_ious   # [batch_size]