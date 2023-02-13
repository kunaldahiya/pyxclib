import argparse
import numpy as np
from xclib.evaluation.xc_metrics import _get_topk, fast_precision_with_indices, fast_recall_with_indices, psprecision, compute_inv_propesity, calc_gt_metrics
import scipy.sparse as sp

class XCEvaluator():
    def __init__(self, args):
        super(XCEvaluator, self).__init__()
        self.args = args
        self.k = 100
        self.top_indices = None
        self.smat = None
        self.tst_X_Y = sp.load_npz(args.tst_lbl_pth)
        self.trn_X_Y = sp.load_npz(args.trn_lbl_pth)
        
    def compute_top_indices(self):
        self.smat = sp.load_npz(args.smat_pth)
        if args.top_indices_pth is not None:
            self.top_indices = np.load(args.top_indices_pth)
        else:
            self.top_indices = _get_topk(self.smat, k = self.k)
    
    def build_metrics_dict(precs, recalls, micro_recalls, psps, recall_at_gt, micro_recall_at_gt):
        ks = [1, 3, 5, 10, 50, 100]
        metrics_dict = {}
        for k in ks:
            metrics_dict[f'P@{k}'] = f'{precs[k - 1]:2.2f}'
            metrics_dict[f'R@{k}'] = f'{recalls[k - 1]:2.2f}'
            metrics_dict[f'PSP@{k}'] = f'{psps[k - 1]:2.2f}'
            metrics_dict[f'MicroR@{k}'] = f'{micro_recalls[k - 1]:2.2f}'
        metrics_dict['R@GT'] = f'{recall_at_gt}:2.2f'
        metrics_dict['MicroR@GT'] = f'{micro_recall_at_gt}:2.2f'
        return metrics_dict
    
    def compute_metrics(self):
        
        inv_psp = compute_inv_propesity(self.trn_X_Y, self.A, self.B)
        precs = fast_precision_with_indices(self.top_indices, self.tst_X_Y, self.k)
        recalls, micro_recalls = fast_recall_with_indices(self.top_indices, self.tst_X_Y, self.k)
        psps = psprecision(self.smat, self.tst_X_Y, inv_psp, k=self.k)
        
        # @ GT metrics
        recall_at_gt, micro_recall_at_gt = calc_gt_metrics(self.top_indices, self.tst_X_Y, self.k)
        
        metrics_dict = self.build_metrics_dict(precs, recalls, micro_recalls, psps, recall_at_gt, micro_recall_at_gt)
        
        for metric_name, metric_val in metrics_dict.items():
            print(metric_name, metric_val)
        
parser = argparse.ArgumentParser()
parser.add_argument('--top-indices-pth', type=str, default=None, help='path to the .npy file that contains the indices labels for a data point sorted in decreasing order of relevance')
parser.add_argument('--smat-pth', type=str, help='path to the .npz score mat, recommended that you use top indices as this code will internally compute top indices if it is none')
parser.add_argument('--tst-lbl-pth', type=str, help='path to the tst_X_Y.npz file')
parser.add_argument('--trn-lbl-pth', type=str, help='path to trn_X_Y.npz file')
parser.add_argument('-A', type=float, help='value of A to calculate PSP score')
parser.add_argument('-B', type=float, help='value of B to calculate PSP score')
args = parser.parse_args()

xc_evaluator = XCEvaluator(args)
xc_evaluator.compute_metrics()
