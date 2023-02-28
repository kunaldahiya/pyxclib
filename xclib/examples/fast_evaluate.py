import argparse
import numpy as np
from xclib.evaluation.xc_metrics import precision, recall, psprecision, compute_inv_propesity, recall_at_gt, micro_recall_at_gt, ndcg
import scipy.sparse as sp

class XCEvaluator():
    def __init__(self, args):
        super(XCEvaluator, self).__init__()
        self.k = args.k
        self.pad_val = args.pad_val
        self.top_indices = None
        self.smat = None
        self.tst_X_Y = sp.load_npz(args.tst_lbl_pth)
        self.trn_X_Y = sp.load_npz(args.trn_lbl_pth)
        self.A = args.A
        self.B = args.B
    
    def build_metrics_dict(self, precs, recalls, psps, ndcgs, recall_at_gt, micro_recall_at_gt):
        ks = [1, 3, 5, 10, 50, 100]
        metrics_dict = {}
        for k in ks:
            metrics_dict[f'P@{k}'] = f'{precs[k - 1] :.5f}'
            metrics_dict[f'R@{k}'] = f'{recalls[k - 1] :.5f}'
            metrics_dict[f'PSP@{k}'] = f'{psps[k - 1] :.5f}'
            metrics_dict[f'nDCG@{k}'] = f'{ndcgs[k - 1]:.5f}'
        metrics_dict['R@GT'] = f'{recall_at_gt :.5f}'
        metrics_dict['MicroR@GT'] = f'{micro_recall_at_gt :.5f}'
        return metrics_dict
    
    def compute_metrics(self):
        self.smat = sp.load_npz(args.smat_pth)
        inv_psp = compute_inv_propesity(self.trn_X_Y, self.A, self.B)
        precs = precision(self.smat, self.tst_X_Y, self.k)
        print("Finished Precision computation")
        recalls = recall(self.smat, self.tst_X_Y, self.k)
        print("Finished Recall computation")
        psps = psprecision(self.smat, self.tst_X_Y, inv_psp, self.k) 
        print("Finished PSP computation")
        ndcgs = ndcg(self.smat, self.tst_X_Y, self.k)
        print("Finished nDCG computation")
        rec_at_gt = recall_at_gt(self.smat, self.tst_X_Y, pad_val=self.pad_val)
        micro_rec_at_gt = micro_recall_at_gt(self.smat, self.tst_X_Y, pad_val=self.pad_val)
        print("Finished @ GT metrics computation")
        metrics_dict = self.build_metrics_dict(precs, recalls, psps, ndcgs, rec_at_gt, micro_rec_at_gt)
        
        for metric_name, metric_val in metrics_dict.items():
            print(metric_name, float(metric_val) * 100)

'''
Sample Command: python fast_evaluate.py --smat-pth <path-to-smat> --tst-lbl-pth <path-to-test-score-mat> --trn-lbl-pth <path-to-train-score-mat> -A 0.6 -B 2.6 --pad-val 100000000
'''
parser = argparse.ArgumentParser()
parser.add_argument('--top-indices-pth', type=str, default=None, help='path to the .npy file that contains the indices labels for a data point sorted in decreasing order of relevance')
parser.add_argument('--smat-pth', type=str, help='path to the .npz score mat, recommended that you use top indices as this code will internally compute top indices if it is none')
parser.add_argument('--tst-lbl-pth', type=str, help='path to the tst_X_Y.npz file')
parser.add_argument('--trn-lbl-pth', type=str, help='path to trn_X_Y.npz file')
parser.add_argument('-A', type=float, help='value of A to calculate PSP score')
parser.add_argument('-B', type=float, help='value of B to calculate PSP score')
parser.add_argument('--pad-val', default=100000000, help='pad value index, should be greater than any label index')
parser.add_argument('--k', type=int, default=100, help='k upto which metrics need to be computed')
args = parser.parse_args()

xc_evaluator = XCEvaluator(args)
xc_evaluator.compute_metrics()
