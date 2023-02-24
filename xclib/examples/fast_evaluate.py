import argparse
import numpy as np
from xclib.evaluation.xc_metrics import _get_topk, precision, recall, psprecision, compute_inv_propesity, calc_gt_metrics
import scipy.sparse as sp

class XCEvaluator():
    def __init__(self, args):
        super(XCEvaluator, self).__init__()
        self.args = args
        self.k = 200
        self.top_indices = None
        self.smat = None
        self.tst_X_Y = sp.load_npz(args.tst_lbl_pth)
        self.trn_X_Y = sp.load_npz(args.trn_lbl_pth)
        self.A = args.A
        self.B = args.B
    
    def process_indices(self):
        '''
        Ensure that there are no duplicate indices for each data point _get_topk will pad with the index num_labels for remaining slots, this will be problematic in in1d later
        '''
        num_docs, num_labels = self.tst_X_Y.shape
        offsets = np.arange(self.k)
        padded_indices = np.where(self.top_indices == num_labels)
        offset_arr = np.zeros((num_docs, self.k))
        
        offset_arr[padded_indices] = 1
        self.top_indices = self.top_indices + offset_arr * offsets
        
    def compute_top_indices(self):
        self.smat = sp.load_npz(args.smat_pth)
        if args.top_indices_pth is not None:
            self.top_indices = np.load(args.top_indices_pth)
        else:
            self.top_indices = _get_topk(self.smat, k = self.k)
        self.process_indices()
    
    def build_metrics_dict(self, precs, recalls, psps, recall_at_gt, micro_recall_at_gt):
        ks = [1, 3, 5, 10, 50, 100]
        metrics_dict = {}
        for k in ks:
            metrics_dict[f'P@{k}'] = f'{precs[k - 1] :.5f}'
            metrics_dict[f'R@{k}'] = f'{recalls[k - 1] :.5f}'
            metrics_dict[f'PSP@{k}'] = f'{psps[k - 1] :.5f}'
        metrics_dict['R@GT'] = f'{recall_at_gt :.5f}'
        metrics_dict['MicroR@GT'] = f'{micro_recall_at_gt :.5f}'
        return metrics_dict
    
    def compute_metrics(self):
        self.compute_top_indices()
        inv_psp = compute_inv_propesity(self.trn_X_Y, self.A, self.B)
        precs = precision(self.smat, self.tst_X_Y, 100)
        print("Finished Precision computation")
        recalls = recall(self.smat, self.tst_X_Y, 100)
        print("Finished Recall computation")
        psps = psprecision(self.smat, self.tst_X_Y, inv_psp, k=100) 
        print("Finished PSP computation")
        # @ GT metrics
        recall_at_gt, micro_recall_at_gt = calc_gt_metrics(self.top_indices, self.tst_X_Y, self.k)
        print("Finished @ GT metrics computation")
        metrics_dict = self.build_metrics_dict(precs, recalls, psps, recall_at_gt, micro_recall_at_gt)
        
        for metric_name, metric_val in metrics_dict.items():
            print(metric_name, float(metric_val) * 100)

'''
Sample Command: python fast_evaluate.py --smat-pth /home/t-abuvanesh/xc/t-abuvanesh/xfc/results/LF-AmazonTitles-131K-Kunal/msmarco-distilbert-base-v4_score_mat.npz --tst-lbl-pth ~/xc/Datasets/LF-AmazonTitles-131K-Kunal/tst_X_Y.npz --trn-lbl-pth ~/xc/Datasets/LF-AmazonTitles-131K-Kunal/trn_X_Y.npz -A 0.6 -B 2.6 --trn-lbl-pth ~/xc/Datasets/LF-AmazonTitles-131K-Kunal/trn_X_Y.npz -A 0.6 -B 2.6
'''
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
