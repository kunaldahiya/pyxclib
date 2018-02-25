"""
    Generate vectorized text and tf-idf for given text
"""
import sys
import numpy as np
import text_utils
import os

def get_stats(sp_mat):
    '''
        Find statistics of features
    '''
    num_docs, num_feats = sp_mat.shape
    rows, _ = sp_mat.nonzero()
    non_zero_docs = len(np.unique(rows))
    print("#Documents: {}, #Words: {}, #NZ: {}".format(
        num_docs, num_feats, num_docs - non_zero_docs))


def main():
    data_dir = sys.argv[1]
    # text feature object
    t_obj = text_utils.TextUtility(max_df=0.8, min_df=2)
    t_obj.fit(os.path.join(data_dir, sys.argv[2]))
    text_utils.save_vocabulary(os.path.join(data_dir, 'vocabulary.json'), t_obj.vocabulary)
    t_obj.save(os.path.join(data_dir, 'text_model.pkl'))
    # Vectorized text as per vocabulary
    vectorized_text = t_obj.transform(sys.argv[1])
    # Save vectorized text
    text_utils.save_vectorized_text(sys.argv[2], vectorized_text)
    #Gen tf-idf features
    tf = text_utils.vectorized_text_to_term_freq(vectorized_text, len(t_obj.vocabulary))
    tfidf = text_utils.tf_idf(tf)
    get_stats(tfidf)
    text_utils.save_mat(sys.argv[3], tfidf)

if __name__ == '__main__':
    main()
