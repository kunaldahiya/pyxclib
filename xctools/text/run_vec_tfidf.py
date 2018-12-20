"""
    Generate vectorized text and tf-idf for given text
"""
import os
import sys
import numpy as np
import text_utils

def main():
    data_dir = sys.argv[1]
    # text feature object
    t_obj = text_utils.TextUtility(max_df=0.8, min_df=1)
    #feat_obj =
    t_obj.fit(os.path.join(data_dir, sys.argv[2]))
    text_utils.save_vocabulary(os.path.join(data_dir, 'vocabulary.json'), t_obj.vocabulary)
    t_obj.save(os.path.join(data_dir, 'text_model.pkl'))
    # Vectorized text as per vocabulary
    vectorized_text = t_obj.transform(os.path.join(data_dir, sys.argv[2]))
    # Save vectorized text
    text_utils.save_vectorized_text(sys.argv[2], vectorized_text)
    #Gen tf-idf features
    tf_obj = text_utils.TFIDF(vocabulary_size=len(t_obj.vocabulary))
    tf_obj.fit(vectorized_text)
    tfidf = tf_obj.transform(vectorized_text)
    text_utils.save_mat(sys.argv[3], tfidf)

if __name__ == '__main__':
    main()
