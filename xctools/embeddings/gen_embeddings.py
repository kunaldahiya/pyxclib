# Run using python gen_embeddings.py <embed> <embedding_dim> <vocabulary> <tokens> <out_file>
import numpy as np
import sys
import word_embedding

__author__='KD'

def main():
    init = 'gaussian'
    embed = sys.argv[1]
    embedding_dim = int(sys.argv[2])
    vocab_file = sys.argv[3]
    tokens = sys.argv[4]
    out_file = sys.argv[5]
    with open(vocab_file, 'r') as f:
        temp = f.readlines()
    dataset_vocab = [item.rstrip("\n") for item in temp] #list of all words in given dataset
    del temp

    word_embeddings = word_embedding.load_embeddings(DATA_DIR='/home/kd/XC/data/word_embeddings', embed=embed, tokens=tokens, embedding_dim=embedding_dim)
    dataset_vocab_embeddings = np.zeros((len(dataset_vocab), embedding_dim))
    not_found_count = 0

    for i in range(len(dataset_vocab)):
        try:
            dataset_vocab_embeddings[i, :] = word_embeddings[dataset_vocab[i]]
        except KeyError:
            if init == 'gaussian':
                dataset_vocab_embeddings[i, :] = np.random.randn(embedding_dim, )*0.01
            not_found_count+=1
    print("#Words with no word embeddings", not_found_count)
    np.save(out_file, dataset_vocab_embeddings)

if __name__ == '__main__':
    main()
