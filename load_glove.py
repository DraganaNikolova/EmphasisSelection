import numpy as np
import os

# Folder where the vectors are saved
current_dir = os.getcwd()
glove_file = os.path.join(current_dir, 'GloVe')
glove_file = os.path.join(glove_file, 'glove.6B.100d.txt')


def load_embeddings(vocabulary):
    embeddings = dict()
    with open(glove_file, 'r', encoding='utf-8') as doc:
        line = doc.readline()
        while line != '':
            line = line.rstrip('\n').lower()
            parts = line.split(' ')
            values = np.array(parts[1:], dtype=np.float)

            if parts[0] in vocabulary:
                embeddings[parts[0]] = values

            line = doc.readline()

    return embeddings
