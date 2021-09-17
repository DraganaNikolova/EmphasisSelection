import load_glove
from sklearn import preprocessing
import numpy as np
import nltk


# Function that returns list for each meta data
def read_test(filename):
    """
    This function reads the data from .txt file.
    :param filename: reading directory
    :return: lists of word_ids, words
    """
    lines = read_lines(filename) + ['']
    word_id_lst, word_id_lsts =[], []
    post_lst, post_lsts = [], []
    for line in lines:
        if line:
            splitted = line.split("\t")
            # Removing dots and commas
            if splitted[1] in ['.', ',', '@', '#']:
                continue
            word_id = splitted[0]
            words = splitted[1]

            word_id_lst.append(word_id)
            post_lst.append(words)

        elif post_lst:
            word_id_lsts.append(word_id_lst)
            post_lsts.append(post_lst)
            word_id_lst =[]
            post_lst =[]
    return word_id_lsts, post_lsts


# Function that returns list for each meta data
def read_data(filename):
    """
    This function reads the data from .txt file.
    :param filename: reading directory
    :return: lists of word_ids, words, emphasis probabilities, POS tags
    """
    lines = read_lines(filename) + ['']
    word_id_lst, word_id_lsts =[], []
    post_lst, post_lsts = [], []
    bio_lst , bio_lsts = [], []
    freq_lst, freq_lsts = [], []
    e_freq_lst, e_freq_lsts = [], []
    pos_lst, pos_lsts =[], []
    for line in lines:
        if line:
            splitted = line.split("\t")
            # Removing dots and commas
            if splitted[1] in ['.', ',', '@']:
                continue
            word_id = splitted[0]
            words = splitted[1]
            bio= splitted[2]
            freq = splitted[3]
            e_freq = splitted[4]
            pos = splitted[5]

            word_id_lst.append(word_id)
            post_lst.append(words)
            bio_lst.append(bio)
            freq_lst.append(freq)
            e_freq_lst.append(e_freq)
            pos_lst.append(pos)

        elif post_lst:
            word_id_lsts.append(word_id_lst)
            post_lsts.append(post_lst)
            bio_lsts.append(bio_lst)
            freq_lsts.append(freq_lst)
            e_freq_lsts.append(e_freq_lst)
            pos_lsts.append(pos_lst)
            word_id_lst =[]
            post_lst =[]
            bio_lst =[]
            freq_lst =[]
            e_freq_lst =[]
            pos_lst =[]
    return word_id_lsts, post_lsts, bio_lsts, freq_lsts, e_freq_lsts, pos_lsts


def read_lines( filename):
    with open(filename, 'r') as fp:
        lines = [line.strip() for line in fp]
    return lines


# Function that returns list of each pos used in the data
def get_all_pos_tags(pos):
    all_pos = []
    for i in pos:
        for j in i:
            if j not in all_pos:
                all_pos.append(j)

    return all_pos


# Function that transforms each pos into number
def enumerate_post_tags(pos, all_pos):
    total = len(all_pos)
    enumerated = []
    for i in range(1, total+1):
        enumerated.append(i)
    a = []
    a.append(enumerated)
    normalized = preprocessing.normalize(a)
    tag_to_num = {j: i for i, j in zip(normalized[0], all_pos)}

    num_to_tag = {i: tag for i, tag in enumerate(sorted(all_pos))}
    new_list = []
    for p in pos:

        new_list.append(tag_to_num.get(p))

    return new_list


def extract_features(word_ids, posts, bios, freqs, e_freqs, pos):
    """
    :return: features list: bio class, pos class
    """
    all_pos = get_all_pos_tags(pos)
    new_pos_lsts = []
    for i in pos:
        new_pos_lsts.append(enumerate_post_tags(i, all_pos))

    total, vocabulary = get_vocabulary(posts)
    glove_vectors = load_glove.load_embeddings(vocabulary)

    posts_features = []
    word_pos_dict = {}
    for post, p in zip(posts, new_pos_lsts):
        concatenated = []
        for w, p in zip(post, p):
            word_features = []
            word_features.append(p)

            glove = glove_vectors.get(w.lower().replace("#", ""))
            if glove is not None:
                word_pos_dict[w] = np.concatenate((word_features, glove.tolist()), axis=None)
                concatenated.append(np.concatenate((word_features, glove.tolist()), axis=None))
            else:
                word_pos_dict[w] = np.concatenate((word_features, np.zeros(100)), axis=None)
                concatenated.append(np.concatenate((word_features, np.zeros(100)), axis=None))

        posts_features.append(concatenated)

    return word_pos_dict, posts_features


def get_vocabulary(words):
    """
    :param words: list of lists of all words
    :return:
    """
    vocabulary = []
    cnt = 0
    for i in words:
        for word in i:
            cnt += 1
            if word.lower() not in vocabulary:
                vocabulary.append(word.lower().replace("#", ""))

    return cnt, vocabulary


def pos_tagger(posts):
    new_pos_lsts = []
    for post in posts:
        new = []
        pos_tagged = nltk.pos_tag(post)

        # print the list of tuples: (word,word_class)
        for word, word_class in pos_tagged:
            new.append(word_class)

        new_pos_lsts.append(new)

    return new_pos_lsts


def get_train_sentences():
    word_ids, posts, bios, freqs, e_freqs, pos = read_data(
        r'SemEval2020_Task10_Emphasis_Selection-master/train_dev_data/train.txt')
    return posts


def get_dev_sentences():
    word_ids, posts, bios, freqs, e_freqs, pos = read_data(
        r'SemEval2020_Task10_Emphasis_Selection-master/train_dev_data/dev.txt')
    return posts


def get_test_sentences():
    word_ids, posts = read_test(r'SemEval2020_Task10_Emphasis_Selection-master/test_data/test_data.txt')
    return posts


def avg_len_sentences(sentences):
    sum = 0
    for s in sentences:
        sum = sum + len(s)

    return sum/len(sentences)


def max_len_sentences(sentences):
    lens = []
    for s in sentences:
        lens.append(len(s))

    return max(lens)


def get_probabilities():
    word_ids, posts, bios, freqs, e_freqs, pos = read_data(
        r'SemEval2020_Task10_Emphasis_Selection-master/train_dev_data/train.txt')

    return e_freqs


def get_probabilities_dev():
    word_ids, posts, bios, freqs, e_freqs, pos = read_data(
        r'SemEval2020_Task10_Emphasis_Selection-master/train_dev_data/dev.txt')

    return e_freqs


if __name__ == '__main__':

    # Create features for train data

    word_ids, posts, bios, freqs, e_freqs, pos = read_data(r'SemEval2020_Task10_Emphasis_Selection-master/train_dev_data/train.txt')

    # Features
    train_features = extract_features(word_ids, posts, bios, freqs, e_freqs, pos)

    # Ground Truth
    train_probabilities = e_freqs

    total, vocabulary = get_vocabulary(posts)
    # total words in the train.txt are 28991 and 4025 distict words, 3943 of 4025 found by glove
    print(total, len(vocabulary))

    ##

    # Create features for dev data

    word_ids, posts, bios, freqs, e_freqs, pos = read_data(
        r'SemEval2020_Task10_Emphasis_Selection-master/train_dev_data/dev.txt')

    # Features
    dev_features = extract_features(word_ids, posts, bios, freqs, e_freqs, pos)

    # Ground Truth
    dev_probabilities = e_freqs

    ##

    # Create features for test data

    word_ids, posts = read_test(r'SemEval2020_Task10_Emphasis_Selection-master/test_data/test_data.txt')
    pos = pos_tagger(posts)
    test_features = extract_features(None, posts, None, None, None, pos)