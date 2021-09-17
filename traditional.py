import data
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn import naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error


def vector_to_string(vector):
    return ' '.join([str(word) for word in vector])


def make_sentences(vectors):
    sentences = []
    for v in vectors:
        sentences.append(vector_to_string(v))
    return sentences


def extract_pos_train():
    # Extract train data
    word_ids, posts, bios, freqs, e_freqs, pos = data.read_data(
        r'SemEval2020_Task10_Emphasis_Selection-master/train_dev_data/train.txt')

    # Create feature vectors
    t, train_features = data.extract_features(word_ids, posts, bios, freqs, e_freqs, pos)
    features_by_word = []
    for f in train_features:
        for word in f:
            v = []
            v.append(word[0])
            features_by_word.append(v)

    return features_by_word


def extract_pos_test():
    # Extract test data
    word_ids, posts = data.read_test(r'SemEval2020_Task10_Emphasis_Selection-master/test_data/test_data.txt')
    pos = data.pos_tagger(posts)

    # Create feature vectors
    test_features = data.extract_features(None, posts, None, None, None, pos)
    features_by_word = []
    for f in test_features:
        for word in f:
            v = []
            v.append(float(word[0]))
            features_by_word.append(v)

    return features_by_word


def extract_pos_dev():
    # Extract train data
    word_ids, posts, bios, freqs, e_freqs, pos = data.read_data(
        r'SemEval2020_Task10_Emphasis_Selection-master/train_dev_data/dev.txt')

    # Create feature vectors
    t, train_features = data.extract_features(word_ids, posts, bios, freqs, e_freqs, pos)
    features_by_word = []
    for f in train_features:
        for word in f:
            v = []
            v.append(word[0])
            features_by_word.append(v)

    return features_by_word


def extract_prob():
    probabilities = data.get_probabilities()
    res = []
    for f in probabilities:
        for word in f:
            res.append(float(word))

    return res


def extract_prob_dev():
    probabilities = data.get_probabilities_dev()
    res = []
    for f in probabilities:
        for word in f:
            res.append(float(word))

    return res


def regression_models():

    # Feature - pos tag
    X = np.array(extract_pos_train())
    print(X)
    # Labels - pos probability
    y = np.array(extract_prob())
    print(y)
    # Regression models
    svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    svr_lin = SVR(kernel='linear', C=100, gamma='auto')
    svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1, coef0=1)

    models = [svr_rbf, svr_lin, svr_poly]
    kernel_label = ['RBF', 'Linear', 'Polynomial']
    model_color = ['m', 'c', 'g']

    X_test = np.array(extract_pos_dev())
    y_test = np.array(extract_prob_dev())

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 10), sharey=True)
    for ix, svr in enumerate(models):
        # calculate mean squared error
        predictions = svr.fit(X, y).predict(X_test)

        print(str(svr) + " Mean Squared Error -> " + str(mean_squared_error(y_test, predictions) * 100))
        axes[ix].plot(X_test, predictions, color=model_color[ix], lw=2,
                      label='{} model'.format(kernel_label[ix]))
        axes[ix].scatter(X[svr.support_], y[svr.support_], facecolor="none",
                         edgecolor=model_color[ix], s=50,
                         label='{} support vectors'.format(kernel_label[ix]))
        axes[ix].scatter(X[np.setdiff1d(np.arange(len(X)), svr.support_)],
                         y[np.setdiff1d(np.arange(len(X)), svr.support_)],
                         facecolor="none", edgecolor="k", s=50,
                         label='other training data')
        axes[ix].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
                        ncol=1, fancybox=True, shadow=True)

    fig.suptitle("Support Vector Regression", fontsize=14)
    plt.show()


def create_lt_idf():
    train_vectors = data.get_train_sentences()
    train_sentences = make_sentences(train_vectors)

    # Create feature vectors
    vectorizer = TfidfVectorizer(min_df=5,
                                 max_df=0.8,
                                 sublinear_tf=True,
                                 use_idf=True)
    train_vectors = vectorizer.fit_transform(train_sentences)
    feature_names = vectorizer.get_feature_names()

    def get_ifidf_for_words_train(word):
        for i in range(len(feature_names)):
            if feature_names[i] == word:
                return vectorizer.idf_[i]
        return 0

    train_vectors = []
    word_ids, posts, bios, freqs, e_freqs, pos = data.read_data(
        r'SemEval2020_Task10_Emphasis_Selection-master/train_dev_data/train.txt')

    for sentence in posts:
        for word in sentence:
            lst = []
            lst.append(get_ifidf_for_words_train(word))
            train_vectors.append(lst)

    probabilities = data.get_probabilities()
    train_y = []
    for p in probabilities:
        for i in p:
            if float(i) >= 0.5:
                train_y.append(1)
            else:
                train_y.append(0)
    train_y = np.array(train_y)

    dev_vectors = data.get_dev_sentences()
    dev_sentences = make_sentences(dev_vectors)
    vectorizer = TfidfVectorizer(min_df=5,
                                 max_df=0.8,
                                 sublinear_tf=True,
                                 use_idf=True)
    dev_vectors = vectorizer.fit_transform(dev_sentences)
    feature_names_dev = vectorizer.get_feature_names()

    def get_ifidf_for_words_dev(word):
        for i in range(len(feature_names_dev)):
            if feature_names_dev[i] == word:
                return vectorizer.idf_[i]
        return 0

    dev_vectors = []
    for sentence in data.get_dev_sentences():
        for word in sentence:
            lst = []
            lst.append(get_ifidf_for_words_dev(word))
            dev_vectors.append(lst)


    probabilities = data.get_probabilities_dev()
    dev_y = []
    for p in probabilities:
        for i in p:
            if float(i) >= 0.5:
                dev_y.append(1)
            else:
                dev_y.append(0)
    dev_y = np.array(dev_y)

    return train_vectors, train_y, dev_vectors, dev_y


def naive_bayes_result(train_vectors, train_y, dev_vectors, dev_y):
    Naive = naive_bayes.MultinomialNB()
    Naive.fit(train_vectors, train_y)
    # predict the labels on validation dataset
    predictions_NB = Naive.predict(dev_vectors)
    # Use accuracy_score to get the accuracy
    print("Naive Bayes Accuracy Score -> ", accuracy_score(predictions_NB, dev_y) * 100)


def svm_result(train_vectors, train_y, dev_vectors, dev_y):
    SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    SVM.fit(train_vectors, train_y)
    # predict the labels on validation dataset
    predictions_SVM = SVM.predict(dev_vectors)
    # Use accuracy_score to get the accuracy
    print("SVM Accuracy Score -> ", accuracy_score(predictions_SVM, dev_y) * 100)


if __name__ == '__main__':

    # Fit and predict regression models
    regression_models()

    # Fit and predict classification models
    train_vectors, train_y, dev_vectors, dev_y = create_lt_idf()
    naive_bayes_result(train_vectors, train_y, dev_vectors, dev_y)
    svm_result(train_vectors, train_y, dev_vectors, dev_y)
