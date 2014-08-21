'''
=================================
Training a Naive Bayes Classifier
=================================

>>> from nltk.corpus import movie_reviews
>>> from featx import label_feats_from_corpus, split_label_feats
>>> movie_reviews.categories()
['neg', 'pos']
>>> lfeats = label_feats_from_corpus(movie_reviews)
>>> lfeats.keys()
dict_keys(['neg', 'pos'])
>>> train_feats, test_feats = split_label_feats(lfeats)
>>> len(train_feats)
1500
>>> len(test_feats)
500

>>> from nltk.classify import NaiveBayesClassifier
>>> nb_classifier = NaiveBayesClassifier.train(train_feats)
>>> nb_classifier.labels()
['neg', 'pos']

>>> from featx import bag_of_words
>>> negfeat = bag_of_words(['the', 'plot', 'was', 'ludicrous'])
>>> nb_classifier.classify(negfeat)
'neg'
>>> posfeat = bag_of_words(['kate', 'winslet', 'is', 'accessible'])
>>> nb_classifier.classify(posfeat)
'pos'

>>> from nltk.classify.util import accuracy
>>> accuracy(nb_classifier, test_feats)
0.728

>>> probs = nb_classifier.prob_classify(test_feats[0][0])
>>> probs.samples()
dict_keys(['neg', 'pos'])
>>> probs.max()
'pos'
>>> probs.prob('pos')
0.9999999646430913
>>> probs.prob('neg')
3.535688969240647e-08

>>> nb_classifier.most_informative_features(n=5)
[('magnificent', True), ('outstanding', True), ('insulting', True), ('vulnerable', True), ('ludicrous', True)]

>>> from nltk.probability import LaplaceProbDist
>>> nb_classifier = NaiveBayesClassifier.train(train_feats, estimator=LaplaceProbDist)
>>> accuracy(nb_classifier, test_feats)
0.716

>>> from nltk.probability import DictionaryProbDist
>>> label_probdist = DictionaryProbDist({'pos': 0.5, 'neg': 0.5})
>>> true_probdist = DictionaryProbDist({True: 1})
>>> feature_probdist = {('pos', 'yes'): true_probdist, ('neg', 'no'): true_probdist}
>>> classifier = NaiveBayesClassifier(label_probdist, feature_probdist)
>>> classifier.classify({'yes': True})
'pos'
>>> classifier.classify({'no': True})
'neg'


===================================
Training a Decision Tree Classifier
===================================

>>> from nltk.classify import DecisionTreeClassifier
>>> dt_classifier = DecisionTreeClassifier.train(train_feats, binary=True, entropy_cutoff=0.8, depth_cutoff=5, support_cutoff=30)
>>> accuracy(dt_classifier, test_feats)
0.688

>>> from nltk.probability import FreqDist, MLEProbDist, entropy
>>> fd = FreqDist({'pos': 30, 'neg': 10})
>>> entropy(MLEProbDist(fd))
0.8112781244591328
>>> fd['neg'] = 25
>>> entropy(MLEProbDist(fd))
0.9940302114769565
>>> fd['neg'] = 30
>>> entropy(MLEProbDist(fd))
1.0
>>> fd['neg'] = 1
>>> entropy(MLEProbDist(fd))
0.20559250818508304


=====================================
Training a Maximum Entropy Classifier
=====================================

>>> from nltk.classify import MaxentClassifier
>>> me_classifier = MaxentClassifier.train(train_feats, trace=0, max_iter=1, min_lldelta=0.5)
>>> accuracy(me_classifier, test_feats)
0.5

>>> me_classifier = MaxentClassifier.train(train_feats, algorithm='gis', trace=0, max_iter=10, min_lldelta=0.5)
>>> accuracy(me_classifier, test_feats)
0.722

=================================
Training Scikit-Learn Classifiers
=================================

>>> from nltk.classify.scikitlearn import SklearnClassifier
>>> from sklearn.naive_bayes import MultinomialNB
>>> sk_classifier = SklearnClassifier(MultinomialNB())
>>> sk_classifier.train(train_feats)
<SklearnClassifier(MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True))>
>>> accuracy(sk_classifier, test_feats)
0.83

>>> from sklearn.naive_bayes import BernoulliNB
>>> sk_classifier = SklearnClassifier(BernoulliNB())
>>> sk_classifier.train(train_feats)
<SklearnClassifier(BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True))>
>>> accuracy(sk_classifier, test_feats)
0.812

>>> from sklearn.linear_model import LogisticRegression
>>> sk_classifier = SklearnClassifier(LogisticRegression()).train(train_feats)
>>> accuracy(sk_classifier, test_feats)
0.892

>>> from sklearn.svm import SVC
>>> sk_classifier = SklearnClassifier(SVC()).train(train_feats)
>>> accuracy(sk_classifier, test_feats)
0.69

>>> from sklearn.svm import LinearSVC
>>> sk_classifier = SklearnClassifier(LinearSVC()).train(train_feats)
>>> accuracy(sk_classifier, test_feats)
0.864

>>> from sklearn.svm import NuSVC
>>> sk_classifier = SklearnClassifier(NuSVC()).train(train_feats)
>>> accuracy(sk_classifier, test_feats)
0.882

==============================================
Measuring Precision and Recall of a Classifier
==============================================

>>> from classification import precision_recall
>>> nb_precisions, nb_recalls = precision_recall(nb_classifier, test_feats)
>>> nb_precisions['pos']
0.6413612565445026
>>> nb_precisions['neg']
0.9576271186440678
>>> nb_recalls['pos']
0.98
>>> nb_recalls['neg']
0.452

>>> me_precisions, me_recalls = precision_recall(me_classifier, test_feats)
>>> me_precisions['pos']
0.6456692913385826
>>> me_precisions['neg']
0.9663865546218487
>>> me_recalls['pos']
0.984
>>> me_recalls['neg']
0.46

>>> sk_precisions, sk_recalls = precision_recall(sk_classifier, test_feats)
>>> sk_precisions['pos']
0.9063829787234042
>>> sk_precisions['neg']
0.8603773584905661
>>> sk_recalls['pos']
0.852
>>> sk_recalls['neg']
0.912


==================================
Calculating High Information Words
==================================

>>> from featx import high_information_words, bag_of_words_in_set
>>> labels = movie_reviews.categories()
>>> labeled_words = [(l, movie_reviews.words(categories=[l])) for l in labels]
>>> high_info_words = set(high_information_words(labeled_words))
>>> feat_det = lambda words: bag_of_words_in_set(words, high_info_words)
>>> lfeats = label_feats_from_corpus(movie_reviews, feature_detector=feat_det)
>>> train_feats, test_feats = split_label_feats(lfeats)

>>> nb_classifier = NaiveBayesClassifier.train(train_feats)
>>> accuracy(nb_classifier, test_feats)
0.91
>>> nb_precisions, nb_recalls = precision_recall(nb_classifier, test_feats)
>>> nb_precisions['pos']
0.8988326848249028
>>> nb_precisions['neg']
0.9218106995884774
>>> nb_recalls['pos']
0.924
>>> nb_recalls['neg']
0.896

>>> me_classifier = MaxentClassifier.train(train_feats, algorithm='gis', trace=0, max_iter=10, min_lldelta=0.5)
>>> accuracy(me_classifier, test_feats)
0.912
>>> me_precisions, me_recalls = precision_recall(me_classifier, test_feats)
>>> me_precisions['pos']
0.8992248062015504
>>> me_precisions['neg']
0.9256198347107438
>>> me_recalls['pos']
0.928
>>> me_recalls['neg']
0.896

>>> dt_classifier = DecisionTreeClassifier.train(train_feats, binary=True, depth_cutoff=20, support_cutoff=20, entropy_cutoff=0.01)
>>> accuracy(dt_classifier, test_feats)
0.688
>>> dt_precisions, dt_recalls = precision_recall(dt_classifier, test_feats)
>>> dt_precisions['pos']
0.6766917293233082
>>> dt_precisions['neg']
0.7008547008547008
>>> dt_recalls['pos']
0.72
>>> dt_recalls['neg']
0.656

>>> sk_classifier = SklearnClassifier(LinearSVC()).train(train_feats)
>>> accuracy(sk_classifier, test_feats)
0.86
>>> sk_precisions, sk_recalls = precision_recall(sk_classifier, test_feats)
>>> sk_precisions['pos']
0.871900826446281
>>> sk_precisions['neg']
0.8488372093023255
>>> sk_recalls['pos']
0.844
>>> sk_recalls['neg']
0.876


=================================
Combining Classifiers with Voting
=================================

>>> from classification import MaxVoteClassifier
>>> mv_classifier = MaxVoteClassifier(nb_classifier, dt_classifier, me_classifier, sk_classifier)
>>> mv_classifier.labels()
['neg', 'pos']
>>> accuracy(mv_classifier, test_feats)
0.894
>>> mv_precisions, mv_recalls = precision_recall(mv_classifier, test_feats)
>>> mv_precisions['pos']
0.9156118143459916
>>> mv_precisions['neg']
0.8745247148288974
>>> mv_recalls['pos']
0.868
>>> mv_recalls['neg']
0.92


============================================
Classifying with Multiple Binary Classifiers
============================================

>>> from nltk.corpus import reuters
>>> len(reuters.categories())
90

>>> from featx import reuters_high_info_words, reuters_train_test_feats
>>> rwords = reuters_high_info_words()
>>> featdet = lambda words: bag_of_words_in_set(words, rwords)
>>> multi_train_feats, multi_test_feats = reuters_train_test_feats(featdet)

>>> from classification import train_binary_classifiers
>>> trainf = lambda train_feats: SklearnClassifier(LogisticRegression()).train(train_feats)
>>> labelset = set(reuters.categories())
>>> classifiers = train_binary_classifiers(trainf, multi_train_feats, labelset)
>>> len(classifiers)
90

>>> from classification import MultiBinaryClassifier, multi_metrics
>>> multi_classifier = MultiBinaryClassifier(*classifiers.items())

>>> multi_precisions, multi_recalls, avg_md = multi_metrics(multi_classifier, multi_test_feats)
>>> avg_md
0.23310715863026216

>>> multi_precisions['soybean']
0.7857142857142857
>>> multi_recalls['soybean']
0.3333333333333333
>>> len(reuters.fileids(categories=['soybean']))
111

>>> multi_precisions['sunseed']
1.0
>>> multi_recalls['sunseed']
0.2
>>> len(reuters.fileids(categories=['sunseed']))
16
'''

if __name__ == '__main__':
	import doctest
	doctest.testmod()