import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from classifiers.logistic_regression import LogisticRegression

if __name__ == "__main__":
    train_df = pd.read_csv("./data/train.csv")
    train_df.Prediction.value_counts(normalize=True)

    review_summaries = list(train_df["Reviews_Summary"].values)
    review_summaries = [l.lower() for l in review_summaries]

    vectorizer = TfidfVectorizer()
    tfidfed = vectorizer.fit_transform(review_summaries)

    X = tfidfed
    y = train_df.Prediction.values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, random_state=42
    )

    X_train_sample = X_train[:10000]
    y_train_sample = y_train[:10000]

    clf = LogisticRegression()

    print("Start manual logistic regression" + "\n")

    clf.train(X_train, y_train, verbose=True, num_iters=1000)

    print("Train f1-score = %.3f" % accuracy_score(y_train, clf.predict(X_train)))
    print("Test f1-score = %.3f" % accuracy_score(y_test, clf.predict(X_test)))

    # clf.w = np.random.randn(X_train_sample.shape[1] + 1) * 2
    # loss, grad = clf.loss(LogisticRegression.append_biases(X_train_sample), y_train_sample, 0.0)
    #
    # f = lambda w: clf.loss(LogisticRegression.append_biases(X_train_sample), y_train_sample, 0.0)[0]
    # grad_numerical = grad_check_sparse(f, clf.w, grad, 10)

    print("\n" + "Start SGDClassifier" + "\n")

    clf = SGDClassifier(
        random_state=42,
        loss="log_loss",
        penalty="l2",
        alpha=1e-3,
        eta0=1.0,
        learning_rate="constant",
        max_iter=1000,
    )
    clf.fit(X_train, y_train)

    print("Train accuracy = %.3f" % accuracy_score(y_train, clf.predict(X_train)))
    print("Test accuracy = %.3f" % accuracy_score(y_test, clf.predict(X_test)))
