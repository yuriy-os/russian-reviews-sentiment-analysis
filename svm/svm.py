import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

if __name__ == "__main__":
    dataset = pd.read_csv('./dataset.csv')
    
    X_train, X_test, y_train, y_test = train_test_split(
        dataset["text"], dataset["label"], test_size=0.2, random_state=1, shuffle=True
    )

    baseline_pipeline = Pipeline(
        [("vect", TfidfVectorizer(ngram_range=(1, 3))), ("svc", LinearSVC())]
    )

    baseline_pipeline.fit(X_train, y_train)
    print(classification_report(y_test, baseline_pipeline.predict(X_test), digits=4))