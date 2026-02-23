from nlh_models import load_data, preprocess, train_decision_tree, evaluate_model, save_model
from sklearn.model_selection import train_test_split


def main():
    df = load_data()
    X, y, meta = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    dt = train_decision_tree(X_train, y_train)
    print('Decision Tree evaluation:')
    evaluate_model(dt, X_test, y_test)
    save_model(dt, 'nlh_decision_tree.joblib')

if __name__ == '__main__':
    main()
