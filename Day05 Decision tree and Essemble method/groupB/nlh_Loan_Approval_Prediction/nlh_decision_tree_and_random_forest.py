from nlh_models import load_data, preprocess, train_decision_tree, train_random_forest, evaluate_model, save_model
from sklearn.model_selection import train_test_split


def main():
    df = load_data()
    X, y, meta = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    dt = train_decision_tree(X_train, y_train)
    print('Decision Tree evaluation:')
    res_dt = evaluate_model(dt, X_test, y_test)
    save_model(dt, 'nlh_decision_tree.joblib')

    rf = train_random_forest(X_train, y_train)
    print('\nRandom Forest evaluation:')
    res_rf = evaluate_model(rf, X_test, y_test)
    save_model(rf, 'nlh_random_forest.joblib')

    print('\nSummary:')
    print(f"DT accuracy: {res_dt['accuracy']:.4f}")
    print(f"RF accuracy: {res_rf['accuracy']:.4f}")

if __name__ == '__main__':
    main()
