from predictive_maintenance.data.prepare_dataset import prepare_dataset


def test_prepare_dataset():

    X_train, X_test, y_train, y_test = prepare_dataset()

    assert len(X_train) > 0
    assert len(X_test) > 0