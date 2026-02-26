import AI_stats_lab as A


def test_diabetes_pipeline():
    train_mse, test_mse, train_r2, test_r2, top3 = A.diabetes_linear_pipeline()
    assert train_mse > 0
    assert -1 <= test_r2 <= 1
    assert len(top3) == 3


def test_diabetes_cv():
    mean_r2, std_r2 = A.diabetes_cross_validation()
    assert -1 <= mean_r2 <= 1
    assert std_r2 >= 0


def test_cancer_pipeline():
    train_acc, test_acc, precision, recall, f1 = A.cancer_logistic_pipeline()
    assert 0 <= test_acc <= 1
    assert 0 <= precision <= 1


def test_logistic_regularization():
    results = A.cancer_logistic_regularization()
    assert len(results) == 5


def test_logistic_cv():
    mean_acc, std_acc = A.cancer_cross_validation()
    assert 0 <= mean_acc <= 1
