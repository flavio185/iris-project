"""Tests for model factory functions."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from iris_project.modeling.models import (
    logistic_regression_model,
    random_forest_model,
    svm_model,
)


def test_logistic_regression_type():
    model = logistic_regression_model()
    assert isinstance(model, LogisticRegression)


def test_logistic_regression_params():
    model = logistic_regression_model()
    assert model.solver == "lbfgs"
    assert model.multi_class == "multinomial"


def test_random_forest_type():
    model = random_forest_model()
    assert isinstance(model, RandomForestClassifier)


def test_random_forest_params():
    model = random_forest_model()
    assert model.n_estimators == 100
    assert model.max_depth == 5


def test_svm_type():
    model = svm_model()
    assert isinstance(model, SVC)


def test_svm_params():
    model = svm_model()
    assert model.kernel == "rbf"
    assert model.probability is True
