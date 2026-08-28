"""
Unit tests for the Multi-Model Benchmark Suite.
"""

from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier


def test_ensemble_and_tree_classifiers_initialization():
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=10, random_state=42)
    et = ExtraTreesClassifier(n_estimators=10, random_state=42)

    X_dummy = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    y_dummy = [0, 1, 0, 1]

    rf.fit(X_dummy, y_dummy)
    gb.fit(X_dummy, y_dummy)
    et.fit(X_dummy, y_dummy)

    assert rf.predict_proba(X_dummy).shape == (4, 2)
    assert gb.predict_proba(X_dummy).shape == (4, 2)
    assert et.predict_proba(X_dummy).shape == (4, 2)
