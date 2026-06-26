# CWSF - Confidence-Weighted Stacking Forest

--- By Kiranmay Barman,
--- 3rd year undegraduate student of the Department of Geology and Geophysics, IIT Kharagpur,
--- Also pursuing B.S, Data Science and Applications, IIT Madras. 

---

## Files in this folder

| File | What it is |
|---|---|
| `cwsf_model.py` | The entire model — confidence math + the CWSFClassifier class |

---

## Step-by-step setup

### Step 1 — Install the only two dependencies

CWSF needs nothing except NumPy and scikit-learn — both of which you
almost certainly already have if you've done any ML work.

```bash
pip install numpy scikit-learn
```


### Step 3 — Import and use it

No install command needed at all. Just:

```python
from cwsf_model import CWSFClassifier
```

That's it. If this line runs without error, you're done with setup.

---

## Usage

### Basic usage — drop-in scikit-learn replacement

```python
from cwsf_model import CWSFClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=500, n_classes=3, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

model = CWSFClassifier(random_state=0)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

### Works inside Pipeline and GridSearchCV

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("cwsf", CWSFClassifier(random_state=0)),
])
pipe.fit(X_train, y_train)
pipe.predict(X_test)
```

### Custom base models

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

custom_models = [
    ("tree", DecisionTreeClassifier(max_depth=5)),
    ("nb", GaussianNB()),
]
model = CWSFClassifier(base_estimators=custom_models)
model.fit(X_train, y_train)
```

### Diagnostic — see WHY a prediction was made

```python
weights, names = model.get_model_weights(X_test)
print(names)            # ['rf', 'logreg', 'knn']
print(weights[0])       # e.g. [0.47, 0.21, 0.32] -- RF trusted most
                         # for this specific sample
```

---

## The model, briefly

CWSF combines several base models (default: Random Forest, Logistic
Regression, KNN) by weighting each one's vote **per individual
prediction** — not with a fixed global weight like standard ensembles.

The weighting uses **prediction entropy** (how confident a model's
own probability output is) corrected by **isotonic calibration**
(so "confident" actually means "historically correct," not just
"mathematically peaked"). This calibration step was added after
benchmarking revealed KNN was being over-trusted simply because it
produces sharply-peaked but poorly-calibrated probabilities — see the
docstring inside `cwsf_model.py` for the full technical explanation.

---

## If you outgrow the single file

If this project grows and you want it on GitHub as a proper installable
package later, the multi-file package version (with `setup.py`,
`pip install -e .`) covers the same model — just ask and I can give
you that version again. For now, this single file is the simplest
way to use and understand CWSF without any packaging overhead.
