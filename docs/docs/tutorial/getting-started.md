# Getting Started

This guide will help you get started with Rustuna.

## Installation

Rustuna supports Python 3.11 or newer. You can install Rustuna via pip. Unlike Optuna, Rustuna doesn't have runtime dependencies, not even on NumPy. This not only eliminates concerns of version conflicts for users but also significantly speeds up imports.

```sh
$ pip install rustuna
```

## Basic Concepts & Usage

We use the terms study, trial, and parameter as follows:

- [Trial](../api/trial.md): A single call of the objective function
- [Study](../api/study.md): An optimization session, which is a set of trials
- Parameter: A variable whose value is to be optimized

### Quadratic Function Example

Rustuna can be used to arbitrary optimization problems such as hyperparameter optimization in machine learning. Here, as a simple example, let’s optimize a quadratic function: $(x-2)^2$.

First of all, import `rustuna`.

```python
import rustuna
```

In Rustuna, conventionally functions to be optimized are named objective.

```python
def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2
```

This function returns the value of $(x-2)^2$. Our goal is to find the value of `x` that minimizes the output of the objective function. This is the "optimization." During the optimization, Rustuna repeatedly calls and evaluates the objective function with different values of `x`.

A [Trial](../api/trial.md) object corresponds to a single execution of the objective function and is internally instantiated upon each invocation of the function.

The suggest APIs (for example, [suggest_float()](../api/trial.md#rustuna.trial.Trial.suggest_float)) are called inside the objective function to obtain parameters for a trial. [suggest_float()](../api/trial.md#rustuna.trial.Trial.suggest_float) selects parameters uniformly within the range provided. In our example, from $−10$ to $10$.

To start the optimization, we create a study object and pass the objective function to method [optimize()](../api/study.md#rustuna.study.Study.optimize) as follows.

```python
study = rustuna.create_study()
study.optimize(objective, n_trials=100)
```

You can get the best parameter as follows.

```python
best_params = study.best_params
found_x = best_params["x"]
print(f"Found x: {found_x}, (x - 2)^2: {(found_x - 2) ** 2}")
```

We will see that the `x` value found by Rustuna is close to the optimal value of $2$.

Example output:

```
Found x: 1.9977979724456008, (x - 2)^2: 4.848925350333516e-06
```

!!! note
    When used to search for hyperparameters in machine learning, usually the objective function would return the loss or accuracy of the model.

## Study Object

In Rustuna, we use the study object to manage optimization. Method [create_study()](../api/study.md#rustuna.study.create_study) returns a study object. A study object has useful properties for analyzing the optimization outcome.

To get the best trial:

```python
study.best_trial
```

Example output:

```
number=82 state=COMPLETE values=Some([4.848925350333516e-6]) params={'x': 1.9977979724456008} distributions={"x": PyDistribution { distribution: Float { low: -10.0, high: 10.0, step: None, log: false }, category_labels: None }} user_attrs={} system_attrs={}
```

To get the dictionary of parameter name and parameter values:

```python
study.best_trial.params
```

Example output:

```
{'x': 1.9977979724456008}
```

To get all trials:

```python
study.trials
for trial in study.trials[:2]:  # Show first two trials
    print(trial)
```

Example output:

```
number=0 state=COMPLETE values=Some([141.39491550422403]) params={'x': -9.890959402177103} distributions={"x": PyDistribution { distribution: Float { low: -10.0, high: 10.0, step: None, log: false }, category_labels: None }} user_attrs={} system_attrs={}
number=1 state=COMPLETE values=Some([16.92416574721206]) params={'x': -2.1138990929788326} distributions={"x": PyDistribution { distribution: Float { low: -10.0, high: 10.0, step: None, log: false }, category_labels: None }} user_attrs={} system_attrs={}
```

To get the number of trials:

```python
print(len(study.trials))
```

Example output:

```
100
```

By executing optimize() again, we can continue the optimization.

```python
study.optimize(objective, n_trials=100)
print(len(study.trials))
```

Example output:

```
200
```

A small improvement will be confirmed after the additional optimizaiton:

```python
best_params = study.best_trial.params
found_x = best_params["x"]
print(f"Found x: {found_x}, (x - 2)^2: {(found_x - 2) ** 2}")
```

Example output:

```
Found x: 1.9991240057627049, (x - 2)^2: 7.673659037742573e-07
```

Rustuna studies can be passed to [Optuna's visualization APIs](https://optuna.readthedocs.io/en/stable/reference/visualization/index.html)
by wrapping them with [ToOptunaStudy](../api/converter.md#rustuna.converter.ToOptunaStudy).

```python
from optuna.visualization import plot_optimization_history

from rustuna.converter import ToOptunaStudy

optuna_study = ToOptunaStudy(study)
fig = plot_optimization_history(optuna_study)
fig.show()
```

![Optimization history plot](../assets/images/getting-started-plot-optimization-history.jpg)

## Defining Search Space

For parameter sampling, Rustuna provides the following features:

- [Trial.suggest_categorical()](../api/trial.md#rustuna.trial.Trial.suggest_categorical) for categorical parameters
- [Trial.suggest_int()](../api/trial.md#rustuna.trial.Trial.suggest_int) for integer parameters
- [Trial.suggest_float()](../api/trial.md#rustuna.trial.Trial.suggest_float) for floating point parameters

With optional arguments of `step` and `log`, we can discretize or take the logarithm of integer and floating point parameters.

```python
import rustuna


def objective(trial):
    # Categorical parameter
    optimizer = trial.suggest_categorical("optimizer", ["MomentumSGD", "Adam"])

    # Integer parameter
    num_layers = trial.suggest_int("num_layers", 1, 3)

    # Integer parameter (log)
    num_channels = trial.suggest_int("num_channels", 32, 512, log=True)

    # Integer parameter (discretized)
    num_units = trial.suggest_int("num_units", 10, 100, step=5)

    # Floating point parameter
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 1.0)

    # Floating point parameter (log)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)

    # Floating point parameter (discretized)
    drop_path_rate = trial.suggest_float("drop_path_rate", 0.0, 1.0, step=0.1)
    ...
```

### Branches and Loops

In Rustuna, we define search spaces using familiar Python syntax including conditionals and loops.

Also, you can use branches or loops depending on the parameter values.

- Branches:

```python
import sklearn.ensemble
import sklearn.svm


def objective(trial):
    classifier_name = trial.suggest_categorical("classifier", ["SVC", "RandomForest"])
    if classifier_name == "SVC":
        svc_c = trial.suggest_float("svc_c", 1e-10, 1e10, log=True)
        classifier_obj = sklearn.svm.SVC(C=svc_c)
    else:
        rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
        classifier_obj = sklearn.ensemble.RandomForestClassifier(max_depth=rf_max_depth)
    ...
```

- Loops:

```python
import torch
import torch.nn as nn


def create_model(trial, in_size):
    n_layers = trial.suggest_int("n_layers", 1, 3)

    layers = []
    for i in range(n_layers):
        n_units = trial.suggest_int(f"n_units_l{i}", 4, 128, log=True)
        layers.append(nn.Linear(in_size, n_units))
        layers.append(nn.ReLU())
        in_size = n_units
    layers.append(nn.Linear(in_size, 10))

    return nn.Sequential(*layers)
```

!!! note
    The difficulty of optimization increases roughly exponentially with regard to the number of parameters. That is, the number of necessary trials increases exponentially when you increase the number of parameters, so it is recommended to not add unimportant parameters.


## Optimization Algorithms

Rustuna employs samplers, i.e., optimization algorithms, proven in Optuna.

Samplers basically continually narrow down the search space using the records of suggested parameter values and evaluated objective values, leading to an optimal search space which giving off parameters leading to better objective values.

Rustuna provides the following sampling algorithms:

- Random Search implemented in [RandomSampler](../api/samplers.md#rustuna.samplers.RandomSampler)
- Tree-structured Parzen Estimator algorithm implemented in [TPESampler](../api/samplers.md#rustuna.samplers.TPESampler)
- Nondominated Sorting Genetic Algorithm II implemented in [NSGAIISampler](../api/samplers.md#rustuna.samplers.NSGAIISampler)
- Covariance Matrix Adaptation Evolution Strategy implemented in [CmaEsSampler](../api/samplers.md#rustuna.samplers.CmaEsSampler)

The default sampler is [TPESampler](../api/samplers.md#rustuna.samplers.TPESampler).

You can specify a sampler using the `sampler` argument in [create_study()](../api/study.md#rustuna.study.create_study) as follows.

```python
study = rustuna.create_study(sampler=rustuna.samplers.NSGAIISampler())
```

### Status of Supported Features in Each Sampler

||[RandomSampler](../api/samplers.md#rustuna.samplers.RandomSampler)|[TPESampler](../api/samplers.md#rustuna.samplers.TPESampler)|[NSGAIISampler](../api/samplers.md#rustuna.samplers.NSGAIISampler)|[CmaEsSampler](../api/samplers.md#rustuna.samplers.CmaEsSampler)|
|-|-|-|-|-|
|Float parameters|✓|✓|▴|✓|
|Integer parameters|✓|✓|▴|✓|
|Categorical parameters|✓|✓|✓|×|
|Multivariate optimization|▴|✓|✓|✓|
|Conditional search space|✓|✓|▴|×|
|Multi-objective optimization|✓|✓|✓ (▴ for single-objective)|×|
|Constrained optimization|×|×|×|×|
|Pure Rust Implementation|✓|✓|✓|×|

!!! note
    ✓: Supports this feature. ▴: Works, but inefficiently. ×: Causes an error, or has no interface.

## Storages

Rustuna uses in-memory storage ([rustuna.storages.InMemoryStorage](../api/storages.md#rustuna.storages.InMemoryStorage)) by default. This is very fast, but it is volatile, and the lifespan of the data is limited to the duration of the program's execution.

### Saving/Resuming Study with RDB Backend/File-Based Journal Storage

An RDB backend enables persistent experiments (i.e., to save and resume a study) as well as access to history of studies. In this section, let’s try simple examples running on a local environment with SQLite DB.

We can create a persistent study by calling [create_study()](../api/study.md#rustuna.study.create_study) function with a sqlite3 storage object as follows. An SQLite file `sqlite3_example.db` is created to store a new study record.

```python
import rustuna


def objective(trial: rustuna.Trial) -> float:
    x = trial.suggest_float("x", -10, 10)
    y = trial.suggest_int("y", -5, 5)
    return (x - 2) ** 2 + y**2


storage = rustuna.storages.SQLite3Storage("rustuna_example.db", create_database=True)
study = rustuna.create_study(
    study_name="sqlite3_example",
    storage=storage,
    direction="minimize",
)
study.optimize(objective, n_trials=50)

print(f"Number of trials: {len(study.trials)}")
print(f"Best value: {study.best_trial.values}")
print(f"Best params: {study.best_trial.params}")
```

Example output:

```
Number of trials: 50
Best value: [0.17297553171217403]
Best params: {'x': 2.415903272062356, 'y': 0}
```

To resume a study, call [load_study](../api/study.md#rustuna.study.load_study) with the `study_name` and `storage` arguments as follows. When loading, you must specify the same database file and study name. Also, set `create_database` to `False` (Note: This argument is `False` by default, so it can be omitted).

```python
storage = rustuna.storages.SQLite3Storage("rustuna_example.db", create_database=False)
study = rustuna.load_study(study_name="sqlite3_example", storage=storage)
study.optimize(objective, n_trials=50)

print(f"Number of trials: {len(study.trials)}")
print(f"Best value: {study.best_trial.values}")
print(f"Best params: {study.best_trial.params}")
```

Example output:

```
Number of trials: 100
Best value: [8.562350946635473e-05]
Best params: {'x': 2.0092532972213344, 'y': 0}
```

Rustuna also provides Journal Storage ([rustuna.storages.JournalFileStorage](../api/storages.md#rustuna.storages.JournalFileStorage)) and it can be available as follows:

```python
storage = rustuna.storages.JournalFileStorage("rustuna_example.log")
```
