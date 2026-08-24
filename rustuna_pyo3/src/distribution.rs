use pyo3::exceptions::PyRuntimeError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::types::PyList;
use pyo3::types::{PyBool, PyFloat, PyInt, PyString, PyType};

use rustuna_core::attr::{get_category_labels, Attrs, CategoryLabel};
use rustuna_core::distribution::Distribution;

#[derive(Clone, Debug)]
#[pyclass(name = "Distribution", from_py_object)]
#[pyo3(module = "rustuna")]
pub struct PyDistribution {
    pub distribution: Distribution,
    pub category_labels: Option<Vec<CategoryLabel>>,
}
impl PyDistribution {
    pub fn new(distribution: Distribution, name: &str, study_attrs: &Attrs) -> PyDistribution {
        match distribution {
            Distribution::Categorical { cardinality } => {
                let category_labels = get_category_labels(study_attrs, name, cardinality);
                PyDistribution {
                    distribution,
                    category_labels,
                }
            }
            _ => PyDistribution {
                distribution,
                category_labels: None,
            },
        }
    }

    pub fn new_with_category_labels(
        distribution: Distribution,
        category_labels: Option<Vec<CategoryLabel>>,
    ) -> PyDistribution {
        PyDistribution {
            distribution,
            category_labels,
        }
    }
}
impl From<PyDistribution> for Distribution {
    fn from(val: PyDistribution) -> Self {
        val.distribution
    }
}
#[pymethods]
impl PyDistribution {
    #[classmethod]
    #[pyo3(signature = (low, high, log=false, step=None))]
    pub fn float(
        _cls: &Bound<'_, PyType>,
        low: f64,
        high: f64,
        log: bool,
        step: Option<f64>,
    ) -> Self {
        PyDistribution {
            distribution: Distribution::new_float(low, high, step, log),
            category_labels: None,
        }
    }

    #[classmethod]
    #[pyo3(signature = (low, high, log=false, step=1))]
    pub fn int(_cls: &Bound<'_, PyType>, low: i64, high: i64, log: bool, step: i64) -> Self {
        PyDistribution {
            distribution: Distribution::new_int(low, high, step, log),
            category_labels: None,
        }
    }

    #[classmethod]
    pub fn categorical(_cls: &Bound<'_, PyType>, choices: Vec<Py<PyAny>>) -> PyResult<Self> {
        let cardinality = choices.len();
        let mut labels: Vec<CategoryLabel> = Vec::with_capacity(choices.len());
        let py = _cls.py();
        for choice in choices {
            let label = pyobject_to_category_label(choice.bind(py).as_any())?;
            labels.push(label);
        }
        let py_dist = PyDistribution {
            distribution: Distribution::new_categorical(cardinality),
            category_labels: Some(labels),
        };
        Ok(py_dist)
    }

    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        match &self.distribution {
            Distribution::Float {
                low,
                high,
                step,
                log,
            } => {
                let dist = PyDict::new(py);
                dist.set_item("type", "FloatDistribution")?;
                dist.set_item("low", low)?;
                dist.set_item("high", high)?;
                dist.set_item("log", log)?;
                if let Some(step) = step {
                    dist.set_item("step", step)?;
                } else {
                    dist.set_item("step", py.None())?;
                }
                Ok(dist)
            }
            Distribution::Int {
                low,
                high,
                step,
                log,
            } => {
                let dist = PyDict::new(py);
                dist.set_item("type", "IntDistribution")?;
                dist.set_item("low", low)?;
                dist.set_item("high", high)?;
                dist.set_item("log", log)?;
                dist.set_item("step", step)?;
                Ok(dist)
            }
            Distribution::Categorical { cardinality } => {
                let dist = PyDict::new(py);
                dist.set_item("type", "CategoricalDistribution")?;

                let mut elements: Vec<Py<PyAny>> = Vec::with_capacity(*cardinality);
                let labels = match self.category_labels {
                    Some(ref labels) => labels.clone(),
                    None => {
                        let mut labels: Vec<CategoryLabel> = Vec::with_capacity(*cardinality);
                        for i in 0..*cardinality {
                            labels.push(CategoryLabel::Int(i as i64));
                        }
                        labels
                    }
                };
                for i in 0..*cardinality {
                    let c = labels.get(i).ok_or(PyValueError::new_err(
                        "Internal representation of categorical value is out of range",
                    ))?;
                    elements.push(category_label_to_pyobject(py, c)?.unbind());
                }
                let choices = PyList::new(py, &elements)?;
                dist.set_item("choices", choices)?;
                Ok(dist)
            }
        }
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let class_obj = slf.get_type();
        let class_name = class_obj.name()?;
        Ok(format!("{}({:?})", class_name, slf.borrow().__str__()?))
    }

    fn __str__(&self) -> PyResult<String> {
        Python::attach(|py| Ok(self.to_dict(py)?.to_string()))
    }
}

pub fn category_label_to_pyobject<'py>(
    py: Python<'py>,
    label: &CategoryLabel,
) -> PyResult<Bound<'py, PyAny>> {
    match label {
        CategoryLabel::String(s) => Ok(s.into_pyobject(py)?.into_any()),
        CategoryLabel::Int(i) => Ok(i.into_pyobject(py)?.into_any()),
        CategoryLabel::Float(f) => Ok(f.into_pyobject(py)?.into_any()),
        CategoryLabel::Bool(b) => Ok(PyBool::new(py, *b).to_owned().into_any()),
        CategoryLabel::None => Ok(py.None().into_bound(py)),
    }
}

pub fn pyobject_to_category_label(obj: &Bound<'_, PyAny>) -> PyResult<CategoryLabel> {
    if obj.is_instance_of::<PyBool>() {
        let x = obj
            .extract::<bool>()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to extract bool: {e:?}")))?;
        Ok(CategoryLabel::Bool(x))
    } else if obj.is_instance_of::<PyInt>() {
        let x = obj
            .extract::<i64>()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to extract i64: {e:?}")))?;
        Ok(CategoryLabel::Int(x))
    } else if obj.is_instance_of::<PyFloat>() {
        let x = obj
            .extract::<f64>()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to extract f64: {e:?}")))?;
        Ok(CategoryLabel::Float(x))
    } else if obj.is_instance_of::<PyString>() {
        let x = obj
            .extract::<String>()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to extract String: {e:?}")))?;
        Ok(CategoryLabel::String(x))
    } else if obj.is_none() {
        Ok(CategoryLabel::None)
    } else {
        Err(PyRuntimeError::new_err(
            "Unsupported type for categorical choice",
        ))
    }
}

pub fn py_to_external_repr<'py>(
    py: Python<'py>,
    dist: &Distribution,
    internal_repr: f64,
    category_labels: Option<&[CategoryLabel]>,
) -> PyResult<Bound<'py, PyAny>> {
    match dist {
        Distribution::Float { .. } => Ok(internal_repr.into_pyobject(py)?.into_any()),
        Distribution::Int { .. } => Ok((internal_repr as i64).into_pyobject(py)?.into_any()),
        Distribution::Categorical { .. } => {
            if let Some(labels) = category_labels {
                let c = labels
                    .get(internal_repr as usize)
                    .ok_or(PyValueError::new_err(
                        "Internal representation of categorical value is out of range",
                    ))?;
                return category_label_to_pyobject(py, c);
            }
            Ok((internal_repr as i64).into_pyobject(py)?.into_any())
        }
    }
}
