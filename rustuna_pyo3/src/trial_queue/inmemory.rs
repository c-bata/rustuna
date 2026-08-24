use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rustuna_core::trial_queue::{InMemoryTrialQueue, TrialQueue};
use rustuna_core::ErrorKind;
use std::sync::{Arc, RwLock};

use crate::exception::err_to_exceptions;

#[derive(Clone)]
#[pyclass(name = "InMemoryTrialQueue", from_py_object)]
#[pyo3(module = "rustuna")]
pub struct PyInMemoryTrialQueue {
    pub queue: Arc<RwLock<InMemoryTrialQueue>>,
}

impl Default for PyInMemoryTrialQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl PyInMemoryTrialQueue {
    pub fn new() -> Self {
        Self {
            queue: Arc::new(RwLock::new(InMemoryTrialQueue::new())),
        }
    }
}

#[pymethods]
impl PyInMemoryTrialQueue {
    #[new]
    fn py_new() -> Self {
        Self::new()
    }

    fn enqueue(&self, py: Python<'_>, trial_id: u32) -> PyResult<()> {
        py.detach(|| {
            let mut guard = self.queue.write().map_err(|error| {
                PyRuntimeError::new_err(format!("Failed to acquire trial queue lock: {error}"))
            })?;
            guard.enqueue(trial_id).map_err(err_to_exceptions)
        })
    }

    fn dequeue(&self, py: Python<'_>) -> PyResult<Option<u32>> {
        py.detach(|| {
            let mut guard = self.queue.write().map_err(|error| {
                PyRuntimeError::new_err(format!("Failed to acquire trial queue lock: {error}"))
            })?;
            match guard.dequeue() {
                Ok(trial_id) => Ok(Some(trial_id)),
                Err(error) if matches!(error.kind, ErrorKind::TrialQueueEmpty) => Ok(None),
                Err(error) => Err(err_to_exceptions(error)),
            }
        })
    }
}
