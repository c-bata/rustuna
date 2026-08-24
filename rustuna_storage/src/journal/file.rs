use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Seek, SeekFrom, Write};
#[cfg(target_os = "macos")]
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use rustuna_core::{Error, ErrorKind, Result};
use serde::Serialize;

use super::{JournalBackend, JournalLog};

const LOCK_FILE_SUFFIX: &str = ".lock";
const RENAME_FILE_SUFFIX: &str = ".rename";
const LOG_OFFSET_CHECKPOINT_INTERVAL: usize = 4096;

fn fsync_file(file: &File) -> std::io::Result<()> {
    #[cfg(target_os = "macos")]
    {
        // NOTE:
        // Rust's File::sync_all() uses F_FULLFSYNC on macOS, which can be very slow.
        // Using fsync(2) is faster but may provide weaker durability guarantees
        // (e.g., device internal cache might not be flushed on power loss).
        loop {
            let rc = unsafe { libc::fsync(file.as_raw_fd()) };
            if rc == 0 {
                return Ok(());
            }
            let err = std::io::Error::last_os_error();
            if err.kind() == std::io::ErrorKind::Interrupted {
                continue; // retry on EINTR
            }
            return Err(err);
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        file.sync_all()
    }
}

pub trait JournalFileLock: Send + Sync {
    /// Acquires the file lock in a blocking manner.
    fn acquire(&self) -> Result<()>;
    /// Releases the file lock.
    fn release(&self) -> Result<()>;
}

#[cfg(unix)]
fn default_journal_file_lock(file_path: &Path) -> Box<dyn JournalFileLock> {
    Box::new(JournalFileSymlinkLock::new(file_path))
}

#[cfg(not(unix))]
fn default_journal_file_lock(file_path: &Path) -> Box<dyn JournalFileLock> {
    Box::new(JournalFileOpenLock::new(file_path))
}

/// File-based backend for [`super::storage::JournalStorage`].
///
/// Logs are appended as newline-delimited JSON records. A pluggable lock implementation is used
/// to coordinate concurrent writers across processes.
pub struct JournalFileBackend {
    file_path: PathBuf,
    lock: Box<dyn JournalFileLock>,
    latest_log_number: usize,
    latest_log_offset: u64,
    log_offset_checkpoints: Vec<u64>,
}

impl JournalFileBackend {
    /// Creates a file-backed journal backend.
    pub fn new(
        file_path: impl AsRef<Path>,
        lock: Option<Box<dyn JournalFileLock>>,
    ) -> Result<Self> {
        let file_path = file_path.as_ref().to_path_buf();
        if !file_path.exists() {
            File::create(&file_path)
                .map_err(|e| Error::with_reason(ErrorKind::StorageError, e.to_string()))?;
        }
        let lock = lock.unwrap_or_else(|| default_journal_file_lock(&file_path));
        Ok(JournalFileBackend {
            file_path,
            lock,
            latest_log_number: 0,
            latest_log_offset: 0,
            log_offset_checkpoints: vec![0],
        })
    }

    fn read_start(&self, log_number_from: usize) -> (usize, u64) {
        if log_number_from >= self.latest_log_number {
            return (self.latest_log_number, self.latest_log_offset);
        }

        let checkpoint_index = log_number_from / LOG_OFFSET_CHECKPOINT_INTERVAL;
        match self.log_offset_checkpoints.get(checkpoint_index) {
            Some(offset) => (checkpoint_index * LOG_OFFSET_CHECKPOINT_INTERVAL, *offset),
            None => (0, 0),
        }
    }

    fn record_log_offset(&mut self, log_number: usize, offset: u64) {
        if log_number > self.latest_log_number {
            self.latest_log_number = log_number;
            self.latest_log_offset = offset;
        }

        if !log_number.is_multiple_of(LOG_OFFSET_CHECKPOINT_INTERVAL) {
            return;
        }
        let checkpoint_index = log_number / LOG_OFFSET_CHECKPOINT_INTERVAL;
        if checkpoint_index == self.log_offset_checkpoints.len() {
            self.log_offset_checkpoints.push(offset);
        } else if let Some(existing_offset) = self.log_offset_checkpoints.get(checkpoint_index) {
            debug_assert_eq!(*existing_offset, offset);
        }
    }
}

impl JournalBackend for JournalFileBackend {
    fn read_logs(
        &mut self,
        log_number_from: usize,
        handler: &mut dyn FnMut(JournalLog) -> Result<()>,
    ) -> Result<()> {
        let mut file = File::open(&self.file_path)
            .map_err(|e| Error::with_reason(ErrorKind::StorageError, e.to_string()))?;
        let file_size = file
            .metadata()
            .map_err(|e| Error::with_reason(ErrorKind::StorageError, e.to_string()))?
            .len();
        let mut remaining_log_size = file_size as i64;

        let (log_number_start, offset_start) = self.read_start(log_number_from);
        file.seek(SeekFrom::Start(offset_start))
            .map_err(|e| Error::with_reason(ErrorKind::StorageError, e.to_string()))?;
        remaining_log_size -= offset_start as i64;

        let mut reader = BufReader::new(file);
        let mut last_decode_error: Option<Error> = None;
        let mut log_number = log_number_start;
        let mut offset = offset_start;
        let mut line = Vec::new();
        loop {
            line.clear();
            let bytes = reader.read_until(b'\n', &mut line).map_err(|e| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Failed to read log file: {e}"),
                )
            })?;
            if bytes == 0 {
                break;
            }

            let byte_len = line.len() as u64;
            remaining_log_size -= byte_len as i64;
            if remaining_log_size < 0 {
                break;
            }
            if let Some(err) = last_decode_error.take() {
                return Err(err);
            }

            if !line.ends_with(b"\n") {
                last_decode_error = Some(Error::with_reason(
                    ErrorKind::StorageError,
                    "Invalid log format.".to_string(),
                ));
                log_number += 1;
                continue;
            }

            if log_number < log_number_from {
                offset = offset.saturating_add(byte_len);
                log_number += 1;
                self.record_log_offset(log_number, offset);
                continue;
            }

            match decode_log_line(&line) {
                Ok(log) => {
                    handler(log)?;
                    offset = offset.saturating_add(byte_len);
                    log_number += 1;
                    self.record_log_offset(log_number, offset);
                }
                Err(err) => {
                    last_decode_error = Some(err);
                    log_number += 1;
                }
            }
        }

        Ok(())
    }
    fn append_logs(&mut self, logs: &[JournalLog]) -> Result<()> {
        let _guard = JournalFileLockGuard::new(self.lock.as_ref())?;
        let what_to_write = encode_logs(logs)?;
        let mut file = OpenOptions::new()
            .append(true)
            .open(&self.file_path)
            .map_err(|e| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Failed to open a file: {e}"),
                )
            })?;
        file.write_all(&what_to_write).map_err(|e| {
            Error::with_reason(
                ErrorKind::StorageError,
                format!("Failed to write logs: {e}"),
            )
        })?;
        file.flush().map_err(|e| {
            Error::with_reason(ErrorKind::StorageError, format!("Failed to flush: {e}"))
        })?;
        fsync_file(&file).map_err(|e| {
            Error::with_reason(ErrorKind::StorageError, format!("Failed to fsync: {e}"))
        })?;
        Ok(())
    }
}

/// Lock implementation based on symlink creation.
///
/// This lock creates a symlink next to the journal file and removes it when the lock is released.
/// Similar to Optuna's symlink-based journal lock, this variant is intended for environments
/// where symlink creation provides a more portable inter-process exclusion mechanism than
/// exclusive file creation.
#[cfg(unix)]
pub struct JournalFileSymlinkLock {
    lock_target_file: PathBuf,
    lock_file: PathBuf,
}

#[cfg(unix)]
impl JournalFileSymlinkLock {
    /// Creates a symlink-based lock next to the journal file.
    pub fn new(filepath: impl AsRef<Path>) -> Self {
        let filepath = filepath.as_ref().to_path_buf();
        JournalFileSymlinkLock {
            lock_target_file: filepath.clone(),
            lock_file: PathBuf::from(format!("{}{}", filepath.display(), LOCK_FILE_SUFFIX)),
        }
    }
}

#[cfg(unix)]
impl JournalFileLock for JournalFileSymlinkLock {
    /// Acquires the lock by creating a symlink in a blocking retry loop.
    fn acquire(&self) -> Result<()> {
        let mut sleep_secs = 0.001f64;
        loop {
            match std::os::unix::fs::symlink(&self.lock_target_file, &self.lock_file) {
                Ok(_) => return Ok(()),
                Err(err) => {
                    if err.kind() == std::io::ErrorKind::AlreadyExists {
                        thread::sleep(Duration::from_secs_f64(sleep_secs));
                        sleep_secs = (sleep_secs * 2.0).min(1.0);
                        continue;
                    }
                    return Err(Error::with_reason(
                        ErrorKind::StorageError,
                        "Failed to create a symlink filej",
                    ));
                }
            }
        }
    }

    /// Releases the lock by renaming and removing the symlink.
    fn release(&self) -> Result<()> {
        let rename_file = PathBuf::from(format!(
            "{}{}{}",
            self.lock_file.display(),
            unique_suffix(),
            RENAME_FILE_SUFFIX
        ));
        match fs::rename(&self.lock_file, &rename_file) {
            Ok(_) => {}
            Err(e) => {
                let _ = fs::remove_file(&rename_file);
                return Err(Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Failed to rename lock file: {e}"),
                ));
            }
        }
        match fs::remove_file(&rename_file) {
            Ok(_) => Ok(()),
            Err(e) => {
                let _ = fs::remove_file(&rename_file);
                Err(Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Failed to remove rename file: {e}"),
                ))
            }
        }
    }
}

/// Lock implementation based on exclusive file creation.
///
/// This lock creates a dedicated lock file with exclusive open semantics and removes it when the
/// lock is released. Similar to Optuna's open-based journal lock, this variant is suitable for
/// environments where `O_EXCL` style file creation can be relied on for process synchronization.
pub struct JournalFileOpenLock {
    lock_file: PathBuf,
}

impl JournalFileOpenLock {
    /// Creates an open-file-based lock next to the journal file.
    pub fn new(filepath: impl AsRef<Path>) -> Self {
        let filepath = filepath.as_ref().to_path_buf();
        JournalFileOpenLock {
            lock_file: PathBuf::from(format!("{}{}", filepath.display(), LOCK_FILE_SUFFIX)),
        }
    }
}

impl JournalFileLock for JournalFileOpenLock {
    /// Acquires the lock by creating the lock file in a blocking retry loop.
    fn acquire(&self) -> Result<()> {
        let mut sleep_secs = 0.001f64;
        loop {
            let res = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&self.lock_file);
            match res {
                Ok(_) => return Ok(()),
                Err(err) => {
                    if err.kind() == std::io::ErrorKind::AlreadyExists {
                        thread::sleep(Duration::from_secs_f64(sleep_secs));
                        sleep_secs = (sleep_secs * 2.0).min(1.0);
                        continue;
                    }
                    return Err(Error::with_reason(ErrorKind::StorageError, err.to_string()));
                }
            }
        }
    }

    /// Releases the lock by removing the created lock file.
    fn release(&self) -> Result<()> {
        let rename_file = PathBuf::from(format!(
            "{}{}{}",
            self.lock_file.display(),
            unique_suffix(),
            RENAME_FILE_SUFFIX
        ));
        match fs::rename(&self.lock_file, &rename_file) {
            Ok(_) => {}
            Err(e) => {
                let _ = fs::remove_file(&rename_file);
                return Err(Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Failed to rename lock file: {e}"),
                ));
            }
        }
        match fs::remove_file(&rename_file) {
            Ok(_) => Ok(()),
            Err(e) => {
                let _ = fs::remove_file(&rename_file);
                Err(Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Failed to remove rename file: {e}"),
                ))
            }
        }
    }
}

struct JournalFileLockGuard<'a> {
    lock: &'a dyn JournalFileLock,
}

impl<'a> JournalFileLockGuard<'a> {
    fn new(lock: &'a dyn JournalFileLock) -> Result<Self> {
        lock.acquire()?;
        Ok(JournalFileLockGuard { lock })
    }
}

impl<'a> Drop for JournalFileLockGuard<'a> {
    fn drop(&mut self) {
        let _ = self.lock.release();
    }
}

fn unique_suffix() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!(".{}-{:?}", nanos, thread::current().id())
}

pub fn encode_logs(logs: &[JournalLog]) -> Result<Vec<u8>> {
    let mut result = Vec::new();
    for (i, log) in logs.iter().enumerate() {
        if i > 0 {
            result.push(b'\n');
        }
        let mut buf = Vec::new();
        let mut serializer =
            serde_json::Serializer::with_formatter(&mut buf, serde_json::ser::CompactFormatter);
        log.serialize(&mut serializer)
            .map_err(|e| Error::with_reason(ErrorKind::StorageError, e.to_string()))?;
        result.extend_from_slice(&buf);
    }
    result.push(b'\n');
    Ok(result)
}

pub fn decode_log_line(line: &[u8]) -> Result<JournalLog> {
    let mut slice = line;
    if slice.ends_with(b"\n") {
        slice = &slice[..slice.len() - 1];
    }
    if slice.ends_with(b"\r") {
        slice = &slice[..slice.len() - 1];
    }
    serde_json::from_slice(slice)
        .map_err(|e| Error::with_reason(ErrorKind::StorageError, e.to_string()))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::test_utils::TempDir;

    fn test_log(number: usize) -> JournalLog {
        JournalLog {
            op_code: 0,
            worker_id: format!("worker-{number}"),
            fields: HashMap::new(),
        }
    }

    #[test]
    fn retains_sparse_offsets_and_reads_incrementally() -> Result<()> {
        let dir =
            TempDir::new().map_err(|e| Error::with_reason(ErrorKind::Unexpected, e.to_string()))?;
        let path = dir.path().join("journal.log");
        let mut backend = JournalFileBackend::new(&path, None)?;
        let num_logs = LOG_OFFSET_CHECKPOINT_INTERVAL * 2 + 3;
        let logs = (0..num_logs).map(test_log).collect::<Vec<_>>();
        backend.append_logs(&logs)?;

        let mut num_read = 0;
        backend.read_logs(0, &mut |_| {
            num_read += 1;
            Ok(())
        })?;
        assert_eq!(num_read, num_logs);
        assert_eq!(backend.latest_log_number, num_logs);
        assert_eq!(backend.log_offset_checkpoints.len(), 3);
        assert_eq!(
            backend.read_start(LOG_OFFSET_CHECKPOINT_INTERVAL + 1).0,
            LOG_OFFSET_CHECKPOINT_INTERVAL
        );

        let mut worker_ids = Vec::new();
        backend.read_logs(LOG_OFFSET_CHECKPOINT_INTERVAL + 1, &mut |log| {
            worker_ids.push(log.worker_id);
            Ok(())
        })?;
        assert_eq!(
            worker_ids,
            ((LOG_OFFSET_CHECKPOINT_INTERVAL + 1)..num_logs)
                .map(|number| format!("worker-{number}"))
                .collect::<Vec<_>>()
        );

        backend.append_logs(&[test_log(num_logs)])?;
        num_read = 0;
        backend.read_logs(num_logs, &mut |_| {
            num_read += 1;
            Ok(())
        })?;
        assert_eq!(num_read, 1);
        assert_eq!(backend.latest_log_number, num_logs + 1);
        assert_eq!(backend.log_offset_checkpoints.len(), 3);
        Ok(())
    }

    #[test]
    fn reads_from_an_arbitrary_log_number() -> Result<()> {
        let dir =
            TempDir::new().map_err(|e| Error::with_reason(ErrorKind::Unexpected, e.to_string()))?;
        let path = dir.path().join("journal.log");
        let mut backend = JournalFileBackend::new(&path, None)?;
        let logs = (0..10).map(test_log).collect::<Vec<_>>();
        backend.append_logs(&logs)?;

        let mut worker_ids = Vec::new();
        backend.read_logs(4, &mut |log| {
            worker_ids.push(log.worker_id);
            Ok(())
        })?;
        assert_eq!(
            worker_ids,
            (4..10)
                .map(|number| format!("worker-{number}"))
                .collect::<Vec<_>>()
        );

        worker_ids.clear();
        backend.read_logs(2, &mut |log| {
            worker_ids.push(log.worker_id);
            Ok(())
        })?;
        assert_eq!(
            worker_ids,
            (2..10)
                .map(|number| format!("worker-{number}"))
                .collect::<Vec<_>>()
        );
        Ok(())
    }

    #[test]
    fn does_not_cache_an_incomplete_log_before_requested_number() -> Result<()> {
        let dir =
            TempDir::new().map_err(|e| Error::with_reason(ErrorKind::Unexpected, e.to_string()))?;
        let path = dir.path().join("journal.log");
        let mut backend = JournalFileBackend::new(&path, None)?;
        backend.append_logs(&[test_log(0), test_log(1)])?;

        let log2 = encode_logs(&[test_log(2)])?;
        let split_at = log2.len() / 2;
        let mut file = OpenOptions::new()
            .append(true)
            .open(&path)
            .map_err(|e| Error::with_reason(ErrorKind::Unexpected, e.to_string()))?;
        file.write_all(&log2[..split_at])
            .map_err(|e| Error::with_reason(ErrorKind::Unexpected, e.to_string()))?;
        drop(file);

        let mut worker_ids = Vec::new();
        backend.read_logs(3, &mut |log| {
            worker_ids.push(log.worker_id);
            Ok(())
        })?;
        assert!(worker_ids.is_empty());
        assert_eq!(backend.latest_log_number, 2);

        let mut file = OpenOptions::new()
            .append(true)
            .open(&path)
            .map_err(|e| Error::with_reason(ErrorKind::Unexpected, e.to_string()))?;
        file.write_all(&log2[split_at..])
            .map_err(|e| Error::with_reason(ErrorKind::Unexpected, e.to_string()))?;
        file.write_all(&encode_logs(&[test_log(3)])?)
            .map_err(|e| Error::with_reason(ErrorKind::Unexpected, e.to_string()))?;
        drop(file);

        backend.read_logs(3, &mut |log| {
            worker_ids.push(log.worker_id);
            Ok(())
        })?;
        assert_eq!(worker_ids, vec!["worker-3"]);
        Ok(())
    }

    #[test]
    fn retries_an_incomplete_trailing_log() -> Result<()> {
        let dir =
            TempDir::new().map_err(|e| Error::with_reason(ErrorKind::Unexpected, e.to_string()))?;
        let path = dir.path().join("journal.log");
        let mut backend = JournalFileBackend::new(&path, None)?;
        backend.append_logs(&[test_log(0), test_log(1)])?;

        let trailing_log = encode_logs(&[test_log(2)])?;
        let split_at = trailing_log.len() / 2;
        let mut file = OpenOptions::new()
            .append(true)
            .open(&path)
            .map_err(|e| Error::with_reason(ErrorKind::Unexpected, e.to_string()))?;
        file.write_all(&trailing_log[..split_at])
            .map_err(|e| Error::with_reason(ErrorKind::Unexpected, e.to_string()))?;
        drop(file);

        let mut num_read = 0;
        backend.read_logs(0, &mut |_| {
            num_read += 1;
            Ok(())
        })?;
        assert_eq!(num_read, 2);
        assert_eq!(backend.latest_log_number, 2);

        let mut file = OpenOptions::new()
            .append(true)
            .open(&path)
            .map_err(|e| Error::with_reason(ErrorKind::Unexpected, e.to_string()))?;
        file.write_all(&trailing_log[split_at..])
            .map_err(|e| Error::with_reason(ErrorKind::Unexpected, e.to_string()))?;
        drop(file);

        num_read = 0;
        backend.read_logs(2, &mut |_| {
            num_read += 1;
            Ok(())
        })?;
        assert_eq!(num_read, 1);
        assert_eq!(backend.latest_log_number, 3);
        Ok(())
    }

    #[test]
    fn reports_a_corrupt_log_when_a_later_log_exists() -> Result<()> {
        let dir =
            TempDir::new().map_err(|e| Error::with_reason(ErrorKind::Unexpected, e.to_string()))?;
        let path = dir.path().join("journal.log");
        let mut backend = JournalFileBackend::new(&path, None)?;
        backend.append_logs(&[test_log(0)])?;

        let mut file = OpenOptions::new()
            .append(true)
            .open(&path)
            .map_err(|e| Error::with_reason(ErrorKind::Unexpected, e.to_string()))?;
        file.write_all(b"{\"op_code\":\n")
            .map_err(|e| Error::with_reason(ErrorKind::Unexpected, e.to_string()))?;
        drop(file);
        backend.append_logs(&[test_log(1)])?;

        let mut worker_ids = Vec::new();
        assert!(backend
            .read_logs(0, &mut |log| {
                worker_ids.push(log.worker_id);
                Ok(())
            })
            .is_err());
        assert_eq!(worker_ids, vec!["worker-0"]);
        assert_eq!(backend.latest_log_number, 1);
        Ok(())
    }

    #[test]
    fn retries_a_log_when_the_handler_fails() -> Result<()> {
        let dir =
            TempDir::new().map_err(|e| Error::with_reason(ErrorKind::Unexpected, e.to_string()))?;
        let path = dir.path().join("journal.log");
        let mut backend = JournalFileBackend::new(&path, None)?;
        backend.append_logs(&[test_log(0), test_log(1)])?;

        let result = backend.read_logs(0, &mut |_| {
            Err(Error::with_reason(ErrorKind::Unexpected, "handler failed"))
        });
        assert!(result.is_err());
        assert_eq!(backend.latest_log_number, 0);

        let mut num_read = 0;
        backend.read_logs(0, &mut |_| {
            num_read += 1;
            Ok(())
        })?;
        assert_eq!(num_read, 2);
        assert_eq!(backend.latest_log_number, 2);
        Ok(())
    }
}
