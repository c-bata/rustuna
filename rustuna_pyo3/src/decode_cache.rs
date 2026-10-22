use std::collections::HashMap;
use std::hash::BuildHasher;
use std::sync::Arc;

/// Bounded cache with frequency-based admission (TinyLFU) and LRU victim selection.
///
/// A scan larger than the cache must not evict every reused value. On a miss, admit the
/// candidate only if its estimated frequency exceeds that of the least-recently-used
/// resident. Prefer the resident on ties. Decaying the sketch lets new workloads replace
/// values that are no longer read. The sketch also counts misses, without retaining keys.
pub(crate) struct DecodeCache<V> {
    indices: HashMap<Arc<str>, usize>,
    entries: Vec<Entry<V>>,
    head: Option<usize>,
    tail: Option<usize>,
    capacity: usize,
    frequencies: FrequencySketch,
}

impl<V> DecodeCache<V> {
    pub(crate) fn new(capacity: usize) -> Self {
        assert!(capacity > 0);
        Self {
            indices: HashMap::new(),
            entries: Vec::new(),
            head: None,
            tail: None,
            capacity,
            frequencies: FrequencySketch::new(capacity),
        }
    }

    pub(crate) fn get(&mut self, text: &str) -> Option<&V> {
        if let Some(&index) = self.indices.get(text) {
            self.frequencies.record(self.entries[index].hash);
            self.move_to_back(index);
            Some(&self.entries[index].value)
        } else {
            self.frequencies
                .record(self.indices.hasher().hash_one(text));
            None
        }
    }

    /// Return a replaced/evicted value to be dropped after releasing the cache lock.
    pub(crate) fn insert(&mut self, text: &str, value: V) -> Option<V> {
        if let Some(&index) = self.indices.get(text) {
            self.move_to_back(index);
            return Some(std::mem::replace(&mut self.entries[index].value, value));
        }
        let hash = self.indices.hasher().hash_one(text);
        let index = if self.entries.len() == self.capacity {
            let index = self.head.unwrap();
            if self.frequencies.estimate(hash)
                <= self.frequencies.estimate(self.entries[index].hash)
            {
                return Some(value);
            }
            self.unlink(index);
            self.indices.remove(self.entries[index].key.as_ref());
            index
        } else {
            self.entries.len()
        };
        let key: Arc<str> = text.into();
        let entry = Entry {
            key: key.clone(),
            hash,
            value,
            prev: None,
            next: None,
        };
        let evicted = if index == self.entries.len() {
            self.entries.push(entry);
            None
        } else {
            Some(std::mem::replace(&mut self.entries[index], entry).value)
        };
        self.indices.insert(key, index);
        self.append(index);
        evicted
    }

    fn move_to_back(&mut self, index: usize) {
        if self.tail != Some(index) {
            self.unlink(index);
            self.append(index);
        }
    }

    fn unlink(&mut self, index: usize) {
        let Entry { prev, next, .. } = self.entries[index];
        if let Some(prev) = prev {
            self.entries[prev].next = next;
        } else {
            self.head = next;
        }
        if let Some(next) = next {
            self.entries[next].prev = prev;
        } else {
            self.tail = prev;
        }
    }

    fn append(&mut self, index: usize) {
        self.entries[index].prev = self.tail;
        self.entries[index].next = None;
        if let Some(tail) = self.tail {
            self.entries[tail].next = Some(index);
        } else {
            self.head = Some(index);
        }
        self.tail = Some(index);
    }
}

// Map keys and slots share the text allocation. Eviction recycles a slot, so both the
// hash map and the index-linked LRU list remain bounded by the cache capacity.
struct Entry<V> {
    key: Arc<str>,
    hash: u64,
    value: V,
    prev: Option<usize>,
    next: Option<usize>,
}

struct FrequencySketch {
    rows: [Vec<u8>; 4],
    mask: usize,
    samples: usize,
    reset_after: usize,
}

impl FrequencySketch {
    fn new(capacity: usize) -> Self {
        let width = capacity.max(64).next_power_of_two();
        Self {
            rows: std::array::from_fn(|_| vec![0; width]),
            mask: width - 1,
            samples: 0,
            reset_after: capacity * 10,
        }
    }

    fn record(&mut self, hash: u64) {
        for (i, row) in self.rows.iter_mut().enumerate() {
            let counter = &mut row[(hash.rotate_right(i as u32 * 16) as usize) & self.mask];
            *counter = counter.saturating_add(1).min(15);
        }
        self.samples += 1;
        if self.samples == self.reset_after {
            for row in &mut self.rows {
                for counter in row {
                    *counter /= 2;
                }
            }
            self.samples = 0;
        }
    }

    fn estimate(&self, hash: u64) -> u8 {
        self.rows
            .iter()
            .enumerate()
            .map(|(i, row)| row[(hash.rotate_right(i as u32 * 16) as usize) & self.mask])
            .min()
            .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(cache: &mut DecodeCache<usize>, text: &str) -> bool {
        if cache.get(text).is_some() {
            true
        } else {
            cache.insert(text, 1);
            false
        }
    }

    fn assert_lru_order(cache: &DecodeCache<usize>, expected: &[&str]) {
        let mut index = cache.head;
        let mut prev = None;
        for key in expected {
            let current = index.expect("missing LRU entry");
            let entry = &cache.entries[current];
            assert_eq!(entry.key.as_ref(), *key);
            assert_eq!(entry.prev, prev);
            assert_eq!(cache.indices.get(*key), Some(&current));
            prev = index;
            index = entry.next;
        }
        assert_eq!(index, None, "extra LRU entries or a cycle");
        assert_eq!(cache.tail, prev);
        assert_eq!(cache.indices.len(), expected.len());
        assert_eq!(cache.entries.len(), expected.len());
    }

    #[test]
    fn lru_order_updates_on_hits_and_replacements() {
        let mut cache = DecodeCache::new(3);
        read(&mut cache, "a");
        read(&mut cache, "b");
        read(&mut cache, "c");
        assert_lru_order(&cache, &["a", "b", "c"]);
        assert!(read(&mut cache, "a"));
        assert_lru_order(&cache, &["b", "c", "a"]);
        assert_eq!(cache.insert("c", 2), Some(1));
        assert_lru_order(&cache, &["b", "a", "c"]);
        assert_eq!(cache.get("c"), Some(&2));
        assert_lru_order(&cache, &["b", "a", "c"]);
        for _ in 0..100 {
            read(&mut cache, "d");
        }
        assert_lru_order(&cache, &["a", "c", "d"]);
    }

    #[test]
    fn capacity_one_recycles_the_only_slot() {
        let mut cache = DecodeCache::new(1);
        read(&mut cache, "old");
        assert_lru_order(&cache, &["old"]);
        for i in 0..100 {
            let key = format!("new-{i}");
            for _ in 0..100 {
                read(&mut cache, &key);
            }
            assert_lru_order(&cache, &[&key]);
        }
    }

    #[test]
    fn reused_value_survives_a_large_scan() {
        let mut cache = DecodeCache::new(128);
        for i in 0..4096 {
            read(&mut cache, "hot");
            read(&mut cache, &format!("cold-{i}"));
            assert!(read(&mut cache, "hot"));
            assert!(cache.entries.len() <= 128);
        }
    }

    #[test]
    fn repeated_oversized_scan_retains_hits() {
        let mut cache = DecodeCache::new(128);
        let mut hits = 0;
        for pass in 0..4 {
            for i in 0..512 {
                let hit = read(&mut cache, &format!("key-{i}"));
                if pass > 0 && hit {
                    hits += 1;
                }
            }
        }
        // Pure LRU misses every time after the first pass on this workload.
        assert!(hits > 64, "only {hits} retained hits");
        assert_eq!(cache.entries.len(), 128);
    }

    #[test]
    fn new_workload_replaces_old_residents() {
        let mut cache = DecodeCache::new(128);
        for _ in 0..10 {
            for i in 0..128 {
                read(&mut cache, &format!("old-{i}"));
            }
        }
        for _ in 0..10000 {
            read(&mut cache, "new-hot");
        }
        assert!(cache.get("new-hot").is_some());
        assert_eq!(cache.entries.len(), 128);
    }

    #[test]
    fn frequency_estimates_decay_and_saturate() {
        let mut sketch = FrequencySketch::new(128);
        for _ in 0..100 {
            sketch.record(1);
        }
        assert_eq!(sketch.estimate(1), 15);
        for _ in 100..1280 {
            sketch.record(2);
        }
        assert_eq!(sketch.estimate(1), 7);
        assert_eq!(sketch.estimate(2), 7);
    }
}
