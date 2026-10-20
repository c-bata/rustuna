use std::collections::HashMap;
use std::sync::Arc;

/// Bounded least-recently-used cache for decoded attribute values.
pub(crate) struct DecodeCache<V> {
    indices: HashMap<Arc<str>, usize>,
    entries: Vec<Entry<V>>,
    head: Option<usize>,
    tail: Option<usize>,
    capacity: usize,
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
        }
    }

    pub(crate) fn get(&mut self, text: &str) -> Option<&V> {
        let index = *self.indices.get(text)?;
        self.move_to_back(index);
        Some(&self.entries[index].value)
    }

    /// Returns a replaced or evicted value so the caller can drop it outside the cache lock.
    pub(crate) fn insert(&mut self, text: &str, value: V) -> Option<V> {
        if let Some(&index) = self.indices.get(text) {
            self.move_to_back(index);
            return Some(std::mem::replace(&mut self.entries[index].value, value));
        }

        let index = if self.entries.len() == self.capacity {
            let index = self.head.expect("a full cache has a least-recently-used entry");
            self.unlink(index);
            self.indices.remove(self.entries[index].key.as_ref());
            index
        } else {
            self.entries.len()
        };

        let key: Arc<str> = text.into();
        let entry = Entry {
            key: key.clone(),
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

// The map and entry share each key allocation. Eviction recycles a slot so all cache storage
// remains bounded by `capacity`.
struct Entry<V> {
    key: Arc<str>,
    value: V,
    prev: Option<usize>,
    next: Option<usize>,
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
            read(&mut cache, &key);
            assert_lru_order(&cache, &[&key]);
        }
    }

    #[test]
    fn recently_used_value_survives_an_interleaved_scan() {
        let mut cache = DecodeCache::new(128);
        for i in 0..4096 {
            read(&mut cache, "hot");
            read(&mut cache, &format!("cold-{i}"));
            assert!(read(&mut cache, "hot"));
            assert!(cache.entries.len() <= 128);
        }
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
}
