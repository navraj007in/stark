//! WP-C6.3d — STARK Core `HashMap`/`HashSet`.
//!
//! **CE4 representation (CD-132, owner decision): an INSERTION-ORDERED vector of entries, with key
//! identity decided by a linear scan using STARK `Eq`.** This mirrors the HIR interpreter's
//! `InsertionMap` structurally, so the engines satisfy the normative contract by construction rather
//! than by keeping a second index consistent.
//!
//! A host `HashMap` is unusable here for three independent reasons: `RandomState` seeds per process
//! so iteration order varies between RUNS; it would key on Rust's `Hash`/`Eq` rather than STARK's
//! lawful `Eq` (which dispatches to a user impl); and rehashing on growth reorders iteration, which
//! STARK requires be capacity-independent.
//!
//! **`Hash` is never consulted here.** STD-HASH-001 permits hashing only to narrow candidates, and
//! CE4 chose not to: hash-narrowing and `Eq`-only scanning differ observably when a user's `Hash`
//! violates its law, and the three engines must agree with each other. The spec's FNV-1a lives in
//! the `Hash` implementations, for direct `Hash::hash` calls — not in map storage.
//!
//! Identity is supplied per call as a comparator, so the map never decides it: the backend passes
//! the user's selected `Eq::eq` for a nominal key, or [`structural_eq`] for a primitive/`String`
//! key, whose structural comparison IS its lawful `Eq`.

/// An insertion-ordered map, held as PARALLEL vectors: `keys[i]` pairs with `values[i]`, and that
/// index order is the iteration order — the only order there is.
///
/// Parallel vectors rather than a `Vec<(K, V)>` because STARK types the keys cursor as
/// `KeysIter<K>`, with no `V` to name: a cursor over `&[K]` is expressible, one over `&[(K, V)]` is
/// not. The representation is otherwise the CE4 decision unchanged.
pub struct StarkMap<K, V> {
    keys: Vec<K>,
    values: Vec<V>,
}

/// The comparator a map operation decides key identity with.
pub type KeyEq<K> = fn(&K, &K) -> bool;

/// Key identity for a primitive or `String` key: Rust's `==`, which for those types IS their lawful
/// STARK `Eq`. A user nominal never reaches this — it must carry an `impl Eq` to be a key at all,
/// and the backend passes that impl instead.
pub fn structural_eq<K: PartialEq>(a: &K, b: &K) -> bool {
    a == b
}

pub fn new<K, V>() -> StarkMap<K, V> {
    StarkMap {
        keys: Vec::new(),
        values: Vec::new(),
    }
}

/// The index of the entry whose key equals `key`, scanning in first-insertion order so the FIRST
/// match wins (STD-HASH-001: an equal key retains the originally stored key and its position).
fn position<K, V>(map: &StarkMap<K, V>, key: &K, eq: KeyEq<K>) -> Option<usize> {
    map.keys.iter().position(|k| eq(k, key))
}

/// `insert` — appends a first-time key, or replaces an existing key's VALUE while keeping both its
/// position and the ORIGINALLY STORED KEY (STD-HASH-001). Returns the replaced value.
pub fn insert<K, V>(map: &mut StarkMap<K, V>, key: K, value: V, eq: KeyEq<K>) -> Option<V> {
    match position(map, &key, eq) {
        // The stored KEY is untouched — only the value is replaced (STD-HASH-001).
        Some(index) => Some(std::mem::replace(&mut map.values[index], value)),
        None => {
            map.keys.push(key);
            map.values.push(value);
            None
        }
    }
}

/// `get` — an interior borrow of the value, or `None`.
pub fn get<'a, K, V>(map: &'a StarkMap<K, V>, key: &K, eq: KeyEq<K>) -> Option<&'a V> {
    position(map, key, eq).map(|index| &map.values[index])
}

pub fn contains_key<K, V>(map: &StarkMap<K, V>, key: &K, eq: KeyEq<K>) -> bool {
    position(map, key, eq).is_some()
}

pub fn len<K, V>(map: &StarkMap<K, V>) -> u64 {
    map.keys.len() as u64
}

pub fn is_empty<K, V>(map: &StarkMap<K, V>) -> bool {
    map.keys.is_empty()
}

/// `remove` — deletes the entry whose key equals `key`, returning its value.
///
/// The remaining entries keep their relative order, so this is a `Vec::remove` rather than a
/// `swap_remove`: 06-Standard-Library's iteration-order rule is normative, and it says removing a
/// key and later re-inserting it places the key at the END as a new insertion. A swap would move
/// an unrelated entry into the hole and change the order of keys nobody touched.
pub fn remove<K, V>(map: &mut StarkMap<K, V>, key: &K, eq: KeyEq<K>) -> Option<V> {
    let index = position(map, key, eq)?;
    map.keys.remove(index);
    Some(map.values.remove(index))
}

pub fn clear<K, V>(map: &mut StarkMap<K, V>) {
    map.keys.clear();
    map.values.clear();
}

// ------------------------------------------------------------------ HashSet --
//
// DEV-116. `HashSet<T>` is `StarkMap<T, ()>`: the element IS the key, so uniqueness is decided by
// exactly the same `Eq` comparator dispatch, and first-insertion iteration order comes for free.
// Reusing the map rather than writing a parallel container is what keeps STD-HASH-001 true of both
// by construction — a second implementation would be a second place for identity to drift.

/// `insert` — `true` when the value was NOT already present.
///
/// Note the inversion: the map returns the REPLACED value, so `None` means "first time". A set
/// reports that as `true`, and an already-present value leaves the ORIGINALLY STORED element in
/// place (STD-HASH-001), which matters when two values are equal by a user `Eq` but structurally
/// different.
pub fn set_insert<T>(set: &mut StarkMap<T, ()>, value: T, eq: KeyEq<T>) -> bool {
    insert(set, value, (), eq).is_none()
}

/// `remove` — `true` when the value was present.
pub fn set_remove<T>(set: &mut StarkMap<T, ()>, value: &T, eq: KeyEq<T>) -> bool {
    remove(set, value, eq).is_some()
}

/// WP-C6.3c/d: a borrowed cursor over the keys, in insertion order — the same shape as
/// [`crate::vec::VecIter`], and like it, `next` lends out of the SOURCE rather than out of the
/// `&mut` borrow of the cursor.
pub struct KeysIter<'a, K> {
    keys: &'a [K],
    index: usize,
}

pub fn keys_iter_new<K, V>(map: &StarkMap<K, V>) -> KeysIter<'_, K> {
    KeysIter {
        keys: &map.keys,
        index: 0,
    }
}

pub fn keys_iter_next<'a, K>(it: &mut KeysIter<'a, K>) -> Option<&'a K> {
    let key = it.keys.get(it.index)?;
    it.index += 1;
    Some(key)
}
