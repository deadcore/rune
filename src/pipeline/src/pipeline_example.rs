//! A tool for constructing multi-threaded pipelines of execution
//!
//! A `Pipeline` consists in one or more stages that each runs in its own thread (or multiple
//! threads). They take in items from the previous stage and produce items for the next stage,
//! similar to a Unix pipeline. This allows for expressing computation as a series of steps that
//! feed into each other and run concurrently
//!
//! # Examples
//!
//! Build the first 10 fibonacci numbers:
//!
//! ```rust
//! use pipelines::Pipeline;
//!
//! fn fibonacci(n:u64)->u64{if n<2 {1} else {fibonacci(n-1) + fibonacci(n-2)}}
//!
//! let nums: Vec<u64> = (0..10).collect();
//! let fibs: Vec<u64> = Pipeline::from(nums)
//!     .map(fibonacci)
//!     .into_iter().collect();
//! ```
//!
//! Build the first 10 fibonacci numbers in parallel, then double them:
//!
//! ```rust
//! use pipelines::Pipeline;
//!
//! let workers = 2;
//! fn fibonacci(n:u64)->u64{if n<2 {1} else {fibonacci(n-1) + fibonacci(n-2)}}
//!
//! let nums: Vec<u64> = (0..10).collect();
//! let fibs: Vec<u64> = Pipeline::from(nums)
//!     .pmap(workers, fibonacci)
//!     .map(|x| x*2)
//!     .into_iter().collect();
//! ```
//!
//! Build the first 10 fibonacci numbers in parallel then group them by evenness, expressed in
//! mapreduce stages
//!
//! ```rust
//! use pipelines::Pipeline;
//!
//! let workers = 2;
//! fn fibonacci(n:u64)->u64{if n<2 {1} else {fibonacci(n-1) + fibonacci(n-2)}}
//!
//! let nums: Vec<u64> = (0..10).collect();
//! let fibs: Vec<(bool, u64)> = Pipeline::from(nums)
//!     .pmap(workers, fibonacci)
//!     .map(|num| (num % 2 == 0, num))
//!     .preduce(workers, |evenness, nums| (evenness, *nums.iter().max().unwrap()))
//!     .into_iter().collect();
//! ```

// HEADUPS: Keep that ^^ in sync with README.md

#[cfg(feature = "chan")]
extern crate chan;

use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;
use std::thread;

pub use filter::Filter;
pub use map::Mapper;
pub use multiplex::Multiplex;
pub use comms::{LockedReceiver, Receiver, ReceiverIntoIterator, Sender};

mod comms {
    use std::cell::RefCell;
    use std::collections::VecDeque;
    use std::sync::mpsc;
    use std::sync::{Arc, Mutex};

    use super::PipelineConfig;

    /// Passed to pipelines as their place to send results
    #[derive(Debug)]
    pub struct Sender<Out> {
        tx: mpsc::SyncSender<VecDeque<Out>>,
        config: PipelineConfig,
        // wrapped in a refcell so we can send using immutable references, like SyncSender does
        buffer: RefCell<VecDeque<Out>>,
    }

    impl<Out> Sender<Out> {
        /// Transmit a value to the next stage in the pipeline
        ///
        /// Panics on failure
        pub fn send(&self, out: Out) -> () {
            let new_len = {
                let mut buff = self.buffer.borrow_mut();
                buff.push_back(out);
                buff.len()
            };
            if new_len >= self.config.batch_size {
                self.flush()
            }
        }

        /// Send any unsent data sitting in the buffer
        ///
        /// Panics on failure to send
        pub fn flush(&self) {
            let old_buffer = self.buffer
                                 .replace(VecDeque::with_capacity(self.config.batch_size));
            if old_buffer.len() > 0 {
                self.tx.send(old_buffer).expect("failed send");
            }
        }

        pub(super) fn pair(config: PipelineConfig) -> (Self, Receiver<Out>) {
            let (tx, rx) = mpsc::sync_channel(config.buff_size);
            let tx_buffer = VecDeque::with_capacity(config.batch_size);
            let rx_buffer = VecDeque::with_capacity(config.batch_size);
            (
                Self {
                    tx,
                    config,
                    buffer: RefCell::new(tx_buffer),
                },
                Receiver {
                    rx,
                    buffer: RefCell::new(rx_buffer),
                },
            )
        }
    }

    impl<Out> Drop for Sender<Out> {
        fn drop(&mut self) {
            self.flush()
        }
    }

    impl<Out> Clone for Sender<Out> {
        fn clone(&self) -> Self {
            Self {
                tx: self.tx.clone(),
                config: self.config.clone(),
                buffer: RefCell::new(VecDeque::with_capacity(
                    self.config.buff_size,
                )),
            }
        }
    }

    /// Passed to pipelines as their place to get incoming data from the previous stage.
    ///
    /// It's possible to use by calling `recv` directly, but is primarily for its `into_iter`
    #[derive(Debug)]
    pub struct Receiver<In> {
        rx: mpsc::Receiver<VecDeque<In>>,
        buffer: RefCell<VecDeque<In>>,
    }

    impl<In> Receiver<In> {
        /// Get an item from the previous stage
        ///
        /// returns None if the remote side has hung up and all data has been received
        pub fn recv(&mut self) -> Option<In> {
            let current_len = {
                let buff = self.buffer.borrow();
                buff.len()
            };
            if current_len > 0 {
                // there's already data in the buffer so we don't have to do anything
                return self.buffer.get_mut().pop_front();
            }

            // no data in the buffer, get some from the pipe
            match self.rx.recv() {
                Ok(val) => {
                    self.buffer.replace(val);
                }
                Err(_recv_err) => return None,
            }

            let current_len = {
                let buff = self.buffer.borrow();
                buff.len()
            };
            // now we should have data in the buffer and can use it
            if current_len == 0 {
                // I guess we got an empty VecDeque? this shouldn't happen
                return None;
            } else {
                return self.buffer.get_mut().pop_front();
            }
        }

        fn recv_buff(&mut self) -> Option<VecDeque<In>> {
            // receive a whole buffer of the batch size

            let current_len = {
                let buff = self.buffer.borrow();
                buff.len()
            };
            if current_len > 0 {
                // if we have a nonzero buffer already, return it and make a new one for ourselves
                return Some(self.buffer.replace(VecDeque::new()));
            }

            // otherwise, pull a buffer from the pipe
            match self.rx.recv() {
                Ok(val) => {
                    // return the one we just received. this leaves our own 0-sized buffer in place
                    // but that's okay
                    return Some(val);
                }
                Err(_recv_err) => return None,
            }
        }
    }

    impl<In> IntoIterator for Receiver<In> {
        type Item = In;
        type IntoIter = ReceiverIntoIterator<In>;

        fn into_iter(self) -> Self::IntoIter {
            ReceiverIntoIterator {
                iter: self.rx.into_iter(),
                buffer: self.buffer.into_inner(),
            }
        }
    }

    pub struct ReceiverIntoIterator<In> {
        iter: mpsc::IntoIter<VecDeque<In>>,
        buffer: VecDeque<In>,
    }

    impl<In> Iterator for ReceiverIntoIterator<In> {
        type Item = In;

        fn next(&mut self) -> Option<In> {
            if self.buffer.len() == 0 {
                // buffer is empty. fill it
                match self.iter.next() {
                    Some(buff) => {
                        self.buffer = buff;
                    }
                    None => {
                        return None;
                    }
                }
            }
            return self.buffer.pop_front();
        }
    }

    #[derive(Debug)]
    pub struct LockedReceiver<T>
        where
            T: Send + 'static,
    {
        lockbox: Arc<Mutex<Receiver<T>>>,
        buffer: VecDeque<T>,
    }

    impl<T> LockedReceiver<T>
        where
            T: Send,
    {
        pub fn new(recv: Receiver<T>) -> Self {
            Self {
                lockbox: Arc::new(Mutex::new(recv)),
                buffer: VecDeque::new(),
            }
        }
    }

    impl<T> Clone for LockedReceiver<T>
        where
            T: Send,
    {
        fn clone(&self) -> Self {
            Self {
                lockbox: self.lockbox.clone(),
                buffer: VecDeque::new(),
            }
        }
    }

    impl<T> Iterator for LockedReceiver<T>
        where
            T: Send,
    {
        type Item = T;

        fn next(&mut self) -> Option<T> {
            if self.buffer.len() == 0 {
                match self.lockbox
                          .lock()
                          .expect("failed unwrap mutex")
                          .recv_buff()
                {
                    Some(buff) => self.buffer = buff,
                    None => {
                        return None;
                    }
                }
            }
            return self.buffer.pop_front();
        }
    }
}

/// Configuration for buffers internal to the Pipeline
///
/// Each stage inherits the configuration from its previous state. As a result, this configures
/// future stages, not past
///
/// # Example
///
/// ```rust
/// use pipelines::{Pipeline, PipelineConfig};
///
/// let nums: Vec<u64> = (0..10).collect();
/// let fibs: Vec<u64> = Pipeline::from(nums)
///     .configure(PipelineConfig::default().buff_size(10))
///     .map(|x| x*2) // *this* stage has its send buffer set to 10
///     .into_iter().collect();
/// ```
#[derive(Debug, Copy, Clone)]
pub struct PipelineConfig {
    buff_size: usize,
    batch_size: usize,
}

impl PipelineConfig {
    /// Set the size of the internal mpsc buffer.
    ///
    /// This can affect the effective parallelism and the length of the backlog between stages when
    /// different stages of the pipeline take different amounts of time
    pub fn buff_size(self, buff_size: usize) -> Self {
        Self { buff_size, ..self }
    }

    /// Set the size of each batch of messages sent
    ///
    /// This tunes how much overhead is spent on synchronisation
    pub fn batch_size(self, batch_size: usize) -> Self {
        Self { batch_size, ..self }
    }
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            buff_size: 10,
            batch_size: 10,
        }
    }
}

#[derive(Debug)]
pub struct Pipeline<Output>
    where
        Output: Send + 'static,
{
    rx: Receiver<Output>,
    config: PipelineConfig,
}

impl<Output> Pipeline<Output>
    where
        Output: Send,
{
    /// Start a Pipeline
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::io::{self, BufRead};
    /// use pipelines::Pipeline;
    /// let pl = Pipeline::new(|tx| {
    ///     let stdin = io::stdin();
    ///     for line in stdin.lock().lines() {
    ///         tx.send(line.unwrap());
    ///     }
    /// });
    /// ```
    pub fn new<F>(func: F) -> Self
        where
            F: FnOnce(Sender<Output>) -> () + Send + 'static,
    {
        let config = PipelineConfig::default();
        let (tx, rx) = Sender::pair(config);
        thread::spawn(move || func(tx));
        Pipeline { rx, config }
    }

    /// Start a pipeline from an IntoIterator
    ///
    /// Example:
    ///
    /// use std::io::{self, BufRead};
    /// use pipelines::Pipeline;
    /// let pl = Pipeline::new((0..100))
    ///     .map(|x| x*2);
    pub fn from<I>(source: I) -> Pipeline<Output>
        where
            I: IntoIterator<Item = Output> + Send + 'static,
    {
        Self::new(move |tx| {
            for item in source {
                tx.send(item);
            }
        })
    }

    /// Change the configuration of the pipeline
    ///
    /// Note that this applies to stages occurring *after* the config, not before. See
    /// `PipelineConfig`
    pub fn configure(self, config: PipelineConfig) -> Self {
        Pipeline {
            rx: self.rx,
            config,
        }
    }

    pub fn then<EntryOut, Entry>(self, next: Entry) -> Pipeline<EntryOut>
        where
            Entry: PipelineEntry<Output, EntryOut> + Send + 'static,
            EntryOut: Send,
    {
        self.pipe(move |tx, rx| next.process(tx, rx))
    }

    /// Express a `PipelineEntry` as a closure
    ///
    /// # Example
    ///
    /// Take some directories and collect their contents
    ///
    /// ```rust
    /// use pipelines::Pipeline;
    /// use std::fs;
    /// use std::path::PathBuf;
    /// let directories = vec!["/usr/bin", "/usr/local/bin"];
    ///
    /// let found_files: Vec<PathBuf> = Pipeline::from(directories)
    ///     .pipe(|out, dirs| {
    ///         for dir in dirs {
    ///             for path in fs::read_dir(dir).unwrap() {
    ///                 out.send(path.unwrap().path());
    ///             }
    ///         }
    ///     })
    ///     .into_iter().collect();
    /// ```
    pub fn pipe<EntryOut, Func>(self, func: Func) -> Pipeline<EntryOut>
        where
            Func: FnOnce(Sender<EntryOut>, Receiver<Output>) -> () + Send + 'static,
            EntryOut: Send,
    {
        let config = self.config.clone();
        let (tx, rx) = Sender::pair(config.clone());
        thread::spawn(move || {
            func(tx, self.rx);
        });

        Pipeline { rx, config: config }
    }

    /// Similar to `pipe`, but with multiple workers that will pull from a shared queue
    ///
    /// # Example
    ///
    /// Take some directories and collect their contents
    ///
    /// ```rust
    /// use pipelines::Pipeline;
    /// use std::fs;
    /// use std::path::PathBuf;
    /// let directories = vec!["/usr/bin", "/usr/local/bin"];
    ///
    /// let found_files: Vec<PathBuf> = Pipeline::from(directories)
    ///     .ppipe(5, |out, dirs| {
    ///         for dir in dirs {
    ///             for path in fs::read_dir(dir).unwrap() {
    ///                 out.send(path.unwrap().path());
    ///             }
    ///         }
    ///     })
    ///     .into_iter().collect();
    /// ```
    pub fn ppipe<EntryOut, Func>(
        self,
        workers: usize,
        func: Func,
    ) -> Pipeline<EntryOut>
        where
            Func: Fn(Sender<EntryOut>, LockedReceiver<Output>) -> ()
            + Send
            + Sync
            + 'static,
            Output: Send,
            EntryOut: Send,
    {
        // we want a final `master_tx` which everyone will send to, and that we will return
        let (master_tx, master_rx) = Sender::pair(self.config.clone());

        // and then a shared rx that everyone will draw from
        let (chan_tx, chan_rx) = Sender::pair(self.config.clone());
        let chan_rx = LockedReceiver::new(chan_rx);

        // so we can send copies into the various threads
        let func = Arc::new(func);

        // bring up the actual workers
        for _ in 0..workers {
            let entry_rx = chan_rx.clone();
            let entry_tx = master_tx.clone();
            let func = func.clone();

            thread::spawn(move || {
                func(entry_tx, entry_rx);
            });
        }

        // otherwise `self` moved into the closure
        let config = self.config;
        let rx = self.rx;

        // now since we're going to return immediately, we need to spawn another thread which will
        // feed our thread-pool
        thread::spawn(move || {
            // now we copy the work from rx into the shared channel. the
            // workers will be putting their results into tx directly so
            // this is the only shuffling around that we have to do
            for item in rx {
                chan_tx.send(item);
            }
        });

        Pipeline {
            rx: master_rx,
            config: config,
        }
    }

    /// Call `func` on every entry in the pipeline
    ///
    /// # Example
    ///
    /// Double every number
    ///
    /// ```rust
    /// use pipelines::Pipeline;
    /// let nums: Vec<u64> = (0..10).collect();
    ///
    /// let doubled: Vec<u64> = Pipeline::from(nums)
    ///     .map(|x| x*2)
    ///     .into_iter().collect();
    /// ```
    pub fn map<EntryOut, Func>(self, func: Func) -> Pipeline<EntryOut>
        where
            Func: Fn(Output) -> EntryOut + Send + 'static,
            EntryOut: Send,
    {
        self.pipe(move |tx, rx| {
            for entry in rx {
                tx.send(func(entry));
            }
        })
    }

    /// Call `func` on every entry in the pipeline using multiple worker threads
    ///
    /// # Example
    ///
    /// Double every number
    ///
    /// ```rust
    /// use pipelines::Pipeline;
    /// let nums: Vec<u64> = (0..10).collect();
    ///
    /// let doubled: Vec<u64> = Pipeline::from(nums)
    ///     .pmap(2, |x| x*2)
    ///     .into_iter().collect();
    /// ```
    pub fn pmap<EntryOut, Func>(
        self,
        workers: usize,
        func: Func,
    ) -> Pipeline<EntryOut>
        where
            Func: Fn(Output) -> EntryOut + Send + Sync + 'static,
            EntryOut: Send,
    {
        if workers == 1 {
            return self.map(func);
        }
        self.ppipe(workers, move |tx, rx| {
            for item in rx {
                tx.send(func(item))
            }
        })
    }

    /// Pass items into the next stage only if `pred` is true
    ///
    /// # Example
    ///
    /// Pass on only even numbers
    ///
    /// ```rust
    /// use pipelines::Pipeline;
    /// let nums: Vec<u64> = (0..10).collect();
    ///
    /// let evens: Vec<u64> = Pipeline::from(nums)
    ///     .filter(|x| x%2 == 0)
    ///     .into_iter().collect();
    /// ```
    pub fn filter<Func>(self, pred: Func) -> Pipeline<Output>
        where
            Func: Fn(&Output) -> bool + Send + 'static,
    {
        self.pipe(move |tx, rx| {
            for entry in rx {
                if pred(&entry) {
                    tx.send(entry);
                }
            }
        })
    }

    /// Consume this Pipeline without collecting the results
    ///
    /// Can be useful if the work was done in the final stage
    ///
    /// # Example
    ///
    /// ```rust
    /// use pipelines::Pipeline;
    /// let nums: Vec<u64> = (0..10).collect();
    ///
    /// Pipeline::from(nums)
    ///     .map(|fname| /* something with side-effects */ ())
    ///     .drain(); // no results to pass on
    /// ```
    pub fn drain(self) {
        for _ in self {}
    }
}

// We can implement reduce/preduce only if entries are (key, value) tuples with a hashable key
impl<OutKey, OutValue> Pipeline<(OutKey, OutValue)>
    where
        OutKey: Hash + Eq + Send,
        OutValue: Send,
{
    /// The reduce phase of a mapreduce-type pipeline.
    ///
    /// The previous entry must have sent tuples of (Key, Value), and this entry
    /// groups them by Key and calls `func` once per Key
    ///
    /// # Example
    ///
    ///
    /// ```rust
    /// use pipelines::Pipeline;
    /// let nums: Vec<u64> = (0..10).collect();
    ///
    /// // find the sum of the even/odd numbers in the doubles of 0..10
    /// let biggests: Vec<(bool, u64)> = Pipeline::from(nums)
    ///     .map(|x| (x % 2 == 0, x*2))
    ///     .reduce(|evenness, nums| (evenness, *nums.iter().max().unwrap()))
    ///     .into_iter().collect();
    /// ```
    pub fn reduce<EntryOut, Func>(self, func: Func) -> Pipeline<EntryOut>
        where
            Func: Fn(OutKey, Vec<OutValue>) -> EntryOut + Send + 'static,
            EntryOut: Send,
    {
        self.pipe(move |tx, rx| {
            // gather up all of the values and group them by key
            let mut by_key: HashMap<OutKey, Vec<OutValue>> = HashMap::new();
            for (key, value) in rx {
                by_key.entry(key).or_insert_with(Vec::new).push(value)
            }

            // now that we have them all grouped by key, we can run the reducer on the groups
            for (key, values) in by_key.into_iter() {
                let output = func(key, values);
                tx.send(output);
            }
        })
    }

    /// Bring up `workers` threads and send values with the same keys to the same thread
    ///
    /// They arrive unordered. This is part of the work of `preduce`
    pub fn distribute<EntryOut, Func>(
        self,
        workers: usize,
        func: Func,
    ) -> Pipeline<EntryOut>
        where
            Func: Fn(Sender<EntryOut>, Receiver<(OutKey, OutValue)>)
            + Send
            + Sync
            + 'static,
            EntryOut: Send,
    {
        let func = Arc::new(func);
        let pl_config = self.config.clone();

        self.pipe(move |tx, rx| {
            // build up the reducer threads
            let mut txs = Vec::with_capacity(workers);
            for _ in 0..workers {
                let func = func.clone();
                // each thread receives data on an rx that we make for it
                let (entry_tx, entry_rx) = Sender::pair(pl_config);
                // but they send their data directly into the next stage
                let tx = tx.clone();

                thread::spawn(move || func(tx, entry_rx));

                txs.push(entry_tx);
            }

            // now iterate through the messages sent into the master reducer thread (us)
            for (key, value) in rx {
                let mut hasher = DefaultHasher::new();
                key.hash(&mut hasher);
                let which = (hasher.finish() as usize) % workers;

                // because we send synchronously like this, we may block if this thread's buffer
                // doesn't have room for this message which may happen if a reducer can't keep up,
                // even if another reducer may have buffer space. (We can't send it to any other
                // thread because a reducer thread must see all instances of a given key). In the
                // case of `preduce`, during this phase the reducers haven't actually started doing
                // any work yet so so any blocking they do will probably be just due to
                // hashmap/vector reallocation.  For other use-cases they may want to use larger
                // buffer sizes or other workarounds for uneven work distribution
                txs[which].send((key, value));
            }
        })
    }

    /// Like `reduce` but called with multiple reducer threads
    ///
    /// Each instance of `func` is called with a Key and every Value that had that Key
    ///
    /// # Example
    ///
    /// Double every number
    ///
    /// ```rust
    /// use pipelines::Pipeline;
    /// let nums: Vec<u64> = (0..10).collect();
    ///
    /// let biggests: Vec<(bool, u64)> = Pipeline::from(nums)
    ///     .map(|x| (x % 2 == 0, x*2))
    ///     .preduce(2, |evenness, nums| (evenness, *nums.iter().max().unwrap()))
    ///     .into_iter().collect();
    /// ```
    pub fn preduce<EntryOut, Func>(
        self,
        workers: usize,
        func: Func,
    ) -> Pipeline<EntryOut>
        where
            Func: Fn(OutKey, Vec<OutValue>) -> EntryOut + Send + Sync + 'static,
            OutKey: Send,
            OutValue: Send,
            EntryOut: Send,
    {
        if workers == 1 {
            return self.reduce(func);
        }
        self.distribute(workers, move |tx, rx| {
            let mut hm = HashMap::new();
            for (k, v) in rx {
                hm.entry(k).or_insert_with(Vec::new).push(v);
            }

            for (k, vs) in hm.into_iter() {
                tx.send(func(k, vs));
            }
        })
    }
}

impl<Output> IntoIterator for Pipeline<Output>
    where
        Output: Send,
{
    type Item = Output;
    type IntoIter = ReceiverIntoIterator<Output>;

    fn into_iter(self) -> ReceiverIntoIterator<Output> {
        self.rx.into_iter()
    }
}

/// A trait for structs that may be used as `Pipeline` entries
pub trait PipelineEntry<In, Out> {
    fn process<I: IntoIterator<Item = In>>(self, tx: Sender<Out>, rx: I) -> ();
}

mod map {
    use std::marker::PhantomData;

    use super::{PipelineEntry, Sender};

    /// A pipeline entry representing a function to be run on each value and its
    /// result to be sent down the pipeline
    #[derive(Debug)]
    pub struct Mapper<In, Out, Func>
        where
            Func: Fn(In) -> Out,
    {
        func: Func,

        // make the compiler happy
        in_: PhantomData<In>,
        out_: PhantomData<Out>,
    }

    /// Make a new `Mapper` out of a function
    impl<In, Out, Func> Mapper<In, Out, Func>
        where
            Func: Fn(In) -> Out,
    {
        pub fn new(func: Func) -> Self {
            Mapper {
                func,
                in_: PhantomData,
                out_: PhantomData,
            }
        }
    }

    impl<In, Out, Func> PipelineEntry<In, Out> for Mapper<In, Out, Func>
        where
            Func: Fn(In) -> Out,
    {
        fn process<I: IntoIterator<Item = In>>(self, tx: Sender<Out>, rx: I) {
            for item in rx {
                let mapped = (self.func)(item);
                tx.send(mapped);
            }
        }
    }

    impl<In, Out, Func> Clone for Mapper<In, Out, Func>
        where
            Func: Fn(In) -> Out + Copy,
    {
        fn clone(&self) -> Self {
            Mapper::new(self.func)
        }
    }

    impl<In, Out, Func> Copy for Mapper<In, Out, Func>
        where
            Func: Fn(In) -> Out + Copy,
    {
    }
}

mod filter {
    use std::marker::PhantomData;

    use super::{PipelineEntry, Sender};

    /// A pipeline entry with a predicate that values must beet to be sent
    /// further in the pipeline
    #[derive(Debug)]
    pub struct Filter<In, Func>
        where
            Func: Fn(&In) -> bool,
    {
        func: Func,

        // make the compiler happy
        in_: PhantomData<In>,
    }

    /// Make a new `Filter` out of a predicate function
    impl<In, Func> Filter<In, Func>
        where
            Func: Fn(&In) -> bool,
    {
        pub fn new(func: Func) -> Self {
            Filter {
                func,
                in_: PhantomData,
            }
        }
    }

    impl<In, Func> PipelineEntry<In, In> for Filter<In, Func>
        where
            Func: Fn(&In) -> bool,
    {
        fn process<I: IntoIterator<Item = In>>(self, tx: Sender<In>, rx: I) {
            for item in rx {
                if (self.func)(&item) {
                    tx.send(item);
                }
            }
        }
    }
}

mod multiplex {
    // work around https://github.com/rust-lang/rust/issues/28229
    // (functions implement Copy but not Clone). This is only necessary for the older-style
    // Multiplex
    #![cfg_attr(feature = "cargo-clippy", allow(expl_impl_clone_on_copy))]

    use std::marker::PhantomData;
    use std::thread;

    #[cfg(feature = "chan")]
    use chan;

    use super::{LockedReceiver, PipelineConfig, PipelineEntry, Sender};

    /// A meta pipeline entry that distributes the work of a `PipelineEntry`
    /// across multiple threads
    #[derive(Debug)]
    pub struct Multiplex<In, Out, Entry>
        where
            Entry: PipelineEntry<In, Out> + Send,
    {
        entries: Vec<Entry>,

        // make the compiler happy
        in_: PhantomData<In>,
        out_: PhantomData<Out>,
    }

    /// Build a `Multiplex` by copying an existing `PipelineEntry`
    ///
    /// Note: this is only applicable where the `PipelineEntry` implements Copy,
    /// which due to [Rust #28229](https://github.com/rust-lang/rust/issues/28229)
    /// is not true of closures
    impl<In, Out, Entry> Multiplex<In, Out, Entry>
        where
            Entry: PipelineEntry<In, Out> + Send + Copy,
    {
        pub fn from(entry: Entry, workers: usize) -> Self {
            Self::new((0..workers).map(|_| entry).collect())
        }
    }

    impl<In, Out, Entry> Multiplex<In, Out, Entry>
        where
            Entry: PipelineEntry<In, Out> + Send,
    {
        pub fn new(entries: Vec<Entry>) -> Self {
            Multiplex {
                entries,
                in_: PhantomData,
                out_: PhantomData,
            }
        }
    }

    impl<In, Out, Entry> PipelineEntry<In, Out> for Multiplex<In, Out, Entry>
        where
            Entry: PipelineEntry<In, Out> + Send + 'static,
            In: Send + 'static,
            Out: Send + 'static,
    {
        fn process<I: IntoIterator<Item = In>>(
            mut self,
            tx: Sender<Out>,
            rx: I,
        ) {
            if self.entries.len() == 1 {
                // if there's only one entry we can skip most of the work.
                // this way client libraries can just create multiplexers
                // without having to worry about wasting performance in the
                // simple case
                let entry = self.entries.pop().expect("len 1 but no entries?");
                return entry.process(tx, rx);
            }

            // TODO both of these methods use PipelineConfig::default() to size their internal
            // channel buffers and aren't able to customise them

            if cfg!(feature = "chan") {
                // if we're compiled when `chan` support, use that
                let (chan_tx, chan_rx) =
                    chan::sync(PipelineConfig::default().buff_size);

                for entry in self.entries {
                    let entry_rx = chan_rx.clone();
                    let entry_tx = tx.clone();

                    thread::spawn(move || {
                        entry.process(entry_tx, entry_rx);
                    });
                }

                for item in rx {
                    chan_tx.send(item);
                }
            } else {
                // if we weren't compiled with `chan` use a Mutex<rx>. workers
                // will read their work out of this channel but send their
                // results directly into the regular tx channel

                let (master_tx, chan_rx) =
                    Sender::pair(PipelineConfig::default());
                let chan_rx = LockedReceiver::new(chan_rx);

                for entry in self.entries {
                    let entry_rx = chan_rx.clone();
                    let entry_tx = tx.clone();

                    thread::spawn(move || {
                        entry.process(entry_tx, entry_rx);
                    });
                }

                // now we copy the work from rx into the shared channel. the
                // workers will be putting their results into tx directly so
                // this is the only shuffling around that we have to do
                for item in rx {
                    master_tx.send(item);
                }
            }
        }
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple() {
        let source: Vec<i32> = vec![1, 2, 3];
        let pbb: Pipeline<i32> = Pipeline::from(source);
        let produced: Vec<i32> = pbb.into_iter().collect();

        assert_eq!(produced, vec![1, 2, 3]);
    }

    #[test]
    fn map() {
        let source: Vec<i32> = (1..100).collect();
        let expect: Vec<i32> = source.iter().map(|x| x * 2).collect();

        let pbb: Pipeline<i32> = Pipeline::from(source).map(|i| i * 2);
        let produced: Vec<i32> = pbb.into_iter().collect();

        assert_eq!(produced, expect);
    }

    #[test]
    fn multiple_map() {
        let source: Vec<i32> = vec![1, 2, 3];
        let expect: Vec<i32> =
            source.iter().map(|x| (x * 2) * (x * 2)).collect();

        let pbb: Pipeline<i32> =
            Pipeline::from(source).map(|i| i * 2).map(|i| i * i);
        let produced: Vec<i32> = pbb.into_iter().collect();

        assert_eq!(produced, expect);
    }

    // just something expensive
    fn fib_work(n: u64) -> u64 {
        const WORK_FACTOR: u64 = 10;
        fib(WORK_FACTOR) + n
    }

    fn fib(n: u64) -> u64 {
        if n == 0 || n == 1 {
            1
        } else {
            fib(n - 1) + fib(n - 2)
        }
    }

    #[test]
    fn multiplex_map_function() {
        // we have two signatures for Multiplex, one that takes a function
        // pointer and one that can take a closure. This is the function pointer
        // side

        let workers: usize = 10;

        let source: Vec<u64> = (1..1000).collect();
        let expect: Vec<u64> =
            source.clone().into_iter().map(fib_work).collect();

        let pbb: Pipeline<u64> = Pipeline::from(source).then(
            multiplex::Multiplex::from(map::Mapper::new(fib_work), workers),
        );
        let mut produced: Vec<u64> = pbb.into_iter().collect();

        produced.sort(); // these may arrive out of order
        assert_eq!(produced, expect);
    }

    #[test]
    fn multiplex_map_closure() {
        let workers: usize = 10;

        let source: Vec<i32> = (1..1000).collect();
        let expect: Vec<i32> = source.iter().map(|x| x * 2).collect();

        let pbb: Pipeline<i32> =
            Pipeline::from(source).then(multiplex::Multiplex::new(
                (0..workers).map(|_| map::Mapper::new(|i| i * 2)).collect(),
            ));
        let mut produced: Vec<i32> = pbb.into_iter().collect();

        produced.sort(); // these may arrive out of order
        assert_eq!(produced, expect);
    }

    #[test]
    fn filter() {
        let source: Vec<i32> = (1..100).collect();
        let expect: Vec<i32> = source
            .iter()
            .map(|x| x + 1)
            .filter(|x| x % 2 == 0)
            .collect();

        let pbb: Pipeline<i32> =
            Pipeline::from(source).map(|i| i + 1).filter(|i| i % 2 == 0);
        let produced: Vec<i32> = pbb.into_iter().collect();

        assert_eq!(produced, expect);
    }

    #[test]
    fn simple_closure() {
        let source: Vec<i32> = (1..100).collect();
        let expect: Vec<i32> = source
            .iter()
            .map(|x| x + 1)
            .filter(|x| x % 2 == 0)
            .collect();

        let pbb: Pipeline<i32> = Pipeline::from(source).pipe(|tx, rx| {
            for item in rx {
                let item = item + 1;
                if item % 2 == 0 {
                    tx.send(item);
                }
            }
        });
        let produced: Vec<i32> = pbb.into_iter().collect();

        assert_eq!(produced, expect);
    }

    #[test]
    fn pmap() {
        let source: Vec<i32> = (1..100).collect();
        let expect: Vec<i32> = source.iter().map(|x| x * 2).collect();
        let workers: usize = 2;

        let pbb: Pipeline<i32> =
            Pipeline::from(source).pmap(workers, |i| i * 2);
        let mut produced: Vec<i32> = pbb.into_iter().collect();
        produced.sort();

        assert_eq!(produced, expect);
    }

    #[test]
    fn preduce() {
        let source: Vec<i32> = (1..1000).collect();
        let workers: usize = 2;

        let expect = vec![(false, 1996), (true, 1998)];

        let mut produced: Vec<(bool, i32)> = Pipeline::from(source)
            .map(|x| (x % 3 == 0, x * 2))
            .preduce(workers, |threevenness, nums| {
                (threevenness, *nums.iter().max().unwrap())
            })
            .into_iter()
            .collect();
        produced.sort();

        assert_eq!(produced, expect);
    }

    #[test]
    fn mapreduce() {
        let source: Vec<i32> = (1..1000).collect();
        let workers: usize = 1;

        let expect = vec![(false, 1996), (true, 1998)];

        let mut produced: Vec<(bool, i32)> = Pipeline::from(source)
            .pmap(workers, |x| x * 2)
            .pmap(workers, |x| (x % 3 == 0, x))
            .preduce(workers, |threevenness, nums| {
                (threevenness, *nums.iter().max().unwrap())
            })
            .into_iter()
            .collect();
        produced.sort();

        assert_eq!(produced, expect);
    }

    #[test]
    fn config() {
        let source: Vec<i32> = (1..100).collect();
        let _: Vec<i32> = Pipeline::from(source)
            .configure(PipelineConfig::default().buff_size(10))
            .map(|x| x * 2)
            .configure(PipelineConfig::default().buff_size(10))
            .filter(|x| x % 3 == 0)
            .into_iter()
            .collect();
    }
}