//! Work-stealing io_uring runtime: one ring per worker thread, async tasks polled on that pool.
//!
//! **Contract:** [`WorkStealingIo`] must only be used from futures that run on this pool (TLS io_uring).

use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::fmt;
use std::future::Future;
use std::io;
use std::num::NonZeroU32;
use std::os::fd::AsRawFd;
use std::pin::Pin;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use async_task::Runnable;
use crossbeam_channel::{Receiver, Sender, TryRecvError};
use io_uring::{EnterFlags, IoUring, cqueue, opcode, squeue, types};

use crate::buffer::AlignedBuf;
use crate::error::{Error, Result};
use crate::io_backend::{FsyncFutBox, IoBackend, ReadFutBox, WriteFutBox};
use crate::io_task::PageWrite;
use crate::sync::cooperative_yield;

type ExecutorTask = Pin<Box<dyn Future<Output = ()> + Send>>;

const URING_NUM_ENTRIES: u32 = 256;
const MAX_CONCURRENT_IO: u32 = 128;
const URING_BATCH_SIZE: u32 = 8;
const URING_SYSCALL_INTERVAL_US: u64 = 5;
const MAX_ACTIVE_TASKS_PER_THREAD: u32 = 5;

thread_local! {
    static WS_WORKER_FD: Cell<Option<i32>> = const { Cell::new(None) };
}

#[inline]
fn ws_worker_fd() -> Option<i32> {
    WS_WORKER_FD.with(|c| c.get())
}

struct WorkerFd(i32);

impl Drop for WorkerFd {
    fn drop(&mut self) {
        unsafe {
            libc::close(self.0);
        }
    }
}

struct NonBlockingInner {
    sender: Option<Sender<ExecutorTask>>,
    handles: Mutex<Option<Vec<JoinHandle<()>>>>,
}

impl Drop for NonBlockingInner {
    fn drop(&mut self) {
        self.sender.take();
        if let Ok(mut guard) = self.handles.lock() {
            if let Some(handles) = guard.take() {
                for h in handles {
                    let _ = h.join();
                }
            }
        }
    }
}

/// Cloneable handle to the work-stealing pool + per-thread rings.
#[derive(Clone)]
pub struct NonBlockingUring {
    inner: Arc<NonBlockingInner>,
}

impl fmt::Debug for NonBlockingUring {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WorkStealingIo").finish_non_exhaustive()
    }
}

impl NonBlockingUring {
    /// Spawn `num_threads` workers; each dups `file`'s fd. `queue_depth` is reserved for future tuning.
    pub fn new(file: &std::fs::File, _queue_depth: NonZeroU32, num_threads: NonZeroU32) -> Result<Self> {
        let raw_fd = file.as_raw_fd();
        let n = num_threads.get() as usize;
        let (sender, receiver) = crossbeam_channel::unbounded::<ExecutorTask>();

        let mut handles = Vec::with_capacity(n);
        for i in 0..n {
            let rx = receiver.clone();
            let h = thread::Builder::new()
                .name(format!("t4-ws-io-{i}"))
                .spawn(move || worker_main_loop(rx, raw_fd))
                .map_err(|e| Error::Io(io::Error::other(e.to_string())))?;
            handles.push(h);
        }

        Ok(Self {
            inner: Arc::new(NonBlockingInner {
                sender: Some(sender),
                handles: Mutex::new(Some(handles)),
            }),
        })
    }

    /// Run a future on the pool to completion (blocking). Use to mount the store from a non-worker thread.
    pub fn run_to_completion<F: Future + Send + 'static>(&self, future: F) -> F::Output
    where
        F::Output: Send + 'static,
    {
        let (tx, rx) = std::sync::mpsc::sync_channel::<F::Output>(1);
        let wrapped: ExecutorTask = Box::pin(async move {
            let out = future.await;
            let _ = tx.send(out);
        });
        self.inner
            .sender
            .as_ref()
            .expect("work-stealing runtime shut down")
            .send(wrapped)
            .expect("work-stealing io workers stopped");
        rx.recv().expect("work-stealing io result channel closed")
    }
}

impl IoBackend for NonBlockingUring {
    fn read_at(&self, buf: AlignedBuf, offset: u64) -> ReadFutBox {
        Box::pin(WsReadFut {
            task: Arc::new(Mutex::new(WsReadTask {
                buf: Some(buf),
                offset,
                error: None,
                result_len: None,
            })),
            uring: None,
        })
    }

    fn write(&self, writes: Vec<PageWrite>) -> WriteFutBox {
        Box::pin(WsWriteFut {
            task: Arc::new(Mutex::new(WsWriteTask {
                pages: writes,
                error: None,
            })),
            uring: None,
        })
    }

    fn fsync(&self) -> FsyncFutBox {
        Box::pin(WsFsyncFut {
            task: Arc::new(Mutex::new(WsFsyncTask { error: None })),
            uring: None,
        })
    }
}

pub(crate) trait IoUringTask: Send {
    fn prepare_sqe(&mut self) -> Vec<squeue::Entry>;
    fn complete(&mut self, cqes: Vec<&cqueue::Entry>);
}

struct WsReadTask {
    buf: Option<AlignedBuf>,
    offset: u64,
    error: Option<Error>,
    result_len: Option<usize>,
}

impl IoUringTask for WsReadTask {
    fn prepare_sqe(&mut self) -> Vec<squeue::Entry> {
        let fd = types::Fd(
            ws_worker_fd().expect("WorkStealing I/O must run on a work-stealing worker thread"),
        );
        let b = self.buf.as_mut().expect("read buffer");
        let entry = opcode::Read::new(fd, b.as_mut_ptr(), b.len_u32())
            .offset(self.offset)
            .build()
            .user_data(0);
        vec![entry]
    }

    fn complete(&mut self, cqes: Vec<&cqueue::Entry>) {
        debug_assert_eq!(cqes.len(), 1);
        let r = cqes[0].result();
        if r < 0 {
            self.error = Some(Error::Io(io::Error::from_raw_os_error(-r)));
        } else {
            self.result_len = Some(r as usize);
        }
    }
}

pub struct WsWriteTask {
    pages: Vec<PageWrite>,
    error: Option<Error>,
}

impl IoUringTask for WsWriteTask {
    fn prepare_sqe(&mut self) -> Vec<squeue::Entry> {
        let fd = ws_worker_fd().expect("WorkStealing I/O must run on a work-stealing worker thread");
        let n = self.pages.len();
        self.pages
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let is_last = i + 1 == n;
                opcode::Write::new(types::Fd(fd), p.buf.as_ptr(), p.buf.len_u32())
                    .offset(p.offset)
                    .build()
                    .flags(if is_last {
                        squeue::Flags::empty()
                    } else {
                        squeue::Flags::IO_LINK
                    })
                    .user_data(0)
            })
            .collect()
    }

    fn complete(&mut self, cqes: Vec<&cqueue::Entry>) {
        for (i, cqe) in cqes.iter().enumerate() {
            let expected = self.pages[i].buf.len();
            let r = cqe.result();
            if r < 0 {
                if self.error.is_none() {
                    self.error = Some(Error::Io(io::Error::from_raw_os_error(-r)));
                }
            } else if r as usize != expected {
                if self.error.is_none() {
                    self.error = Some(Error::Io(io::Error::new(
                        io::ErrorKind::WriteZero,
                        format!("short write: expected {expected}, got {r}"),
                    )));
                }
            }
        }
    }
}

pub struct WsFsyncTask {
    error: Option<Error>,
}

impl IoUringTask for WsFsyncTask {
    fn prepare_sqe(&mut self) -> Vec<squeue::Entry> {
        let fd = ws_worker_fd().expect("WorkStealing I/O must run on a work-stealing worker thread");
        vec![opcode::Fsync::new(types::Fd(fd)).build().user_data(0)]
    }

    fn complete(&mut self, cqes: Vec<&cqueue::Entry>) {
        debug_assert_eq!(cqes.len(), 1);
        let r = cqes[0].result();
        if r < 0 {
            self.error = Some(Error::Io(io::Error::from_raw_os_error(-r)));
        }
    }
}

struct AsyncIoTask {
    inner: Arc<Mutex<dyn IoUringTask>>,
    waker: Waker,
    completed: Arc<AtomicBool>,
    pending_completions: usize,
    completions: Vec<cqueue::Entry>,
}

unsafe impl Send for AsyncIoTask {}

impl AsyncIoTask {
    fn complete(self) {
        self.inner
            .lock()
            .expect("io task mutex poisoned")
            .complete(self.completions.iter().collect());
        self.completed.store(true, Ordering::Release);
        self.waker.wake();
    }

    fn set_completions(&mut self, count: usize) {
        self.pending_completions = count;
    }

    fn reduce_completions(&mut self) {
        self.pending_completions -= 1;
    }

    fn push_completion(&mut self, cqe: cqueue::Entry) {
        self.completions.push(cqe);
    }
}

struct IoDriver {
    ring: IoUring,
    submitted_tasks: Vec<Option<AsyncIoTask>>,
    queued_entries: VecDeque<squeue::Entry>,
    last_syscall: Instant,
    tokens: VecDeque<u16>,
    queued_submissions: u64,
}

impl IoDriver {
    fn new() -> io::Result<IoDriver> {
        let ring = IoUring::<squeue::Entry, cqueue::Entry>::builder()
            .setup_single_issuer()
            .setup_defer_taskrun()
            .build(URING_NUM_ENTRIES)?;

        let mut tokens = VecDeque::with_capacity(MAX_CONCURRENT_IO as usize);
        let mut submitted_tasks = Vec::with_capacity(MAX_CONCURRENT_IO as usize);
        for i in 0..MAX_CONCURRENT_IO {
            tokens.push_back(i as u16);
            submitted_tasks.push(None);
        }

        Ok(IoDriver {
            ring,
            submitted_tasks,
            tokens,
            queued_entries: VecDeque::with_capacity(URING_NUM_ENTRIES as usize),
            last_syscall: Instant::now(),
            queued_submissions: 0,
        })
    }

    #[inline]
    fn need_syscall(&self) -> bool {
        let is_batch_full = self.queued_entries.len() >= URING_BATCH_SIZE as usize;
        is_batch_full || self.last_syscall.elapsed() > Duration::from_micros(URING_SYSCALL_INTERVAL_US)
    }

    fn poll_completions(&mut self) {
        let cq = &mut self.ring.completion();
        loop {
            cq.sync();
            match cq.next() {
                Some(cqe) => {
                    let token = cqe.user_data() as usize;
                    let pending = self.submitted_tasks[token]
                        .as_ref()
                        .expect("missing submitted task for cqe")
                        .pending_completions;
                    if pending == 1 {
                        let mut task = self.submitted_tasks[token]
                            .take()
                            .expect("missing submitted task for cqe");
                        task.push_completion(cqe);
                        task.complete();
                        self.tokens.push_back(token as u16);
                    } else {
                        let task = self.submitted_tasks[token]
                            .as_mut()
                            .expect("missing submitted task for cqe");
                        task.push_completion(cqe);
                        task.reduce_completions();
                    }
                }
                None => break,
            }
        }
    }

    fn drain_intermediate_queue(&mut self) {
        let sq = &mut self.ring.submission();
        while !sq.is_full() && !self.queued_entries.is_empty() {
            let sqe = self.queued_entries.pop_front().unwrap();
            unsafe {
                sq.push(&sqe).expect("push sqe");
            }
            sq.sync();
            self.queued_submissions += 1;
        }
    }

    fn submit_task(&mut self, mut task: AsyncIoTask) {
        let token = self.tokens.pop_front().expect("io_uring token pool exhausted");
        let sq = &mut self.ring.submission();
        let sqes = task.inner.lock().expect("io task mutex poisoned").prepare_sqe();
        let num_sqes = sqes.len();
        task.set_completions(num_sqes);
        self.submitted_tasks[token as usize] = Some(task);
        let mut sqes_submitted = 0;

        for sqe in sqes.iter() {
            let res = unsafe { sq.push(&sqe.clone().user_data(token as u64)) };
            if res.is_err() {
                break;
            }
            sqes_submitted += 1;
            self.queued_submissions += 1;
            sq.sync();
        }
        for i in sqes_submitted..sqes.len() {
            self.queued_entries
                .push_back(sqes[i].clone().user_data(token as u64));
        }
    }

    fn add_task(task: AsyncIoTask) {
        IO_REACTOR.with(|reactor| {
            reactor
                .borrow_mut()
                .as_mut()
                .expect("io reactor not initialized on this thread")
                .submit_task(task);
        });
    }
}

thread_local! {
    static EXECUTOR: RefCell<RuntimeWorker> = RefCell::new(RuntimeWorker::new());
    static IO_REACTOR: RefCell<Option<IoDriver>> = const { RefCell::new(None) };
}

fn worker_main_loop(receiver: Receiver<ExecutorTask>, parent_fd: i32) {
    let dup_fd = unsafe { libc::dup(parent_fd) };
    if dup_fd < 0 {
        panic!("dup store fd failed: {}", io::Error::last_os_error());
    }
    let _guard = WorkerFd(dup_fd);
    WS_WORKER_FD.with(|c| c.set(Some(dup_fd)));

    EXECUTOR.with(|worker| {
        worker.borrow_mut().set_context(receiver);
    });

    let driver = IoDriver::new().unwrap_or_else(|e| panic!("IoUring init failed: {e}"));
    IO_REACTOR.with(|reactor| {
        *reactor.borrow_mut() = Some(driver);
    });

    loop {
        let disconnect = EXECUTOR.with(|worker| {
            let worker = &mut *worker.borrow_mut();
            worker.try_tick();
            worker.disconnected && worker.is_idle()
        });

        IO_REACTOR.with(|reactor| {
            let mut reactor = reactor.borrow_mut();
            let reactor = reactor.as_mut().expect("reactor");
            reactor.drain_intermediate_queue();
            if reactor.need_syscall() {
                let mut flags = EnterFlags::empty();
                flags.insert(EnterFlags::GETEVENTS);
                loop {
                    let res = unsafe {
                        reactor.ring.submitter().enter::<libc::sigset_t>(
                            reactor.queued_submissions as u32,
                            0,
                            flags.bits(),
                            None,
                        )
                    };
                    match res {
                        Ok(_) => break,
                        Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                        Err(e) => panic!("io_uring enter failed: {e}"),
                    }
                }
                reactor.queued_submissions = 0;
                reactor.last_syscall = Instant::now();
            }
            reactor.poll_completions();
        });

        if disconnect {
            break;
        }
    }

    WS_WORKER_FD.with(|c| c.set(None));
    IO_REACTOR.with(|r| *r.borrow_mut() = None);
}

struct RuntimeWorker {
    task_receiver: Option<Receiver<ExecutorTask>>,
    active_tasks: Rc<Cell<u32>>,
    local: Rc<RefCell<VecDeque<Runnable>>>,
    disconnected: bool,
}

impl RuntimeWorker {
    fn new() -> Self {
        Self {
            task_receiver: None,
            active_tasks: Rc::new(Cell::new(0)),
            local: Rc::new(RefCell::new(VecDeque::new())),
            disconnected: false,
        }
    }

    fn set_context(&mut self, receiver: Receiver<ExecutorTask>) {
        self.task_receiver = Some(receiver);
    }

    fn is_idle(&self) -> bool {
        self.local.borrow().is_empty() && self.active_tasks.get() == 0
    }

    fn try_tick(&mut self) {
        let mut runnable = self.local.borrow_mut().pop_front();
        if runnable.is_none() && self.active_tasks.get() < MAX_ACTIVE_TASKS_PER_THREAD {
            match self
                .task_receiver
                .as_ref()
                .expect("receiver set in worker_main_loop")
                .try_recv()
            {
                Ok(future) => {
                    self.active_tasks.set(self.active_tasks.get().saturating_add(1));
                    let active_tasks = Rc::clone(&self.active_tasks);
                    let local_clone = Rc::clone(&self.local);
                    let wrapped = async move {
                        future.await;
                        active_tasks.set(active_tasks.get().saturating_sub(1));
                    };
                    let schedule = move |r: Runnable| {
                        local_clone.borrow_mut().push_back(r);
                    };
                    let (r, task) = unsafe { async_task::spawn_unchecked(wrapped, schedule) };
                    task.detach();
                    runnable = Some(r);
                }
                Err(TryRecvError::Empty) => {}
                Err(TryRecvError::Disconnected) => {
                    self.disconnected = true;
                }
            }
        }
        if let Some(r) = runnable {
            r.run();
        }
    }
}

enum UringState {
    Created,
    Submitted,
}

pub struct UringFuture<T: IoUringTask + 'static> {
    state: UringState,
    task: Arc<Mutex<T>>,
    completed: Arc<AtomicBool>,
}

impl<T: IoUringTask + 'static> Future for UringFuture<T> {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        loop {
            match this.state {
                UringState::Created => {
                    let async_task = AsyncIoTask {
                        inner: this.task.clone() as Arc<Mutex<dyn IoUringTask>>,
                        waker: cx.waker().clone(),
                        completed: this.completed.clone(),
                        pending_completions: 0,
                        completions: Vec::new(),
                    };
                    IoDriver::add_task(async_task);
                    this.state = UringState::Submitted;
                }
                UringState::Submitted => {
                    if this.completed.load(Ordering::Acquire) {
                        return Poll::Ready(());
                    }
                    cooperative_yield();
                    return Poll::Pending;
                }
            }
        }
    }
}

fn uring_future_from_arc<T: IoUringTask + 'static>(task: Arc<Mutex<T>>) -> UringFuture<T> {
    UringFuture {
        state: UringState::Created,
        task,
        completed: Arc::new(AtomicBool::new(false)),
    }
}

fn take_ws_read_result(t: &mut WsReadTask) -> Result<(AlignedBuf, usize)> {
    if let Some(e) = t.error.take() {
        return Err(e);
    }
    let n = t.result_len.take().ok_or_else(|| {
        Error::Io(io::Error::other("read completed without result length"))
    })?;
    let buf = t
        .buf
        .take()
        .ok_or_else(|| Error::Io(io::Error::other("read buffer missing at completion")))?;
    Ok((buf, n))
}

pub struct WsReadFut {
    task: Arc<Mutex<WsReadTask>>,
    uring: Option<Pin<Box<UringFuture<WsReadTask>>>>,
}

impl Future for WsReadFut {
    type Output = Result<(AlignedBuf, usize)>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        if this.uring.is_none() {
            this.uring = Some(Box::pin(uring_future_from_arc(this.task.clone())));
        }
        let pinned = this.uring.as_mut().expect("uring future");
        match pinned.as_mut().poll(cx) {
            Poll::Ready(()) => {
                let mut g = this.task.lock().expect("poisoned");
                Poll::Ready(take_ws_read_result(&mut *g))
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

fn take_ws_write_result(t: &mut WsWriteTask) -> Result<()> {
    if let Some(e) = t.error.take() {
        return Err(e);
    }
    Ok(())
}

pub struct WsWriteFut {
    task: Arc<Mutex<WsWriteTask>>,
    uring: Option<Pin<Box<UringFuture<WsWriteTask>>>>,
}

impl Future for WsWriteFut {
    type Output = Result<()>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        if this.uring.is_none() {
            this.uring = Some(Box::pin(uring_future_from_arc(this.task.clone())));
        }
        let pinned = this.uring.as_mut().expect("uring future");
        match pinned.as_mut().poll(cx) {
            Poll::Ready(()) => {
                let mut g = this.task.lock().expect("poisoned");
                Poll::Ready(take_ws_write_result(&mut *g))
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

fn take_ws_fsync_result(t: &mut WsFsyncTask) -> Result<()> {
    if let Some(e) = t.error.take() {
        return Err(e);
    }
    Ok(())
}

pub struct WsFsyncFut {
    task: Arc<Mutex<WsFsyncTask>>,
    uring: Option<Pin<Box<UringFuture<WsFsyncTask>>>>,
}

impl Future for WsFsyncFut {
    type Output = Result<()>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        if this.uring.is_none() {
            this.uring = Some(Box::pin(uring_future_from_arc(this.task.clone())));
        }
        let pinned = this.uring.as_mut().expect("uring future");
        match pinned.as_mut().poll(cx) {
            Poll::Ready(()) => {
                let mut g = this.task.lock().expect("poisoned");
                Poll::Ready(take_ws_fsync_result(&mut *g))
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

#[cfg(all(test, not(feature = "shuttle")))]
mod tests {
    use std::io::Write;
    use std::num::NonZeroU32;

    use crate::io_backend::IoBackend;

    use super::*;

    #[test]
    fn work_stealing_read_roundtrip() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.write_all(&[0xabu8; 4096]).unwrap();
        tmp.flush().unwrap();

        let ws = NonBlockingUring::new(
            tmp.as_file(),
            NonZeroU32::new(32).unwrap(),
            NonZeroU32::new(2).unwrap(),
        )
        .unwrap();
        let ws2 = ws.clone();
        ws.run_to_completion(async move {
            let buf = AlignedBuf::new_zeroed(NonZeroU32::new(4096).unwrap()).unwrap();
            let (buf, n) = ws2.read_at(buf, 0).await.unwrap();
            assert_eq!(n, 4096);
            assert_eq!(buf.as_slice()[0], 0xab);
        });
    }

    #[test]
    fn work_stealing_write_read_fsync() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.as_file().set_len(4096).unwrap();

        let ws = NonBlockingUring::new(
            tmp.as_file(),
            NonZeroU32::new(32).unwrap(),
            NonZeroU32::new(2).unwrap(),
        )
        .unwrap();
        let ws2 = ws.clone();
        ws.run_to_completion(async move {
            let mut page = AlignedBuf::new_zeroed(NonZeroU32::new(4096).unwrap()).unwrap();
            page.as_mut_slice()[0] = 0x77;
            ws2
                .write(vec![PageWrite {
                    buf: page,
                    offset: 0,
                }])
                .await
                .unwrap();
            ws2.fsync().await.unwrap();

            let buf = AlignedBuf::new_zeroed(NonZeroU32::new(4096).unwrap()).unwrap();
            let (buf, n) = ws2.read_at(buf, 0).await.unwrap();
            assert_eq!(n, 4096);
            assert_eq!(buf.as_slice()[0], 0x77);
        });
    }
}
