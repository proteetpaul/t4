//! Disk I/O for [`IoWorker`](crate::io_worker::IoWorker) and work-stealing
//! [`NonBlockingUring`](crate::uring_runtime::NonBlockingUring).
//!
//! # Design: enum dispatch + associated-type futures
//!
//! Store and WAL hold [`IoBackendRef`] (`Arc<IoDispatcher>`). [`IoDispatcher`] is a closed enum
//! over the built-in backends; [`IoReadFut`] / [`IoWriteFut`] / [`IoFsyncFut`] wrap each backend’s
//! concrete future and implement [`Future`] with a single `match` (static dispatch, no `dyn Future`).
//!
//! [`IoBackend`] uses associated types for each operation’s future so [`read_at`](IoBackend::read_at),
//! [`write`](IoBackend::write), and [`fsync`](IoBackend::fsync) return stack-sized futures with no
//! extra `Box` solely to erase the type.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use crate::buffer::ReadBuf;
use crate::error::{Error, Result};
use crate::io_task::{FileFsyncTask, FileReadTask, FileWriteTask, PageWrite};
use crate::io_worker::IoWorker;
#[cfg(not(feature = "shuttle"))]
use crate::uring_runtime::{NonBlockingUring, WsFsyncFut, WsReadFut, WsWriteFut};

/// Shared handle (clone is cheap: clones the `Arc`).
pub(crate) type IoBackendRef = Arc<IoDispatcher>;

/// Built-in I/O backends used by the store.
pub(crate) enum IoDispatcher {
    Dedicated(IoWorker),
    #[cfg(not(feature = "shuttle"))]
    WorkStealing(NonBlockingUring),
}

/// Read future: dispatches to the active backend’s concrete future.
pub(crate) enum IoReadFut {
    Dedicated(FileReadTask),
    #[cfg(not(feature = "shuttle"))]
    WorkStealing(WsReadFut),
}

/// Write future: dispatches to the active backend’s concrete future.
pub(crate) enum IoWriteFut {
    Dedicated(FileWriteTask),
    #[cfg(not(feature = "shuttle"))]
    WorkStealing(WsWriteFut),
}

/// Fsync future: dispatches to the active backend’s concrete future.
pub(crate) enum IoFsyncFut {
    Dedicated(FileFsyncTask),
    #[cfg(not(feature = "shuttle"))]
    WorkStealing(WsFsyncFut),
}

impl Future for IoReadFut {
    type Output = Result<(ReadBuf, usize)>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match self.get_mut() {
            IoReadFut::Dedicated(f) => Pin::new(f).poll(cx),
            #[cfg(not(feature = "shuttle"))]
            IoReadFut::WorkStealing(f) => Pin::new(f).poll(cx),
        }
    }
}

impl Future for IoWriteFut {
    type Output = Result<()>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match self.get_mut() {
            IoWriteFut::Dedicated(f) => Pin::new(f).poll(cx),
            #[cfg(not(feature = "shuttle"))]
            IoWriteFut::WorkStealing(f) => Pin::new(f).poll(cx),
        }
    }
}

impl Future for IoFsyncFut {
    type Output = Result<()>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match self.get_mut() {
            IoFsyncFut::Dedicated(f) => Pin::new(f).poll(cx),
            #[cfg(not(feature = "shuttle"))]
            IoFsyncFut::WorkStealing(f) => Pin::new(f).poll(cx),
        }
    }
}

/// Async store I/O: reads, writes, and fsync against one backing file.
///
/// Implemented by [`IoWorker`], [`NonBlockingUring`], and [`IoDispatcher`].
pub(crate) trait IoBackend: Send + Sync {
    type ReadFut: Future<Output = Result<(ReadBuf, usize)>> + Send;
    type WriteFut: Future<Output = Result<()>> + Send;
    type FsyncFut: Future<Output = Result<()>> + Send;

    fn read_at(&self, buf: ReadBuf, offset: u64) -> Self::ReadFut;
    fn write(&self, writes: Vec<PageWrite>) -> Self::WriteFut;
    fn fsync(&self) -> Self::FsyncFut;
}

impl IoBackend for IoDispatcher {
    type ReadFut = IoReadFut;
    type WriteFut = IoWriteFut;
    type FsyncFut = IoFsyncFut;

    fn read_at(&self, buf: ReadBuf, offset: u64) -> IoReadFut {
        match self {
            IoDispatcher::Dedicated(w) => IoReadFut::Dedicated(w.read_at(buf, offset)),
            #[cfg(not(feature = "shuttle"))]
            IoDispatcher::WorkStealing(ws) => IoReadFut::WorkStealing(ws.read_at(buf, offset)),
        }
    }

    fn write(&self, writes: Vec<PageWrite>) -> IoWriteFut {
        match self {
            IoDispatcher::Dedicated(w) => IoWriteFut::Dedicated(w.write(writes)),
            #[cfg(not(feature = "shuttle"))]
            IoDispatcher::WorkStealing(ws) => IoWriteFut::WorkStealing(ws.write(writes)),
        }
    }

    fn fsync(&self) -> IoFsyncFut {
        match self {
            IoDispatcher::Dedicated(w) => IoFsyncFut::Dedicated(w.fsync()),
            #[cfg(not(feature = "shuttle"))]
            IoDispatcher::WorkStealing(ws) => IoFsyncFut::WorkStealing(ws.fsync()),
        }
    }
}

/// How to run io_uring for the store file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IoBackendKind {
    /// One background thread services a single ring (default).
    #[default]
    DedicatedThread,
    /// Work-stealing pool with a ring per worker (not available with `shuttle`). Callers must
    /// poll store futures only on that pool (see crate docs).
    #[cfg(not(feature = "shuttle"))]
    WorkStealing {
        /// Number of worker threads (each owns a dup of the store fd and an io_uring ring).
        io_threads: std::num::NonZeroU32,
    },
}

/// Exact-length read: issues [`IoBackend::read_at`] and checks length.
pub(crate) fn read_exact_at(io: &IoBackendRef, buf: ReadBuf, offset: u64) -> ReadExactAt {
    ReadExactAt::new(io.clone(), buf, offset)
}

/// Exact-length read built from [`read_exact_at`].
pub(crate) struct ReadExactAt {
    io: IoBackendRef,
    state: Option<ReadExactState>,
    expected: usize,
}

enum ReadExactState {
    Start {
        buf: Option<ReadBuf>,
        offset: u64,
    },
    Reading(IoReadFut),
}

impl ReadExactAt {
    fn new(io: IoBackendRef, buf: ReadBuf, offset: u64) -> Self {
        let expected = buf.len();
        Self {
            io,
            state: Some(ReadExactState::Start {
                buf: Some(buf),
                offset,
            }),
            expected,
        }
    }
}

impl Future for ReadExactAt {
    type Output = Result<ReadBuf>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.as_mut().get_mut();
        loop {
            match this.state.take() {
                Some(ReadExactState::Start { buf, offset }) => {
                    let buf = buf.expect("read_exact_at buffer");
                    let fut = this.io.read_at(buf, offset);
                    this.state = Some(ReadExactState::Reading(fut));
                }
                Some(ReadExactState::Reading(mut fut)) => match Pin::new(&mut fut).poll(cx) {
                    Poll::Ready(Ok((buf, n))) => {
                        if n != this.expected {
                            this.state = None;
                            return Poll::Ready(Err(Error::Io(std::io::Error::new(
                                std::io::ErrorKind::UnexpectedEof,
                                format!("short read: expected {}, got {n}", this.expected),
                            ))));
                        }
                        this.state = None;
                        return Poll::Ready(Ok(buf));
                    }
                    Poll::Ready(Err(e)) => {
                        this.state = None;
                        return Poll::Ready(Err(e));
                    }
                    Poll::Pending => {
                        this.state = Some(ReadExactState::Reading(fut));
                        return Poll::Pending;
                    }
                },
                None => panic!("ReadExactAt polled after completion"),
            }
        }
    }
}
