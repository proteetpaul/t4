//! Pluggable disk I/O: [`IoBackend`] for [`IoWorker`](crate::io_worker::IoWorker) and
//! [`WorkStealingIo`](crate::uring_runtime::WorkStealingIo).
//!
//! # Design: trait objects and boxed futures (current approach)
//!
//! [`IoBackend`] is **object-safe**: methods return type-erased
//! `Pin<Box<dyn Future<Output = …> + Send + 'static>>` instead of associated types. Store and WAL
//! hold a single [`IoArc`] (`Arc<dyn IoBackend + Send + Sync>`), so there is no hand-written
//! dispatcher enum for backends.
//!
//! **Pros**
//! - One concrete handle type for [`crate::store::T4Store`] / [`crate::wal::Wal`].
//! - Object-safe trait; third-party backends could implement [`IoBackend`] behind the same `Arc`.
//!
//! **Cons**
//! - **Extra heap allocation** per `read_at` / `write` / `fsync` (the `Box` for the future).
//! - **Dynamic dispatch** on every `poll` of those futures (vtable), vs static dispatch for a
//!   concrete future type.
//!
//! # Alternative: associated types + enum (lower overhead)
//!
//! For **maximum performance**, prefer either:
//! - **Generic store**: `T4Store<B: IoBackend>` with an [`IoBackend`] trait that uses associated
//!   types for each return future (`FileReadTask`, `WsReadFut`, …), **or**
//! - **Manual enum dispatch**: a dedicated `IoDispatcher` enum plus per-operation enums
//!   (`IoReadFut`, …) that wrap each backend’s future and implement [`Future`] with a single `match`.
//!
//! In both cases the **pollable future** is typically stored **inline** in the caller’s async
//! state machine (no per-call `Box` solely to erase the future type), and dispatch is **static**
//! (monomorphized or enum branch) rather than `dyn Future`. Use this crate’s object-safe API when
//! simplicity and pluggability matter more than shaving those allocations.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use crate::buffer::ReadBuf;
use crate::error::{Error, Result};
use crate::io_task::PageWrite;

/// Shared handle to any [`IoBackend`] (clone is cheap: clones the `Arc`).
pub(crate) type IoBackendRef = Arc<dyn IoBackend + Send + Sync>;

/// Type-erased read future (one heap allocation per `read_at` in this design).
pub(crate) type ReadFutBox = Pin<Box<dyn Future<Output = Result<(ReadBuf, usize)>> + Send + 'static>>;

pub(crate) type WriteFutBox = Pin<Box<dyn Future<Output = Result<()>> + Send + 'static>>;

pub(crate) type FsyncFutBox = Pin<Box<dyn Future<Output = Result<()>> + Send + 'static>>;

/// Async store I/O: reads, writes, and fsync against one backing file.
///
/// Implemented by [`crate::io_worker::IoWorker`] and [`crate::uring_runtime::WorkStealingIo`].
/// Sharing uses [`IoArc`]; the trait does not require [`Clone`] on `Self`.
pub(crate) trait IoBackend: Send + Sync {
    fn read_at(&self, buf: ReadBuf, offset: u64) -> ReadFutBox;
    fn write(&self, writes: Vec<PageWrite>) -> WriteFutBox;
    fn fsync(&self) -> FsyncFutBox;
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
    Reading(ReadFutBox),
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
                Some(ReadExactState::Reading(mut fut)) => match fut.as_mut().poll(cx) {
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
