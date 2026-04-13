use std::collections::HashMap;
use std::fmt;
use std::fs::OpenOptions;
use std::num::NonZeroU32;
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;
use std::sync::Arc;

use verified::input_kv::{T4Key, T4KeyRef, T4Value, ValueRef};
use verified::{CheckedRangeU32, RangeRequestU32};

use crate::buffer::{align_down_u64, align_up_u32, align_up_u64, try_read_buf};
use crate::memory::pool::FixedBufferPool;
use crate::error::{Error, Result};
use crate::io_backend::{read_exact_at, IoBackend, IoBackendRef, IoDispatcher};
#[cfg(not(feature = "shuttle"))]
use crate::uring_runtime::NonBlockingUring;
use crate::io_worker::IoWorker;
use crate::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use crate::wal::Wal;
use crate::{PAGE_SIZE_NZ_U32, PAGE_SIZE_U64};

pub use crate::io_backend::IoBackendKind;

#[derive(Debug, Clone)]
pub struct MountOptions {
    pub queue_depth: u32,
    pub direct_io: bool,
    pub dsync: bool,
    pub io_backend: IoBackendKind,
    /// Size of the mmap fixed-buffer arena (used for reads via `ReadFixed` when registered).
    pub fixed_buffer_pool_size_mb: usize,
}

impl Default for MountOptions {
    fn default() -> Self {
        Self {
            queue_depth: 256,
            direct_io: true,
            dsync: true,
            io_backend: IoBackendKind::default(),
            fixed_buffer_pool_size_mb: 128,
        }
    }
}

pub(crate) struct T4Store {
    io: IoBackendRef,
    wal: Wal,
    index: RwLock<HashMap<T4Key, ValueRef>>,
}

impl fmt::Debug for T4Store {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("T4Store").finish_non_exhaustive()
    }
}

impl T4Store {
    pub async fn mount_with_options(path: impl AsRef<Path>, options: MountOptions) -> Result<Self> {
        FixedBufferPool::init(options.fixed_buffer_pool_size_mb);

        let mut open = OpenOptions::new();
        open.read(true).write(true).create(true);

        let mut custom_flags = 0;
        if options.direct_io {
            custom_flags = custom_flags | libc::O_DIRECT;
        }
        if options.dsync {
            custom_flags = custom_flags | libc::O_DSYNC;
        }
        open.custom_flags(custom_flags);

        let file = open.open(path)?;
        let len = file.metadata()?.len();
        let queue_depth = NonZeroU32::new(options.queue_depth)
            .ok_or(Error::InvalidArgument("queue_depth must be > 0"))?;

        #[cfg(not(feature = "shuttle"))]
        {
            if let IoBackendKind::WorkStealing { io_threads } = options.io_backend {
                let ws = NonBlockingUring::new(&file, queue_depth, io_threads)?;
                let io: IoBackendRef = Arc::new(IoDispatcher::WorkStealing(ws.clone()));
                let io_mount = io.clone();
                let (wal, index) = ws.run_to_completion(async move {
                    Self::open_wal_and_index(io_mount, len).await
                })?;
                return Ok(Self {
                    io,
                    wal,
                    index: RwLock::new(index),
                });
            }
        }

        let io: IoBackendRef = Arc::new(IoDispatcher::Dedicated(IoWorker::new(queue_depth, file)?));
        let (wal, index) = Self::open_wal_and_index(io.clone(), len).await?;
        Ok(Self {
            io,
            wal,
            index: RwLock::new(index),
        })
    }

    async fn open_wal_and_index(
        io: IoBackendRef,
        len: u64,
    ) -> Result<(Wal, HashMap<T4Key, ValueRef>)> {
        if len == 0 {
            let wal = Wal::create(io.clone()).await?;
            Ok((wal, HashMap::new()))
        } else {
            Wal::replay(io.clone(), len).await
        }
    }

    fn read_index(&self) -> Result<RwLockReadGuard<'_, HashMap<T4Key, ValueRef>>> {
        self.index.read().map_err(|_| Error::LockPoisoned)
    }

    fn write_index(&self) -> Result<RwLockWriteGuard<'_, HashMap<T4Key, ValueRef>>> {
        self.index.write().map_err(|_| Error::LockPoisoned)
    }

    pub async fn put(&self, key: T4Key, value: T4Value) -> Result<()> {
        let value_ref = self.wal.put(key.clone(), &value).await?;
        self.write_index()?.insert(key, value_ref);
        Ok(())
    }

    pub async fn get(&self, key: T4KeyRef<'_>) -> Result<Vec<u8>> {
        let value = {
            let index = self.read_index()?;
            *index.get(key.as_bytes()).ok_or(Error::NotFound)?
        };
        let Some(value_len_u32) = NonZeroU32::new(value.length) else {
            return Ok(Vec::new());
        };
        let padded_u32 = align_up_u32(value_len_u32, PAGE_SIZE_NZ_U32)
            .map_err(|_| Error::Format("value length exceeds io buffer limit".into()))?;
        let buf = try_read_buf(padded_u32)?;
        let buf = read_exact_at(&self.io, buf, value.offset).await?;
        let value_len = value_len_u32.get() as usize;
        Ok(buf.as_slice()[..value_len].to_vec())
    }

    pub async fn get_range(&self, key: T4KeyRef<'_>, range: RangeRequestU32) -> Result<Vec<u8>> {
        let value = {
            let index = self.read_index()?;
            *index.get(key.as_bytes()).ok_or(Error::NotFound)?
        };

        let range: CheckedRangeU32 = range
            .checked_against(value.length)
            .ok_or(Error::RangeOutOfBounds)?;
        if range.is_empty() {
            return Ok(Vec::new());
        }

        let abs_start = value
            .offset
            .checked_add(u64::from(range.start()))
            .ok_or(Error::RangeOutOfBounds)?;
        let abs_end = value
            .offset
            .checked_add(u64::from(range.end()))
            .ok_or(Error::RangeOutOfBounds)?;

        let aligned_start = align_down_u64(abs_start, PAGE_SIZE_U64);
        let aligned_end = align_up_u64(abs_end, PAGE_SIZE_U64).ok_or(Error::RangeOutOfBounds)?;
        let read_len_u64 = aligned_end
            .checked_sub(aligned_start)
            .ok_or(Error::RangeOutOfBounds)?;
        let read_len_u32: u32 = read_len_u64
            .try_into()
            .map_err(|_| Error::RangeOutOfBounds)?;
        let read_len_u32 = NonZeroU32::new(read_len_u32).ok_or(Error::RangeOutOfBounds)?;
        let buf = try_read_buf(read_len_u32)?;
        let buf = read_exact_at(&self.io, buf, aligned_start).await?;

        let slice_start_u64 = abs_start
            .checked_sub(aligned_start)
            .ok_or(Error::RangeOutOfBounds)?;
        let slice_start_u32: u32 = slice_start_u64
            .try_into()
            .map_err(|_| Error::RangeOutOfBounds)?;
        let slice_start = slice_start_u32 as usize;
        let slice_len = range.len() as usize;
        let slice_end = slice_start
            .checked_add(slice_len)
            .ok_or(Error::RangeOutOfBounds)?;
        Ok(buf.as_slice()[slice_start..slice_end].to_vec())
    }

    pub async fn remove(&self, key: T4Key) -> Result<bool> {
        self.wal.tombstone(key.clone()).await?;
        let existed = self.write_index()?.remove(&key).is_some();
        Ok(existed)
    }

    pub async fn sync(&self) -> Result<()> {
        self.io.fsync().await
    }

    pub fn len(&self) -> Result<usize> {
        Ok(self.read_index()?.len())
    }

    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.read_index()?.is_empty())
    }
}
