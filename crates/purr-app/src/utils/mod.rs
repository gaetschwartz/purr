#![allow(dead_code)]

use bytes::Bytes;

pub trait BytesExt {
    fn chunks(&self, chunk_size: usize) -> impl Iterator<Item = Bytes> + '_;
}

impl BytesExt for Bytes {
    fn chunks(&self, chunk_size: usize) -> impl Iterator<Item = Bytes> + '_ {
        ByteChunks {
            bytes: self,
            chunk_size,
            position: 0,
        }
    }
}

struct ByteChunks<'a> {
    bytes: &'a Bytes,
    chunk_size: usize,
    position: usize,
}

impl Iterator for ByteChunks<'_> {
    type Item = Bytes;

    fn next(&mut self) -> Option<Self::Item> {
        if self.position >= self.bytes.len() {
            return None;
        }
        let end = (self.position + self.chunk_size).min(self.bytes.len());
        let chunk = self.bytes.slice(self.position..end);
        self.position = end;
        Some(chunk)
    }
}
