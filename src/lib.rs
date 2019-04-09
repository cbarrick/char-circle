//! A circular buffer for strings and traits for in-place string transforms.
//!
//! This crate provides two key types: the [`CharCircle`] struct and the
//! [`StringTransform`] trait. The `CharCircle` is a circular buffer
//! specialized for UTF-8 strings, and the `StringTransform` trait builds upon
//! it to provide a character-oriented API for in-place string transformations.
//! In short, `StringTransform` allows you to implement transformations as
//! iterator adaptors, with copy-on-write optimizations in mind.
//!
//! The `CharCircle` uses internal mutability. This enables its contents to be
//! consumed by an external iterator, [`Chars`]. As a consequence, the
//! `CharCircle` is not `Sync`. It is implemented as a `RefCell` around the
//! [`RawCircle`], which has a nearly identical API, uses external mutability,
//! is thread-safe, and does not provide a consuming iterator.
//!
//! The `StringTransform` trait is implemented by factories of iterator
//! adaptors. For simple cases, the [`SimpleTransform`] trait provides an
//! alternative that is implemented directly by the adaptor.
//!
//!
//! Example: To Uppercase
//! ---------------------------------------------------------------------------
//!
//! Transforms which don't require configuration are most easily implemented
//! with `SimpleTransform`.
//!
//! Here we implement an uppercase transform:
//!
//! ```
//! use char_circle::{SimpleTransform, Chars};
//!
//! // Step 1: Define the transform as an iterator adaptor.
//! struct ToUpper<I>(I);
//!
//! impl<I> Iterator for ToUpper<I> where I: Iterator<Item=char> {
//!     type Item = char;
//!     fn next(&mut self) -> Option<char> {
//!         self.0.next().map(|ch| ch.to_ascii_uppercase())
//!     }
//! }
//!
//! // Step 2: Define a constructor for the adaptor with `SimpleTransform`.
//! impl<'a> SimpleTransform<'a> for ToUpper<Chars<'a>> {
//!     fn transform_chars(chars: Chars<'a>) -> Self {
//!         ToUpper(chars)
//!     }
//! }
//!
//! // Step 3: Profit!
//! let s = "can you hear me in the back?";
//! let s = ToUpper::transform(s);
//! assert_eq!(&s, "CAN YOU HEAR ME IN THE BACK?");
//! ```
//!
//!
//! Example: Caesar Cipher
//! ---------------------------------------------------------------------------
//!
//! Transforms that need to be configured should define a factory which
//! implements `StringTransform`.
//!
//! Here we implement a Caesar cipher configured with its key:
//!
//! ```
//! use char_circle::{StringTransform, Chars};
//!
//! // Step 1: Define the transform as an iterator adaptor.
//! struct CaesarCipherIter<I> {
//!     inner: I,
//!     key: i32,
//! }
//!
//! impl<I> Iterator for CaesarCipherIter<I> where I: Iterator<Item=char> {
//!     type Item = char;
//!     fn next(&mut self) -> Option<char> {
//!         let plaintext = self.inner.next()?;
//!         let ciphertext = plaintext as i32 + self.key;
//!         let ciphertext = std::char::from_u32(ciphertext as u32).unwrap();
//!         Some(ciphertext)
//!     }
//! }
//!
//! // Step 2: Define a factory for the adaptor with `StringTransform`.
//! struct CaesarCipher(i32);
//!
//! impl<'a> StringTransform<'a> for CaesarCipher {
//!     type Iter = CaesarCipherIter<Chars<'a>>;
//!     fn transform_chars(&self, chars: Chars<'a>) -> Self::Iter {
//!         CaesarCipherIter { inner: chars, key: self.0 }
//!     }
//! }
//!
//! // Step 3: Profit!
//! let encoder = CaesarCipher(8);
//! let decoder = CaesarCipher(-8);
//! let plaintext = "Veni, vidi, vici";
//! let ciphertext = encoder.transform(plaintext);
//! assert_eq!(&ciphertext, "^mvq4(~qlq4(~qkq");
//! let plaintext = decoder.transform(ciphertext);
//! assert_eq!(&plaintext, "Veni, vidi, vici");
//! ```

use std::borrow::Cow;
use std::cell::RefCell;
use std::char;
use std::cmp;
use std::io::{self, Read, Write};
use std::marker::PhantomData;
use std::mem;
use std::ptr;


// RawCircle
// ---------------------------------------------------------------------------

/// A thread-safe version of [`CharCircle`].
///
/// The API of this buffer is almost identical to `CharCircle` except that it
/// uses external mutability, and it does not provide a means to consume its
/// characters from an external iterator.
#[derive(Debug, Default, Clone)]
pub struct RawCircle {
    buf: Vec<u8>,    // The backing storage.
    len: usize,      // The number of used bytes in the buffer.
    n_chars: usize,  // The number of characters in the buffer.
    read: usize,     // Index of the read head. May equal the capacity.
    write: usize,    // Index of the write head. May equal the capacity.
}

impl RawCircle {
    /// Construct a new `RawCircle` using a string as the initial buffer.
    pub fn new(s: String) -> RawCircle {
        let n_chars = s.chars().count();
        let buf = s.into_bytes();
        let len = buf.len();
        RawCircle { buf, len, n_chars, read: 0, write: 0 }
    }

    /// Construct a new, empty `RawCircle`.
    pub fn empty() -> RawCircle {
        RawCircle::default()
    }

    /// The number of UTF-8 bytes in the buffer.
    pub fn len(&self) -> usize {
        self.len
    }

    /// The number of characters in the buffer.
    pub fn n_chars(&self) -> usize {
        self.n_chars
    }

    /// The number of bytes the buffer can hold before reallocating.
    ///
    /// This refers to the length of the backing vector. That vector may have
    /// additional capacity allocated to it that is not reported by this method.
    pub fn capacity(&self) -> usize {
        self.buf.len()
    }

    /// Reallocate the buffer if it is full.
    fn grow_if_full(&mut self) {
        // SAFTEY: There are several safety concerns here.
        //
        // 1. The read and write heads may equal the capacity when this method is
        //    called (but may not otherwise be out of bounds). We move the read head
        //    to 0 if it is at the capacity, and we prove the write head is in bounds
        //    before we use it.
        //
        // 2. We move data around using `ptr::copy`. We must maintain that the affected
        //    regions are in bounds and that the destination is trash bytes.
        //
        // 3. We must properly maintain the read and write heads. When we return, the
        //    [BACK] and [FRONT] areas must be valid UTF-8, and the read head may equal
        //    the capacity. The length should never change.
        //
        // It is safe to directly set the length of the buffer to the capacity. It is
        // in-bounds and there are no initialization or drop-safety concerns with
        // plain-old bytes.
        unsafe {
            let old_cap = self.capacity();
            if self.read == old_cap { self.read = 0 };
            if self.read == 0 { self.write = self.len };

            if old_cap == 0 {
                // We have no capacity, so add some.
                // Only hit the allocator if the backing vector has no extra capacity.
                debug_assert!(self.read == 0);
                debug_assert!(self.write == 0);
                if self.buf.capacity() == 0 { self.buf.reserve(1); }
                let new_cap = self.buf.capacity();
                self.buf.set_len(new_cap);

            } else if self.len == old_cap {
                // If the backing vector has excess capacity, we just use that.
                // Otherwise we ask the allocator to double our capacity.
                if old_cap == self.buf.capacity() {
                    self.buf.reserve(old_cap);
                }
                let new_cap = self.buf.capacity();
                self.buf.set_len(new_cap);

                if self.write == self.read {
                    // The memory areas are [BACK, FRONT, TMP].
                    // The read and write heads are between [BACK] and [FRONT].
                    // Rearange: [BACK, FRONT, TMP] -> [BACK, TMP, FRONT].
                    let front_size = old_cap - self.read;
                    let new_read = new_cap - front_size;
                    let src = self.buf.get_unchecked_mut(self.read) as *mut u8;
                    let dest = self.buf.get_unchecked_mut(new_read) as *mut u8;
                    ptr::copy(src, dest, front_size);
                    self.read = new_read;
                } else {
                    debug_assert!(self.read == 0);
                    debug_assert!(self.write == self.len);
                }
            }
        }
    }

    /// Return the number of UTF-8 bytes of the next character.
    fn peek_char_len(&self) -> Option<usize> {
        // SAFETY: It is safe to use an unchecked get at the read head when the
        // length is non-zero.
        if self.len == 0 { return None };
        let byte = unsafe { self.buf.get_unchecked(self.read) };
        let reverse = byte ^ 0b11111111;
        let leading_ones = reverse.leading_zeros();
        match leading_ones {
            0 => Some(1),
            1 => unreachable!("invalid utf-8"),
            2 => Some(2),
            3 => Some(3),
            4 => Some(4),
            _ => unreachable!("invalid utf-8"),
        }
    }

    /// Read a single byte from the buffer.
    ///
    /// It is unsafe to partially read a multibyte UTF-8 character.
    ///
    /// This method DOES update `len` but DOES NOT update `n_chars`.
    /// Callers MUST update `n_chars`.
    unsafe fn read_byte(&mut self) -> Option<u8> {
        // SAFTEY: The read head may equal the capacity,
        // but may not otherwise be out of bounds.
        if self.len == 0 { return None };
        if self.read == self.capacity() { self.read = 0 };
        let byte = self.buf.get_unchecked(self.read);
        self.read += 1;
        self.len -= 1;
        Some(*byte)
    }

    /// Read the next character in the buffer.
    pub fn read_char(&mut self) -> Option<char> {
        // SAFETY: We MUST read an entire character, and we MUST update `n_chars`.
        unsafe {
            let byte = self.read_byte()?;
            let reverse = byte ^ 0b11111111;
            let leading_ones = reverse.leading_zeros();
            let mask = 0b11111111 >> leading_ones;
            let mut ch = (byte & mask) as u32;
            let n_additional_bytes = leading_ones.saturating_sub(1);
            for _ in 0..n_additional_bytes {
                let byte = self.read_byte().unwrap();
                let byte = byte & 0b00111111;
                ch = (ch << 6) | byte as u32;
            }
            self.n_chars -= 1;
            Some(char::from_u32_unchecked(ch))
        }
    }

    /// Read bytes from this circle into a buffer.
    ///
    /// This method will only ever read complete UTF-8 characters. It returns the
    /// number of bytes read; it never returns an error.
    ///
    /// This is the implementation of [`std::io::Read`] for `RawCircle`.
    pub fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let max_len = cmp::min(buf.len(), self.len());
        let mut len = 0;
        loop {
            match self.peek_char_len() {
                None => return Ok(len),
                Some(n) => {
                    if len + n <= max_len {
                        // SAFETY: We MUST read an entire character, and we MUST update `n_chars`.
                        for i in 0..n {
                            buf[len+i] = unsafe { self.read_byte().unwrap() };
                        }
                        self.n_chars -= 1;
                        len += n;
                    } else {
                        return Ok(len);
                    }
                }
            }
        }
    }

    /// Read bytes from this circle into a buffer.
    ///
    /// This method is equivalent to [`RawCircle::read`] except the return value
    /// is the buffer cast to a `&str`.
    pub fn read_str<'a>(&mut self, buf: &'a mut [u8]) -> &'a str {
        // SAFETY: It is safe to cast to a string because `self.read` only ever reads
        // valid UTF-8.
        let n = self.read(buf).unwrap();
        let str = &buf[..n] as *const [u8] as *const str;
        return unsafe { &*str }
    }

    /// Write a single byte into the buffer.
    ///
    /// It is unsafe to write invalid UTF-8.
    ///
    /// This method DOES update `len` but DOES NOT update `n_chars`.
    /// Callers MUST update `n_chars`.
    unsafe fn write_byte(&mut self, byte: u8) {
        // SAFTEY: The write head may equal the capacity,
        // but may not otherwise be out of bounds.
        self.grow_if_full();
        if self.write == self.capacity() { self.write = 0 };
        let byte_ref = self.buf.get_unchecked_mut(self.write);
        *byte_ref = byte;
        self.write += 1;
        self.len += 1;
    }

    /// Write a character into the buffer.
    pub fn write_char(&mut self, ch: char) {
        // SAFETY: We MUST write an entire character, and we MUST update `n_chars`.
        unsafe {
            let mut utf8 = [0u8; 4];
            for byte in ch.encode_utf8(&mut utf8).as_bytes() {
                self.write_byte(*byte)
            }
            self.n_chars += 1;
        }
    }

    /// Read bytes from a string into this buffer;
    ///
    /// This method will only ever write complete UTF-8 characters. It returns the
    /// number of bytes written. This method returns an error if the input is not
    /// valid UTF-8.
    ///
    /// This is the implementation of [`std::io::Write`] for `RawCircle`.
    pub fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let mut len = 0;
        let mut tmp = [0u8; 4];
        loop {
            // SAFETY: We must make sure the read is in-bounds to use `get_unchecked`.
            if len == buf.len() { return Ok(len) };
            let first_byte = unsafe { *buf.get_unchecked(len) };
            let reverse = first_byte ^ 0b11111111;
            let leading_ones = reverse.leading_zeros();
            let n = match leading_ones {
                0 => 1,
                1 => return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid UTF-8")),
                2 => 2,
                3 => 3,
                4 => 4,
                _ => return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid UTF-8")),
            };
            tmp[0] = first_byte;
            for i in 1..n {
                // SAFETY: We must make sure the read is in-bounds to use `get_unchecked`.
                if len + i == buf.len() { return Ok(len) };
                let byte = unsafe { *buf.get_unchecked(len + i) };
                let reverse = byte ^ 0b11111111;
                let leading_ones = reverse.leading_zeros();
                if leading_ones != 1 {
                    return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid UTF-8"));
                }
                tmp[i] = byte;
            }
            // SAFETY: We MUST write an entire character, and we MUST update `n_chars`.
            for i in 0..n {
                unsafe { self.write_byte(tmp[i]) };
            }
            self.n_chars += 1;
            len += n;
        }
    }

    /// Read bytes from a string into this buffer;
    ///
    /// This method is equivalent to [`RawCircle::write`] except that it cannot return
    /// an error because the input is valid UTF-8.
    pub fn write_str(&mut self, buf: &str) -> usize {
        self.write(buf.as_bytes()).unwrap()
    }

    /// Rearange the contents so that the read head is at 0.
    fn realign(&mut self) {
        // SAFTEY: There are several safety concerns here.
        //
        // 1. The read and write heads may equal the capacity when this method is
        //    called (but may not otherwise be out of bounds). We move the read head
        //    to 0 if it is at the capacity, and we prove the write head is in bounds
        //    before we use it.
        //
        // 2. We move data around using `ptr::copy`, `ptr::copy_nonoverlapping`, and
        //    `ptr::swap_nonoverlapping`. We must maintain that all affected regions
        //    are in bounds and non-overlapping when required. We must also maintain
        //    that the destination of copys is trash bytes.
        //
        // 3. We must properly maintain the read and write heads. When we return, the
        //    read head should be at 0, and the write head should be at `self.len`.
        //    The range `read..write` must be valid UTF-8, and the write head may
        //    equal the capacity. The length should never change.
        unsafe {
            if self.read == self.capacity() { self.read = 0 };
            if self.read == 0 { self.write = self.len };

            if self.len == 0 || self.read == 0 {
                // Nothing to do.

            } else if self.read < self.write {
                // The areas in memory are [LEFT, STR, RIGHT].
                // [LEFT, STR, RIGHT] -> [STR, LEFT, RIGHT].
                let src = self.buf.get_unchecked_mut(self.read) as *mut u8;
                let dest = self.buf.get_unchecked_mut(0) as *mut u8;
                ptr::copy(src, dest, self.len);

            } else {
                // SAFTEY: Note that the write head is in bounds in this case.
                debug_assert!(self.write < self.read);

                // The areas in memory are [BACK, TMP, FRONT]. [TMP] may be empty.
                // We are trying to get to [FRONT, BACK, TMP] where [FRONT, BACK] is our string.
                let mut back = (0, self.write);
                let mut tmp = (self.write, self.read);
                let mut front = (self.read, self.capacity());
                let len = |bounds: (usize, usize)| bounds.1 - bounds.0;

                // [BACK, TMP, FRONT] -> [BACK, FRONT, TMP]
                let src = self.buf.get_unchecked_mut(front.0) as *mut u8;
                let dest = self.buf.get_unchecked_mut(tmp.0) as *mut u8;
                let count = len(front);
                ptr::copy(src, dest, count);
                front = (tmp.0, tmp.0 + count);
                tmp = (front.1, self.capacity());

                // Iterativly move parts of [FRONT] to the correct position.
                // When we're done, we still have three areas [BACK, FRONT, TMP],
                // but they do not include the bytes that are already in place.
                while len(tmp) < len(back) {
                    if len(front) < len(back) {
                        // Say [BACK] = [B0, B1] where [B0] has the same size as [FRONT].
                        // [B0, B1, FRONT, TMP] -> [FRONT, B1, B0, TMP]
                        let count = len(front);
                        let b0 = (back.0, back.0 + count);
                        let b1 = (back.0 + count, back.1);
                        let x = self.buf.get_unchecked_mut(b0.0) as *mut u8;
                        let y = self.buf.get_unchecked_mut(front.0) as *mut u8;
                        ptr::swap_nonoverlapping(x, y, count);
                        back = b1;
                    } else {
                        // Say [FRONT] = [F0, F1] where [F0] has the same size as [BACK].
                        // [BACK, F0, F1, TMP] -> [F0, BACK, F1, TMP]
                        let count = len(back);
                        let f0 = (front.0, front.0 + count);
                        let f1 = (front.0 + count, front.1);
                        let x = self.buf.get_unchecked_mut(f0.0) as *mut u8;
                        let y = self.buf.get_unchecked_mut(back.0) as *mut u8;
                        ptr::swap_nonoverlapping(x, y, count);
                        back = f0;
                        front = f1;
                    }
                }

                if len(tmp) != 0 {
                    // Say [TMP] = [T0, T1] where [T0] has the same size as [BACK].
                    // [BACK, FRONT, T0, T1] -> [T0, FRONT, BACK, T1]
                    let src = self.buf.get_unchecked_mut(back.0) as *mut u8;
                    let dest = self.buf.get_unchecked_mut(tmp.0) as *mut u8;
                    let count = len(back);
                    ptr::copy_nonoverlapping(src, dest, count);
                    let t0 = (back.0, back.0 + count);
                    back = (tmp.0, tmp.0 + count);

                    // [T0, FRONT, BACK, T1] -> [FRONT, BACK, T0, T1]
                    let src = self.buf.get_unchecked_mut(front.0) as *mut u8;
                    let dest = self.buf.get_unchecked_mut(t0.0) as *mut u8;
                    let count = len(front) + len(back);
                    ptr::copy(src, dest, count);
                } else {
                    // [BACK, FRONT, TMP] == [FRONT]
                    debug_assert!(len(back) == 0);
                }
            }
        }

        // Fix the read and write heads.
        self.read = 0;
        self.write = self.len;
    }

    /// Unpack this circular buffer into a byte vector.
    pub fn into_vec(mut self) -> Vec<u8> {
        // SAFTEY: It is safe to directly set the length of the buffer to the length
        // of the content. It is in-bounds and there are no drop-safety concerns with
        // plain-old bytes. When the buffer has been realigned, the content will be
        // on the range `0..len`.
        self.realign();
        let mut vec = self.buf;
        unsafe { vec.set_len(self.len) };
        vec
    }

    /// Unpack this circular buffer into a string.
    pub fn into_string(self) -> String {
        // SAFTEY: The public API only allows valid UTF-8 to be read or written from
        // the buffer. Thus it is safe to cast to a string.
        let vec = self.into_vec();
        unsafe { String::from_utf8_unchecked(vec) }
    }
}

impl From<String> for RawCircle {
    fn from(s: String) -> RawCircle {
        RawCircle::new(s)
    }
}

impl From<RawCircle> for String {
    fn from(c: RawCircle) -> String {
        c.into_string()
    }
}

impl From<RawCircle> for Vec<u8> {
    fn from(c: RawCircle) -> Vec<u8> {
        c.into_vec()
    }
}

impl Read for RawCircle {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        RawCircle::read(self, buf)
    }
}

impl Write for RawCircle {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        RawCircle::write(self, buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}


// CharCircle
// ---------------------------------------------------------------------------

/// A circular buffer of characters.
///
/// The reallocation policy of [`std::collections::VecDeque`] makes it a poor
/// choice as a circular buffer for modifying strings in place. It reallocates
/// early (!) and when it does, it places its contents in such a way that
/// requires shuffling to convert it back into a string in the common case.
/// This circular buffer provides a better alternative.
///
/// This circular buffer will not allocate early and is optimized for the case
/// when you read exactly the contents that were in the initial buffer. In that
/// case, the read head is guaranteed to be at index 0 of the underlying
/// buffer, making the conversion into a string trivial.
///
/// Additionally, this circular buffer provides a `char` oriented interface, but
/// uses UTF-8 internally, allowing it to operate directly on [`String`]s.
#[derive(Debug, Default, Clone)]
pub struct CharCircle {
    raw: RefCell<RawCircle>,
}

impl CharCircle {
    /// Construct a new `CharCircle` using a string as the initial buffer.
    pub fn new(s: String) -> CharCircle {
        CharCircle { raw: RefCell::new(RawCircle::new(s)) }
    }

    /// Construct a new, empty `CharCircle`.
    pub fn empty() -> CharCircle {
        CharCircle::default()
    }

    /// The number of UTF-8 bytes in the buffer.
    pub fn len(&self) -> usize {
        self.raw.borrow().len()
    }

    /// The number of characters in the buffer.
    pub fn n_chars(&self) -> usize {
        self.raw.borrow().n_chars()
    }

    /// The number of bytes the buffer can hold before reallocating.
    ///
    /// This refers to the length of the backing vector. That vector may have
    /// additional capacity allocated to it that is not reported by this method.
    pub fn capacity(&self) -> usize {
        self.raw.borrow().capacity()
    }

    /// Read the next character in the buffer.
    pub fn read_char(&self) -> Option<char> {
        self.raw.borrow_mut().read_char()
    }

    /// Read bytes from this circle into a buffer.
    ///
    /// This method will only ever read complete UTF-8 characters. It returns the
    /// number of bytes read; it never returns an error.
    ///
    /// This is the implementation of [`std::io::Read`] for `CharCircle`.
    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        self.raw.borrow_mut().read(buf)
    }

    /// Read bytes from this circle into a buffer.
    ///
    /// This method is equivalent to [`RawCircle::read`] except the return value
    /// is the buffer cast to a `&str`.
    pub fn read_str<'a>(&self, buf: &'a mut [u8]) -> &'a str {
        self.raw.borrow_mut().read_str(buf)
    }

    /// Write a character into the buffer.
    pub fn write_char(&self, ch: char) {
        self.raw.borrow_mut().write_char(ch)
    }

    /// Read bytes from a string into this buffer;
    ///
    /// This method will only ever write complete UTF-8 characters. It returns the
    /// number of bytes written. This method returns an error if the input is not
    /// valid UTF-8.
    ///
    /// This is the implementation of [`std::io::Write`] for `CharCircle`.
    pub fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.raw.borrow_mut().write(buf)
    }

    /// Read bytes from a string into this buffer;
    ///
    /// This method is equivalent to [`CharCircle::write`] except that it cannot return
    /// an error because the input is valid UTF-8.
    pub fn write_str(&self, buf: &str) -> usize {
        self.raw.borrow_mut().write_str(buf)
    }

    /// Unpack this circular buffer into a byte vector.
    pub fn into_vec(self) -> Vec<u8> {
        self.raw.into_inner().into_vec()
    }

    /// Unpack this circular buffer into a string.
    pub fn into_string(self) -> String {
        self.raw.into_inner().into_string()
    }

    /// Read characters from the buffer with an iterator.
    ///
    /// The returned iterator will read at most `n` characters and ensures that it
    /// has been exhausted upon drop.
    ///
    /// Calling `next` on the iterator is equivalent to calling `read_char` on
    /// this buffer.
    pub fn take_chars(&self, n: usize) -> Chars {
        Chars::new(&self, n)
    }

    /// Read the current characters from the buffer with an iterator.
    ///
    /// The returned iterator will read at most `n` characters, where `n` is the
    /// number of characters currently in the buffer, and ensures that it has been
    /// exhausted upon drop.
    ///
    /// Calling `next` on the iterator is equivalent to calling `read_char` on
    /// this buffer.
    ///
    /// This is equivalent to calling [`CharCircle::take_chars`] with the current
    /// number of characters in the buffer. In particular, interleaving calls to
    /// `read_char` and `write_char` on the buffer with calls to `next` on the
    /// iterator may cause the iterator to consume characters that were not in the
    /// buffer at the time it was created.
    pub fn take_current_chars(&self) -> Chars {
        self.take_chars(self.n_chars())
    }
}

impl From<String> for CharCircle {
    fn from(s: String) -> CharCircle {
        CharCircle::new(s)
    }
}

impl From<CharCircle> for String {
    fn from(c: CharCircle) -> String {
        c.into_string()
    }
}

impl From<CharCircle> for Vec<u8> {
    fn from(c: CharCircle) -> Vec<u8> {
        c.into_vec()
    }
}

impl Read for CharCircle {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        CharCircle::read(self, buf)
    }
}

impl Write for CharCircle {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        CharCircle::write(self, buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}


// Chars
// ---------------------------------------------------------------------------

/// An iterator that consumes characters in a [`CharCircle`].
///
/// Calling `next` on this iterator is equivalent to calling
/// [`CharCircle::read_char`] on the corresponding `CharCircle`.
///
/// This iterator ensures that it has been exhausted upon drop.
///
/// In the [`StringTransform`] API, this struct is an iterator over the
/// characters of the string being transformed.
#[derive(Debug)]
pub struct Chars<'a> {
    circle: &'a CharCircle,
    n_chars: usize,
}

impl<'a> Chars<'a> {
    fn new(circle: &'a CharCircle, n_chars: usize) -> Chars<'a> {
        Chars{ circle, n_chars }
    }
}

impl<'a> Iterator for Chars<'a> {
    type Item = char;

    fn next(&mut self) -> Option<char> {
        if self.n_chars == 0 {
            None
        } else {
            let ch = self.circle.read_char()?;
            self.n_chars -= 1;
            Some(ch)
        }
    }
}

impl<'a> Drop for Chars<'a> {
    /// Ensure the iterator is exhausted.
    fn drop(&mut self) {
        for _ in self { }
    }
}


// StringTransform
// ---------------------------------------------------------------------------

/// A factory trait for in-place string transformations.
///
/// This trait allows character-iterator adaptors to modify strings in-place.
/// It is implemented by adaptor factories with a single required method,
/// `transform_chars`, for constructing the adaptors. In return, this trait
/// provides a method, `transform`, which can apply the transformation to
/// both owned strings and string slices, without allocating when possible.
///
/// The in-place operation works by treating the underlying string as a
/// circular buffer. The transform reads characters from the front of the
/// buffer and writes characters to the back of the buffer. Once the transform
/// returns `None`, the unread characters are deleted and the circular buffer
/// is cast back into a string. Note that the transformation cannot always be
/// applied in place; if the transform ever returns more bytes than it has
/// read, an allocation is required to grow the buffer.
///
/// This trait supports copy-on-write optimizations. Implementors can override
/// the [`StringTransform::will_modify`] method to short-circuit the transform.
///
/// See [`SimpleTransform`] for a variant of this trait that is implemented
/// directly by iterator adaptors.
///
/// The lifetime `'a` refers to the lifetime of the `transform` method.
pub trait StringTransform<'a> {
    /// The type after applying this transform to a [`Chars`] iterator.
    type Iter: Iterator<Item=char>;

    /// Transform the characters of a string.
    fn transform_chars(&self, chars: Chars<'a>) -> Self::Iter;

    /// A hint to short-circuit string transformations.
    ///
    /// If true, the string might be modified by the transform. If false, the
    /// string will definitely not be modified by the transform.
    ///
    /// Implementors may override this function to facilitate copy-on-write
    /// optimizations. The default implementation always returns true, which
    /// disables copy-on-write.
    #[allow(unused_variables)]
    fn will_modify(&self, val: &str) -> bool {
        return true
    }

    /// Transform a string in-place
    ///
    /// This method can operate on both `String` and `&str`.
    ///
    /// A new string may be allocated if the input is not owned or if the
    /// transformation needs to buffer more characters than the string has capacity.
    fn transform<'b, T: Into<Cow<'b, str>>>(&self, s: T) -> Cow<'b, str> {
        // SAFTEY: The lifetime `'a` refers to the circular buffer, but the borrow
        // checker isn't smart enough to know that, because `'a` is a type parameter
        // of the trait. Transumting the iterator to a new lifetime solves the issue.
        // The safety condition is that we must not let the iterator outlive the
        // buffer. This means that `old_chars` must be dropped before `circle`.
        //
        // Note that it is not safe for the buffer and iterator to be used from
        // different threads. `CharCircle` is implemented with a `RefCell`, ensuring
        // that it is not `Send` and that `Chars` is neither `Send` nor `Sync`.
        let s = s.into();
        if self.will_modify(&s) {
            let s = s.into_owned();
            let circle = CharCircle::new(s);
            let old_chars = circle.take_current_chars();
            let old_chars: Chars<'_> = unsafe { mem::transmute(old_chars) };
            let new_chars = self.transform_chars(old_chars);
            for ch in new_chars { circle.write_char(ch) };  // `old_chars` is dropped after loop.
            let t = circle.into_string();  // `circle` is dropped here.
            Cow::Owned(t)
        } else {
            s
        }
    }
}


// SimpleTransform
// ---------------------------------------------------------------------------

/// A simple trait for in-place string transformations.
///
/// This trait allows character-iterator adaptors to modify strings in-place.
/// It is implemented by adaptors with a single required associated function,
/// `transform_chars`, for constructing the adaptor. In return, this trait
/// provides an associated function, `transform`, which can apply the
/// transformation to both owned strings and string slices, without allocating
/// when possible.
///
/// This trait is a simplified version of [`StringTransform`]. See the
/// documentation of that trait for more details.
pub trait SimpleTransform<'a>: Iterator<Item=char> + Sized {
    /// Transform the characters of a string.
    fn transform_chars(chars: Chars<'a>) -> Self;

    /// A hint to short-circuit string transformations.
    ///
    /// If true, the string might be modified by the transform. If false, the
    /// string will definitely not be modified by the transform.
    ///
    /// Implementors may override this function to facilitate copy-on-write
    /// optimizations. The default implementation always returns true, which
    /// disables copy-on-write.
    #[allow(unused_variables)]
    fn will_modify(val: &str) -> bool {
        return true
    }

    /// Transform a string in-place
    ///
    /// This function can operate on both `String` and `&str`.
    ///
    /// A new string may be allocated if the input is not owned or if the
    /// transformation needs to buffer more characters than the string has capacity.
    fn transform<'b, T: Into<Cow<'b, str>>>(s: T) -> Cow<'b, str> {
        SimpleStringTransform::<Self>::new().transform(s)
    }
}


// SimpleStringTransform
// ---------------------------------------------------------------------------

/// Used when you have a [`SimpleTransform`] but need a [`StringTransform`].
///
/// ### Example
///
/// ```
/// # use char_circle::*;
/// // `Identity` is a `SimpleTransform`.
/// struct Identity<I>(I);
///
/// impl<I: Iterator<Item=char>> Iterator for Identity<I> {
///     type Item = char;
///     fn next(&mut self) -> Option<char> {
///         self.0.next()
///     }
/// }
///
/// impl<'a> SimpleTransform<'a> for Identity<Chars<'a>> {
///     fn transform_chars(chars: Chars<'a>) -> Self {
///         Identity(chars)
///     }
/// }
///
/// // This function takes a `StringTransform`.
/// fn apply_transform<'a, T: StringTransform<'a>>(t: T, s: String) -> String {
///     t.transform(s).into_owned()
/// }
///
/// // Use `SimpleStringTransform` to bridge the gap.
/// let t = SimpleStringTransform::<Identity<Chars>>::new();
/// let s = apply_transform(t, "Hello World".to_string());
/// assert_eq!(&s, "Hello World");
/// ```
pub struct SimpleStringTransform<T>(pub PhantomData<T>);

impl<'a, T: SimpleTransform<'a>> SimpleStringTransform<T> {
    /// Create a new `SimpleStringTransform`.
    pub fn new() -> Self {
        SimpleStringTransform(PhantomData)
    }
}

impl<'a, T: SimpleTransform<'a>> StringTransform<'a> for SimpleStringTransform<T> {
    type Iter = T;

    fn transform_chars(&self, chars: Chars<'a>) -> Self::Iter {
        Self::Iter::transform_chars(chars)
    }

    fn will_modify(&self, val: &str) -> bool {
        Self::Iter::will_modify(val)
    }
}


// Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Identity transform
    // -------------------------

    /// An identity function as a string transform.
    struct Identity<I>(I);

    impl<I: Iterator<Item=char>> Iterator for Identity<I> {
        type Item = char;
        fn next(&mut self) -> Option<char> {
            self.0.next()
        }
    }

    impl<'a> SimpleTransform<'a> for Identity<Chars<'a>> {
        fn transform_chars(chars: Chars<'a>) -> Self {
            Identity(chars)
        }
    }

    // Half transform
    // -------------------------

    /// Deletes every other character.
    struct Half<I>(I);

    impl<I: Iterator<Item=char>> Iterator for Half<I> {
        type Item = char;
        fn next(&mut self) -> Option<char> {
            self.0.next()?;
            self.0.next()
        }
    }

    impl<'a> SimpleTransform<'a> for Half<Chars<'a>> {
        fn transform_chars(chars: Chars<'a>) -> Self {
            Half(chars)
        }
    }

    // Double transform
    // -------------------------

    /// Duplicates every character.
    struct Double<I> {
        inner: I,
        state: Option<char>,
    }

    impl<I: Iterator<Item=char>> Iterator for Double<I> {
        type Item = char;
        fn next(&mut self) -> Option<char> {
            match self.state {
                Some(_) => self.state.take(),
                None => {
                    let state = self.inner.next();
                    self.state = state;
                    state
                }
            }
        }
    }

    impl<'a> SimpleTransform<'a> for Double<Chars<'a>> {
        fn transform_chars(chars: Chars<'a>) -> Self {
            Double { inner: chars, state: None }
        }
    }

    // Tests
    // -------------------------

    #[test]
    fn test_transform() {
        let s: String = "Hello World".to_string();

        // An identity transform does not reallocate.
        let ptr = s.as_ptr();
        let cap = s.capacity();
        let s = Identity::transform(s).into_owned();
        assert_eq!(&s, "Hello World");
        assert_eq!(s.as_ptr(), ptr);
        assert_eq!(s.capacity(), cap);

        // A half transform does not reallocate.
        let ptr = s.as_ptr();
        let cap = s.capacity();
        let s = Half::transform(s).into_owned();
        assert_eq!(&s, "el ol");
        assert_eq!(s.as_ptr(), ptr);
        assert_eq!(s.capacity(), cap);

        // A first double transform does not reallocate because there is excess capacity.
        let ptr = s.as_ptr();
        let cap = s.capacity();
        let s = Double::transform(s).into_owned();
        assert_eq!(&s, "eell  ooll");
        assert_eq!(s.as_ptr(), ptr);
        assert_eq!(s.capacity(), cap);

        // A second double transform _does_ reallocate because it needs additional capacity.
        let cap = s.capacity();
        let s = Double::transform(s).into_owned();
        assert_eq!(&s, "eeeellll    oooollll");
        assert_ne!(s.capacity(), cap);
    }

    #[test]
    fn test_chars() {
        let s: String = "Hello World".to_string();
        let circle = CharCircle::new(s);

        // The iterator consumes characters from the buffer.
        let mut chars = circle.take_chars(6);
        assert_eq!(chars.next(), Some('H'));
        assert_eq!(chars.next(), Some('e'));
        assert_eq!(chars.next(), Some('l'));

        // The iterator consumes all `n` characters upon drop.
        std::mem::drop(chars);
        let s = circle.into_string();
        assert_eq!(s.as_str(), "World");

        // The iterator does not consume more than `n` characters.
        let s: String = "Hello World".to_string();
        let circle = CharCircle::new(s);
        let mut chars = circle.take_chars(6);
        assert_eq!(chars.next(), Some('H'));
        assert_eq!(chars.next(), Some('e'));
        assert_eq!(chars.next(), Some('l'));
        assert_eq!(chars.next(), Some('l'));
        assert_eq!(chars.next(), Some('o'));
        assert_eq!(chars.next(), Some(' '));
        assert_eq!(chars.next(), None);
    }

    #[test]
    fn test_circle() {
        // `read` and `write` work as expected.
        // The `read_str` and `write_str` versions hit that code path.
        let circle = CharCircle::empty();
        circle.write_str("Hello World!");
        assert_eq!(circle.len(), 11);
        assert_eq!(circle.n_chars(), 11);
        let mut buf = [0u8; 6];
        let buf_str = circle.read_str(&mut buf);
        assert_eq!(buf_str, "Hello ");
        assert_eq!(circle.len(), 5);
        assert_eq!(circle.n_chars(), 5);
        assert_eq!(&circle.into_string(), "World!");

        // A more complicated test that alternates between reading and writing.
        let circle = CharCircle::empty();
        circle.write_char('F');
        circle.write_char('o');
        circle.write_char('o');
        circle.write_char('B');
        circle.write_char('a');
        circle.write_char('r');
        assert_eq!(circle.read_char(), Some('F'));
        assert_eq!(circle.read_char(), Some('o'));
        assert_eq!(circle.read_char(), Some('o'));
        assert_eq!(circle.read_char(), Some('B'));
        assert_eq!(circle.read_char(), Some('a'));
        assert_eq!(circle.read_char(), Some('r'));
        assert_eq!(circle.read_char(), None);
        circle.write_char('H');
        circle.write_char('e');
        circle.write_char('l');
        circle.write_char('l');
        circle.write_char('o');
        circle.write_char(' ');
        circle.write_char('W');
        circle.write_char('o');
        circle.write_char('r');
        circle.write_char('l');
        circle.write_char('d');
        circle.write_char('!');
        assert_eq!(circle.read_char(), Some('H'));
        assert_eq!(circle.read_char(), Some('e'));
        assert_eq!(circle.read_char(), Some('l'));
        assert_eq!(circle.read_char(), Some('l'));
        assert_eq!(circle.read_char(), Some('o'));
        assert_eq!(circle.read_char(), Some(' '));
        circle.write_char('F');
        circle.write_char('o');
        circle.write_char('o');
        circle.write_char('F');
        circle.write_char('o');
        circle.write_char('o');
        assert_eq!(circle.read_char(), Some('W'));
        assert_eq!(circle.read_char(), Some('o'));
        assert_eq!(circle.read_char(), Some('r'));
        assert_eq!(circle.read_char(), Some('l'));
        assert_eq!(circle.read_char(), Some('d'));
        assert_eq!(circle.read_char(), Some('!'));
        assert_eq!(circle.read_char(), Some('F'));
        assert_eq!(circle.read_char(), Some('o'));
        assert_eq!(circle.read_char(), Some('o'));
        circle.write_char('B');
        circle.write_char('a');
        circle.write_char('r');
        assert_eq!(&circle.into_string(), "FooBar");
    }
}
