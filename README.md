Char-circle
==================================================

A circular buffer for strings and traits for in-place string transforms.

This crate provides two key types: the `CharCircle` struct and the `StringTransform` trait. The `CharCircle` is a circular buffer specialized for UTF-8 strings, and the `StringTransform` trait builds upon it to provide a character-oriented API for in-place string transformations. In short, `StringTransform` allows you to implement transformations as iterator adaptors, with copy-on-write optimizations in mind.


### Documentation
https://docs.rs/char-circle


### Usage
Char-circle is a pure Rust library. To use it, simply add it to your `Cargo.toml`:

```toml
[dependencies]
char-circle = "0.1.0"
```
