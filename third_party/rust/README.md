# Rust dependency license texts

This directory preserves license and notice files copied byte-for-byte from the
offline Cargo source cache and the vendored Fend source. `manifest.json` records
crate names, versions, original license metadata, sources, copied file paths,
SHA-256 hashes, and the Cargo.lock/Cargo.toml hashes used for collection.
`SHA256SUMS` covers every copied license and notice.

The recorded closure is for default features on `x86_64-unknown-linux-gnu`:
normal and build dependencies, including proc macros and their dependencies.
Dev-only dependencies and the application's own package are excluded. The
installed Rust standard-library copyright document is included separately,
because it is not a Cargo.lock package. Its recorded Rust compiler commit ties
that document to the local toolchain used for the application.

The collection includes 59 crates and the Rust Standard Library, with 115 copied
license/notice files. This includes Fend's actual published `LICENSE.md`, the
local Fend patch notice, the mandatory Unicode license text accompanying
`unicode-ident`, and alternate license texts when the cached package ships them.
License expressions are retained as published; this collection does not select
one alternative for dual- or multi-licensed crates.

`wide` 0.7.33 lists `Zlib OR Apache-2.0 OR MIT` in Cargo metadata, while its cached
published package includes only `LICENSE-ZLIB.md`. That complete available text
is included and the discrepancy is recorded. No alternate text or copyright
holder was guessed. All other recorded metadata has corresponding available
license text, and no unresolved item remains. The standard-library HTML document
contains its own full in-tree and dependency notices.

Recollect from the project root without network access or package downloads:

```sh
nice -n 10 python3 third_party/rust/collect.py
```

Then verify the copied bytes with `third_party/rust` as the working directory:

```sh
sha256sum --check --quiet SHA256SUMS
```

Recollect if the dependency lockfile, selected features, target, or
Rust toolchain changes. Model/native-runtime notices are maintained separately
under the other `third_party` directories.
