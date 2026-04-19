<!--
 * @SPDX-License-Identifier: Apache-2.0
 * @Author: xuxianghong12
 * @Date: 2026-04-19 15:49:56
 * @LastEditTime: 2026-04-19 16:26:23
 * @LastEditors: xuxianghong12
-->
# llm.sql English Book

## Tooling

Use mdBook 0.5.2 together with mdbook-bib 0.5.2.

```bash
cargo install mdbook --version 0.5.2
cargo install mdbook-bib --version 0.5.2
```

## Build

From the repository root:

```bash
cd book/en
mdbook build
mdbook serve --open
```

The generated site is written to `book/en/dist/`.