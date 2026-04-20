<!--
 * @SPDX-License-Identifier: Apache-2.0
 * @Author: xuxianghong12
 * @Date: 2026-04-19 15:49:56
 * @LastEditTime: 2026-04-20 21:41:41
 * @LastEditors: xuxianghong12
-->
# llm.sql English Book

Online documents: [llm.sql Documents](https://xuxianghong12.github.io/llm.sql/en/)

Build the document locally can refer to the following instructions.

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