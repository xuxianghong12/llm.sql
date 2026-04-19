<!--
 * @SPDX-License-Identifier: Apache-2.0
 * @Author: xuxianghong12
 * @Date: 2026-04-19 11:43:49
 * @LastEditTime: 2026-04-19 11:43:49
 * @LastEditors: xuxianghong12
-->
# llm.sql Books

This directory contains two independent mdBooks used for implementation-facing product documentation.

## Tooling

Use mdBook 0.5.2 together with mdbook-bib 0.5.2.

- LaTeX formulas are enabled through `output.html.mathjax-support = true`.
- Bibliography rendering is enabled through `mdbook-bib` and each book's own `src/refs.bib`.
- English and Chinese are maintained as separate books under `book/en` and `book/zh`.

Recommended installation:

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

cd ../zh
mdbook build
mdbook serve --open
```

Each generated site is written to its own `dist/` directory.

## Deployment

CI builds both books and publishes them to GitHub Pages under:

- `/llm.sql/en/`
- `/llm.sql/zh/`

## Content Layout

- `en/`: standalone English book.
- `zh/`: standalone Chinese book.
