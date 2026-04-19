<!--
 * @SPDX-License-Identifier: Apache-2.0
 * @Author: xuxianghong12
 * @Date: 2026-04-19 15:49:58
 * @LastEditTime: 2026-04-19 16:15:29
 * @LastEditors: xuxianghong12
-->
# llm.sql 中文文档

这个目录包含独立的中文 mdBook，用来记录面向实现的功能说明和路线图。

## 工具链

使用 mdBook 0.5.2 和 mdbook-bib 0.5.2。

```bash
cargo install mdbook --version 0.5.2
cargo install mdbook-bib --version 0.5.2
```

## 构建

在仓库根目录下执行：

```bash
cd book/zh
mdbook build
mdbook serve --open
```

生成站点输出到 `book/zh/dist/`。