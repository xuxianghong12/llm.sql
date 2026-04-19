<!--
 * @SPDX-License-Identifier: Apache-2.0
 * @Author: xuxianghong12
 * @Date: 2026-04-19 12:05:00
 * @LastEditors: xuxianghong12
 * @LastEditTime: 2026-04-19 16:25:09
-->
# 功能特性

[首页](index.md) | [English](../en/features.html) | [路线图](roadmap.md)

## 当前已实现

| 模块 | 状态 | 说明 |
| --- | --- | --- |
| 模型路径 | 已实现 | 仓库中已经包含 Qwen2.5-0.5B-Instruct 的导出、量化、运行和 native graph 脚本。 |
| SQLite 运行时 | 已实现 | 算子执行已经接入 SQLite 扩展，并可由基于 SQL 的运行时逻辑驱动。 |
| 运行模式 | 已实现 | Python 运行脚本同时提供 `eager` 全量加载和 `layer_stream` 分层流式加载。 |
| 存储格式 | 已实现 | 模型产物以 SQLite 数据库为核心，并配套 graph、manifest、tokenizer 等元数据文件。 |
| 内存控制 | 已实现 | 代码里已经有全量加载、按需加载以及与 page cache 相关的受控执行组件。 |
| 语言接口 | 已实现 / PoC | Python 是当前主路径，C/C++ demo 可用于更底层的 token id 级验证。 |
| 可观测性 | 已实现 | 现有脚本参数和工具模块已经能做轻量级时间与内存观测。 |

## 现在可以直接做的事

- 编译 SQLite 扩展后，用导出的模型产物直接运行 Qwen2.5-0.5B 推理。
- 在 `eager` 和 `layer_stream` 之间切换，用吞吐换取更低峰值内存。
- 导出 `model.db`、`model_int8.db`、graph JSON、tokenizer 等 SQLite 原生产物。
- 在 Python 路径和当前 C/C++ demo 路径之间复用同一套导出资产。
- 在改后端逻辑之前，先用现有脚本参数和工具模块做 profiling。

## 当前边界

- 当前对外最顺的主路径仍然是 Qwen2.5-0.5B-Instruct。
- 轻量推理路径目前仍依赖 `tokenizers` 处理 BPE。
- C/C++ demo 已经能验证运行时链路，但还不是主要的终端用户接口。