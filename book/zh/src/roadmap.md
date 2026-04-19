<!--
 * @SPDX-License-Identifier: Apache-2.0
 * @Author: xuxianghong12
 * @Date: 2026-04-19 12:05:00
 * @LastEditors: xuxianghong12
 * @LastEditTime: 2026-04-19 16:25:39
-->
# 路线图

[首页](index.md) | [English](../en/roadmap.html) | [功能特性](features.md)


## 近期优先级

- 降低模型导出阶段的峰值内存，避免写入 SQLite BLOB 表时出现过大的临时内存占用。
- 去掉轻量 Python 推理路径里剩余的 `tokenizers` 依赖。
- 收紧 SQLite page cache 的上界控制，让运行时内存预算在压力下也更可预测。
- 继续优化 x86 算子性能，并清理 graph 到 runtime 的衔接路径。

## 计划中的扩展

- 增加 ARM NEON、RISC-V RVV 等 ISA 感知加速路径，并继续加强 x86 专项优化。
- 提升 INT8 的物理存储效率，并把量化支持继续推进到 INT4 等更小格式。
- 把模型覆盖范围从当前的 Qwen2.5-0.5B 路径扩展到更多架构。
- 把导出资产整理成更自包含、更加易于分发和复用的发布形态。

## 开发者工具链

- 增强 tracing 和 profiling，把瓶颈更明确地定位到层、算子和 swap 行为。
- 补强 C/C++ 路径的文本输入输出能力和更完整的运行时观测能力。
- 增加 SQL 原生 prompt helper 和更顺手的流式解码体验。
- 在 Python 之外继续扩展到 Go、Java、Rust 等语言绑定。

## 更长期的运行时工作

- 持久化 KV cache，支持容错恢复和任务预填充等流程。
- 按需加载 tokenizer 与 embedding，继续压低基础内存成本。
- 支持多模型同时运行时的资源调度与内存竞争管理。
- 在保持 SQL 接口稳定的前提下，为底层接入 NPU 或 GPU 算子组合预留空间。