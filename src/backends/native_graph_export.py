from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List

from src.backends.standalone_engine import (
    StandaloneEngine,
    _DT_ALIAS,
    _DT_FUSED,
    _DT_LAZY,
    _DT_NORMAL,
    _DT_OUTPUT,
    _DT_PLACEHOLDER,
)


_FORMAT = "llmsql-native-sql-v1"


def _arg_ref(slot: int) -> dict:
    return {"kind": "ref", "slot": slot}


def _arg_text(value: str) -> dict:
    return {"kind": "text", "value": value}


def _arg_int(value: int) -> dict:
    return {"kind": "int", "value": int(value)}


def _arg_float(value: float) -> dict:
    return {"kind": "float", "value": float(value)}


def _arg_bool(value: bool) -> dict:
    return {"kind": "bool", "value": bool(value)}


def _arg_null() -> dict:
    return {"kind": "null"}


class _SqlLowerer:
    def __init__(
        self,
        name_to_slot: Dict[str, int],
        scalar_names: set[str],
        param_names: set[str],
        alias_map: Dict[str, str],
    ) -> None:
        self._name_to_slot = name_to_slot
        self._scalar_names = scalar_names
        self._param_names = param_names
        self._alias_map = alias_map
        self._bound_args: List[dict] = []
        self._builders: Dict[str, Callable[[list, dict], str]] = {
            "aten.add.Tensor": self._lower_add,
            "aten.sub.Tensor": self._lower_sub,
            "aten.mul.Tensor": self._lower_mul,
            "aten.div.Tensor": self._lower_div,
            "aten.le.Tensor": self._lower_le,
            "aten.le.Scalar": self._lower_le,
            "aten.lt.Tensor": self._lower_lt,
            "aten.lt.Scalar": self._lower_lt,
            "aten.gt.Tensor": self._lower_gt,
            "aten.gt.Scalar": self._lower_gt,
            "aten.ge.Tensor": self._lower_ge,
            "aten.ge.Scalar": self._lower_ge,
            "aten.eq.Tensor": self._lower_eq,
            "aten.eq.Scalar": self._lower_eq,
            "aten.ne.Tensor": self._lower_ne,
            "aten.ne.Scalar": self._lower_ne,
            "aten.mean.dim": self._lower_mean_dim,
            "aten.sum.dim_IntList": self._lower_sum_dim,
            "aten.sym_size.int": self._lower_sym_size,
            "aten.arange.default": self._lower_arange,
            "aten.arange.start": self._lower_arange_start,
            "aten.embedding.default": self._lower_embedding,
            "aten.linear.default": self._lower_linear,
            "aten.matmul.default": self._lower_matmul,
            "aten.unsqueeze.default": self._lower_unsqueeze,
            "aten.expand.default": self._lower_expand,
            "aten.reshape.default": self._lower_view,
            "aten.view.default": self._lower_view,
            "aten._unsafe_view.default": self._lower_view,
            "aten.transpose.int": self._lower_transpose,
            "aten.slice.Tensor": self._lower_slice,
            "aten.select.int": self._lower_select,
            "aten.cat.default": self._lower_cat,
            "aten.cos.default": self._lower_cos,
            "aten.sin.default": self._lower_sin,
            "aten.scaled_dot_product_attention.default": self._lower_sdpa,
            "aten.logical_not.default": self._lower_logical_not,
            "aten.where.self": self._lower_where,
            "aten.clamp.default": self._lower_clamp,
            "aten.full.default": self._lower_full,
            "aten.repeat.default": self._lower_repeat,
            "llm_silu_mul_nd": self._lower_silu_mul,
            "llm_rope_nd": self._lower_rope,
            "llm_rmsnorm_nd": self._lower_rmsnorm,
            "llm_repeat_kv_nd": self._lower_repeat_kv,
            "llm_view_t12_nd": self._lower_view_t12,
            "<built-in function add>": self._lower_builtin_add,
            "<built-in function mul>": self._lower_builtin_mul,
        }

    def lower_call(self, instr: dict) -> tuple[str, List[dict]]:
        self._bound_args = []
        target = instr["target"]
        builder = self._builders.get(target)
        if builder is None:
            raise ValueError(f"unsupported native target: {target!r}")
        sql = builder(instr.get("args", []), instr.get("kwargs", {}))
        return sql, list(self._bound_args)

    def lower_fused(self, instr: dict) -> tuple[str, List[dict]]:
        self._bound_args = []
        fused_type = instr.get("_fused_type")
        args = list(instr.get("_fused_args", ()))
        if fused_type == "silu_mul":
            sql = self._call_sql(
                "llm_silu_mul_nd",
                self._bind_tensor_source(args[0]),
                self._bind_tensor_source(args[1]),
            )
        elif fused_type == "rope":
            sql = self._call_sql(
                "llm_rope_nd",
                self._bind_tensor_source(args[0]),
                self._bind_tensor_source(args[1]),
                self._bind_tensor_source(args[2]),
            )
        elif fused_type == "rmsnorm":
            sql = self._call_sql(
                "llm_rmsnorm_nd",
                self._bind_tensor_source(args[0]),
                self._bind_tensor_source(args[1]),
                self._bind_scalar(args[2]),
            )
        else:
            raise ValueError(f"unsupported fused type: {fused_type!r}")
        return sql, list(self._bound_args)

    def lower_lazy(self, instr: dict) -> tuple[str, List[dict]]:
        self._bound_args = []
        sql_parts = [self._bind_tensor_source(instr["_pre_blob"])]
        sql_parts.extend(self._bind_scalar(value) for value in instr.get("_pre_static", []))
        fragment = instr["_pre_sql"]
        sql = "SELECT " + fragment.replace("?", "{}", fragment.count("?"))
        sql = sql.format(*sql_parts)
        return sql, list(self._bound_args)

    def _call_sql(self, fn_name: str, *parts: str) -> str:
        return f"SELECT {fn_name}({', '.join(parts)})"

    def _classify(self, value: Any) -> tuple[str, Any]:
        if isinstance(value, bool):
            return "bool", value
        if value is None:
            return "null", None
        if isinstance(value, int):
            return "int", value
        if isinstance(value, float):
            return "float", value
        if isinstance(value, str):
            while value in self._alias_map:
                value = self._alias_map[value]
            if value in self._name_to_slot:
                if value in self._scalar_names:
                    return "scalar_ref", self._name_to_slot[value]
                return "ref", self._name_to_slot[value]
            if value in self._param_names:
                return "param", value
            return "text", value
        raise TypeError(f"unsupported native argument type: {type(value)!r}")

    def _append_arg(self, kind: str, payload: Any) -> str:
        if kind in {"ref", "scalar_ref"}:
            self._bound_args.append(_arg_ref(payload))
        elif kind in {"param", "text"}:
            self._bound_args.append(_arg_text(str(payload)))
        elif kind == "int":
            self._bound_args.append(_arg_int(int(payload)))
        elif kind == "float":
            self._bound_args.append(_arg_float(float(payload)))
        elif kind == "bool":
            self._bound_args.append(_arg_bool(bool(payload)))
        elif kind == "null":
            self._bound_args.append(_arg_null())
        else:
            raise ValueError(f"unsupported native arg kind: {kind!r}")
        return "?"

    def _bind_scalar(self, value: Any) -> str:
        kind, payload = self._classify(value)
        if kind == "ref":
            raise TypeError(f"expected scalar-like value, got tensor ref: {value!r}")
        return self._append_arg(kind, payload)

    def _bind_text(self, value: Any) -> str:
        kind, payload = self._classify(value)
        if kind not in {"param", "text"}:
            raise TypeError(f"expected text-like value, got {kind!r}: {value!r}")
        return self._append_arg(kind, payload)

    def _bind_tensor_source(self, value: Any) -> str:
        kind, payload = self._classify(value)
        if kind == "ref":
            return self._append_arg(kind, payload)
        if kind == "param":
            return f"llm_get_param({self._append_arg(kind, payload)})"
        raise TypeError(f"expected tensor source, got {kind!r}: {value!r}")

    def _bind_tensor_or_scalar(self, value: Any) -> str:
        kind, payload = self._classify(value)
        if kind == "ref":
            return self._append_arg(kind, payload)
        if kind == "scalar_ref":
            return f"llm_full_nd({self._append_arg(kind, payload)}, 1)"
        if kind == "param":
            return f"llm_get_param({self._append_arg(kind, payload)})"
        if kind in {"int", "float", "bool"}:
            return f"llm_full_nd({self._append_arg(kind, payload)}, 1)"
        raise TypeError(f"expected tensor or scalar, got {kind!r}: {value!r}")

    def _flatten_list(self, value: Any) -> list:
        if isinstance(value, list):
            return value
        return [value]

    def _lower_add(self, args: list, kwargs: dict) -> str:
        alpha = kwargs.get("alpha", 1)
        left = self._bind_tensor_or_scalar(args[0])
        right = self._bind_tensor_or_scalar(args[1])
        if isinstance(alpha, (int, float)) and alpha == 1:
            return self._call_sql("llm_add_nd", left, right)
        return self._call_sql(
            "llm_add_nd",
            left,
            f"llm_scale_nd({right}, {self._bind_scalar(alpha)})",
        )

    def _lower_sub(self, args: list, kwargs: dict) -> str:
        return self._call_sql(
            "llm_sub_nd",
            self._bind_tensor_or_scalar(args[0]),
            self._bind_tensor_or_scalar(args[1]),
        )

    def _lower_mul(self, args: list, kwargs: dict) -> str:
        left_kind, _ = self._classify(args[0])
        right_kind, _ = self._classify(args[1])
        if left_kind in {"int", "float", "bool"}:
            return self._call_sql(
                "llm_scale_nd",
                self._bind_tensor_source(args[1]),
                self._bind_scalar(args[0]),
            )
        if right_kind in {"int", "float", "bool"}:
            return self._call_sql(
                "llm_scale_nd",
                self._bind_tensor_source(args[0]),
                self._bind_scalar(args[1]),
            )
        return self._call_sql(
            "llm_mul_nd",
            self._bind_tensor_or_scalar(args[0]),
            self._bind_tensor_or_scalar(args[1]),
        )

    def _lower_div(self, args: list, kwargs: dict) -> str:
        right_kind, _ = self._classify(args[1])
        if right_kind in {"int", "float", "bool"}:
            return self._call_sql(
                "llm_scale_nd",
                self._bind_tensor_source(args[0]),
                f"1.0 / {self._bind_scalar(args[1])}",
            )
        return self._call_sql(
            "llm_div_nd",
            self._bind_tensor_or_scalar(args[0]),
            self._bind_tensor_or_scalar(args[1]),
        )

    def _lower_cmp(self, fn_name: str, args: list) -> str:
        return self._call_sql(
            fn_name,
            self._bind_tensor_or_scalar(args[0]),
            self._bind_tensor_or_scalar(args[1]),
        )

    def _lower_le(self, args: list, kwargs: dict) -> str:
        return self._lower_cmp("llm_le_nd", args)

    def _lower_lt(self, args: list, kwargs: dict) -> str:
        return self._lower_cmp("llm_lt_nd", args)

    def _lower_gt(self, args: list, kwargs: dict) -> str:
        return self._lower_cmp("llm_gt_nd", args)

    def _lower_ge(self, args: list, kwargs: dict) -> str:
        return self._lower_cmp("llm_ge_nd", args)

    def _lower_eq(self, args: list, kwargs: dict) -> str:
        return self._lower_cmp("llm_eq_nd", args)

    def _lower_ne(self, args: list, kwargs: dict) -> str:
        return self._lower_cmp("llm_ne_nd", args)

    def _lower_mean_dim(self, args: list, kwargs: dict) -> str:
        dim = args[1][0] if isinstance(args[1], list) else args[1]
        keepdim = args[2] if len(args) > 2 else 0
        return self._call_sql(
            "llm_mean_nd",
            self._bind_tensor_source(args[0]),
            self._bind_scalar(dim),
            self._bind_scalar(keepdim),
        )

    def _lower_sum_dim(self, args: list, kwargs: dict) -> str:
        dim = args[1][0] if isinstance(args[1], list) else args[1]
        keepdim = args[2] if len(args) > 2 else 0
        return self._call_sql(
            "llm_sum_nd",
            self._bind_tensor_source(args[0]),
            self._bind_scalar(dim),
            self._bind_scalar(keepdim),
        )

    def _lower_sym_size(self, args: list, kwargs: dict) -> str:
        return self._call_sql(
            "llm_shape",
            self._bind_tensor_source(args[0]),
            self._bind_scalar(args[1]),
        )

    def _lower_arange(self, args: list, kwargs: dict) -> str:
        return self._call_sql("llm_arange_nd", self._bind_scalar(args[0]))

    def _lower_arange_start(self, args: list, kwargs: dict) -> str:
        end_1 = self._bind_scalar(args[1])
        start_1 = self._bind_scalar(args[0])
        start_2 = self._bind_scalar(args[0])
        end_2 = self._bind_scalar(args[1])
        start_3 = self._bind_scalar(args[0])
        span_1 = f"({end_1} - {start_1})"
        span_2 = f"({end_2} - {start_3})"
        return (
            "SELECT llm_add_nd("
            f"llm_arange_nd({span_1}), llm_full_nd({start_2}, {span_2}))"
        )

    def _lower_embedding(self, args: list, kwargs: dict) -> str:
        weight_kind, _ = self._classify(args[0])
        if weight_kind == "param":
            return self._call_sql(
                "llm_embedding_param_nd",
                self._bind_text(args[0]),
                self._bind_tensor_source(args[1]),
            )
        return self._call_sql(
            "llm_embedding_nd",
            self._bind_tensor_source(args[0]),
            self._bind_tensor_source(args[1]),
        )

    def _lower_linear(self, args: list, kwargs: dict) -> str:
        weight_kind, _ = self._classify(args[1])
        if weight_kind == "param":
            if len(args) >= 3 and args[2] is not None:
                bias_kind, _ = self._classify(args[2])
                if bias_kind == "param":
                    return (
                        "SELECT llm_linear_param_bias_nd("
                        f"{self._bind_tensor_source(args[0])}, "
                        f"{self._bind_text(args[1])}, "
                        f"llm_get_param({self._bind_text(args[2])}))"
                    )
                return (
                    "SELECT llm_add_nd("
                    f"llm_linear_param_nd({self._bind_tensor_source(args[0])}, {self._bind_text(args[1])}), "
                    f"{self._bind_tensor_or_scalar(args[2])})"
                )
            return (
                "SELECT llm_linear_param_nd("
                f"{self._bind_tensor_source(args[0])}, {self._bind_text(args[1])})"
            )
        if len(args) >= 3 and args[2] is not None:
            return (
                "SELECT llm_add_nd("
                f"llm_linear_nd({self._bind_tensor_source(args[0])}, {self._bind_tensor_source(args[1])}), "
                f"{self._bind_tensor_or_scalar(args[2])})"
            )
        return self._call_sql(
            "llm_linear_nd",
            self._bind_tensor_source(args[0]),
            self._bind_tensor_source(args[1]),
        )

    def _lower_matmul(self, args: list, kwargs: dict) -> str:
        return self._call_sql(
            "llm_bmm",
            self._bind_tensor_source(args[0]),
            self._bind_tensor_source(args[1]),
        )

    def _lower_unsqueeze(self, args: list, kwargs: dict) -> str:
        return self._call_sql(
            "llm_unsqueeze_nd",
            self._bind_tensor_source(args[0]),
            self._bind_scalar(args[1]),
        )

    def _lower_expand(self, args: list, kwargs: dict) -> str:
        tensor = self._bind_tensor_source(args[0])
        dims = [self._bind_scalar(item) for item in self._flatten_list(args[1])]
        return self._call_sql(
            "llm_expand_nd",
            tensor,
            *dims,
        )

    def _lower_view(self, args: list, kwargs: dict) -> str:
        tensor = self._bind_tensor_source(args[0])
        dims = [self._bind_scalar(item) for item in self._flatten_list(args[1])]
        return self._call_sql(
            "llm_view_nd",
            tensor,
            *dims,
        )

    def _lower_transpose(self, args: list, kwargs: dict) -> str:
        return self._call_sql(
            "llm_transpose_nd",
            self._bind_tensor_source(args[0]),
            self._bind_scalar(args[1]),
            self._bind_scalar(args[2]),
        )

    def _lower_slice(self, args: list, kwargs: dict) -> str:
        dim = args[1] if len(args) > 1 else 0
        start = args[2] if len(args) > 2 else 0
        end = args[3] if len(args) > 3 else 2147483647
        step = args[4] if len(args) > 4 else 1
        return self._call_sql(
            "llm_slice_nd",
            self._bind_tensor_source(args[0]),
            self._bind_scalar(dim),
            self._bind_scalar(start),
            self._bind_scalar(end),
            self._bind_scalar(step),
        )

    def _lower_select(self, args: list, kwargs: dict) -> str:
        return self._call_sql(
            "llm_select_nd",
            self._bind_tensor_source(args[0]),
            self._bind_scalar(args[1]),
            self._bind_scalar(args[2]),
        )

    def _lower_cat(self, args: list, kwargs: dict) -> str:
        dim_value = args[1] if len(args) > 1 else 0
        items = self._flatten_list(args[0])
        if len(items) == 2:
            tensors = [self._bind_tensor_source(item) for item in items]
            dim = self._bind_scalar(dim_value)
            return self._call_sql("llm_cat_nd", tensors[0], tensors[1], dim)
        dim = self._bind_scalar(dim_value)
        tensors = [self._bind_tensor_source(item) for item in items]
        return self._call_sql("llm_cat_multi_nd", dim, *tensors)

    def _lower_cos(self, args: list, kwargs: dict) -> str:
        return self._call_sql("llm_cos_nd", self._bind_tensor_source(args[0]))

    def _lower_sin(self, args: list, kwargs: dict) -> str:
        return self._call_sql("llm_sin_nd", self._bind_tensor_source(args[0]))

    def _lower_sdpa(self, args: list, kwargs: dict) -> str:
        scale = kwargs.get("scale")
        if scale is None:
            raise ValueError("native export requires explicit SDPA scale")
        parts = [
            self._bind_tensor_source(args[0]),
            self._bind_tensor_source(args[1]),
            self._bind_tensor_source(args[2]),
            self._bind_scalar(scale),
            self._bind_scalar(1 if kwargs.get("is_causal", False) else 0),
        ]
        if len(args) > 3 and args[3] is not None:
            parts.append(self._bind_tensor_source(args[3]))
        return self._call_sql("llm_sdpa_nd", *parts)

    def _lower_logical_not(self, args: list, kwargs: dict) -> str:
        return self._call_sql(
            "llm_logical_not_nd",
            self._bind_tensor_source(args[0]),
        )

    def _lower_where(self, args: list, kwargs: dict) -> str:
        return self._call_sql(
            "llm_where_nd",
            self._bind_tensor_source(args[0]),
            self._bind_tensor_or_scalar(args[1]),
            self._bind_tensor_or_scalar(args[2]),
        )

    def _lower_clamp(self, args: list, kwargs: dict) -> str:
        min_value = args[1] if len(args) > 1 else None
        max_value = args[2] if len(args) > 2 else None
        return self._call_sql(
            "llm_clamp_nd",
            self._bind_tensor_source(args[0]),
            self._bind_scalar(min_value),
            self._bind_scalar(max_value),
        )

    def _lower_full(self, args: list, kwargs: dict) -> str:
        fill_value = self._bind_scalar(args[1])
        dims = [self._bind_scalar(item) for item in self._flatten_list(args[0])]
        return self._call_sql(
            "llm_full_nd",
            fill_value,
            *dims,
        )

    def _lower_repeat(self, args: list, kwargs: dict) -> str:
        tensor = self._bind_tensor_source(args[0])
        dims = [self._bind_scalar(item) for item in self._flatten_list(args[1])]
        return self._call_sql(
            "llm_repeat_nd",
            tensor,
            *dims,
        )

    def _lower_silu_mul(self, args: list, kwargs: dict) -> str:
        return self._call_sql(
            "llm_silu_mul_nd",
            self._bind_tensor_source(args[0]),
            self._bind_tensor_source(args[1]),
        )

    def _lower_rope(self, args: list, kwargs: dict) -> str:
        return self._call_sql(
            "llm_rope_nd",
            self._bind_tensor_source(args[0]),
            self._bind_tensor_source(args[1]),
            self._bind_tensor_source(args[2]),
        )

    def _lower_rmsnorm(self, args: list, kwargs: dict) -> str:
        return self._call_sql(
            "llm_rmsnorm_nd",
            self._bind_tensor_source(args[0]),
            self._bind_tensor_source(args[1]),
            self._bind_scalar(args[2]),
        )

    def _lower_repeat_kv(self, args: list, kwargs: dict) -> str:
        return self._call_sql(
            "llm_repeat_kv_nd",
            self._bind_tensor_source(args[0]),
            self._bind_scalar(args[1]),
        )

    def _lower_view_t12(self, args: list, kwargs: dict) -> str:
        tensor = self._bind_tensor_source(args[0])
        dims = [self._bind_scalar(item) for item in args[1:]]
        return self._call_sql(
            "llm_view_t12_nd",
            tensor,
            *dims,
        )

    def _lower_builtin_add(self, args: list, kwargs: dict) -> str:
        return f"SELECT {self._bind_scalar(args[0])} + {self._bind_scalar(args[1])}"

    def _lower_builtin_mul(self, args: list, kwargs: dict) -> str:
        return f"SELECT {self._bind_scalar(args[0])} * {self._bind_scalar(args[1])}"


def _native_graph_from_instrs(graph_name: str, instrs: list, param_names: set[str]) -> dict:
    alias_map: Dict[str, str] = {}
    scalar_targets = {
        "aten.sym_size.int",
        "<built-in function add>",
        "<built-in function mul>",
    }
    for instr in instrs:
        if instr.get("_dt") == _DT_ALIAS:
            target = instr.get("_alias_target")
            if isinstance(target, str):
                while target in alias_map:
                    target = alias_map[target]
                alias_map[instr["name"]] = target

    name_to_slot: Dict[str, int] = {}
    scalar_names: set[str] = set()
    next_slot = 0
    for instr in instrs:
        if instr.get("_dt") in {
            _DT_NORMAL,
            _DT_FUSED,
            _DT_LAZY,
            _DT_PLACEHOLDER,
        }:
            name_to_slot[instr["name"]] = next_slot
            if instr.get("target") in scalar_targets:
                scalar_names.add(instr["name"])
            next_slot += 1

    lowerer = _SqlLowerer(
        name_to_slot=name_to_slot,
        scalar_names=scalar_names,
        param_names=param_names,
        alias_map=alias_map,
    )
    native_instrs: List[dict] = []

    for instr in instrs:
        dt = instr.get("_dt")
        layer = int(instr.get("_layer", -1))
        if dt == _DT_PLACEHOLDER:
            native_instrs.append(
                {
                    "kind": "placeholder",
                    "slot": name_to_slot[instr["name"]],
                    "name": instr["name"],
                    "layer": layer,
                }
            )
            continue

        if dt == _DT_OUTPUT:
            slots = []
            for item in instr.get("_output_args", []):
                if not isinstance(item, str):
                    raise ValueError(f"output arg must be a value ref: {item!r}")
                while item in alias_map:
                    item = alias_map[item]
                if item not in name_to_slot:
                    raise ValueError(f"output arg missing slot: {item!r}")
                slots.append(name_to_slot[item])
            native_instrs.append({"kind": "output", "slots": slots})
            continue

        if dt == _DT_NORMAL:
            sql, args = lowerer.lower_call(instr)
            native_instrs.append(
                {
                    "kind": "sql",
                    "slot": name_to_slot[instr["name"]],
                    "name": instr["name"],
                    "target": instr["target"],
                    "layer": layer,
                    "sql": sql,
                    "args": args,
                }
            )
            continue

        if dt == _DT_FUSED:
            sql, args = lowerer.lower_fused(instr)
            native_instrs.append(
                {
                    "kind": "sql",
                    "slot": name_to_slot[instr["name"]],
                    "name": instr["name"],
                    "target": f"fused:{instr.get('_fused_type')}",
                    "layer": layer,
                    "sql": sql,
                    "args": args,
                }
            )
            continue

        if dt == _DT_LAZY:
            sql, args = lowerer.lower_lazy(instr)
            native_instrs.append(
                {
                    "kind": "sql",
                    "slot": name_to_slot[instr["name"]],
                    "name": instr["name"],
                    "target": instr["target"],
                    "layer": layer,
                    "sql": sql,
                    "args": args,
                }
            )
            continue

        if dt == _DT_ALIAS:
            continue

        raise ValueError(f"unsupported native dispatch type: {dt!r}")

    return {
        "format": _FORMAT,
        "graph": graph_name,
        "slot_count": next_slot,
        "instructions": native_instrs,
    }


def export_native_graphs(
    export_dir: str,
    db_filename: str = "model.db",
    output_dir: str | None = None,
) -> Dict[str, int]:
    output_root = output_dir or export_dir
    os.makedirs(output_root, exist_ok=True)

    engine = StandaloneEngine(
        export_dir,
        db_filename=db_filename,
        single_sql_mode=True,
    )
    try:
        param_names = set(engine._param_blobs)
        graphs = {
            "prefill": _native_graph_from_instrs("prefill", engine._prefill_instrs, param_names),
            "decode": _native_graph_from_instrs("decode", engine._decode_instrs, param_names),
        }
    finally:
        engine.close()

    counts: Dict[str, int] = {}
    for graph_name, payload in graphs.items():
        path = os.path.join(output_root, f"{graph_name}.native.json")
        with open(path, "w") as f:
            json.dump(payload, f, separators=(",", ":"))
        counts[graph_name] = len(payload["instructions"])
    return counts