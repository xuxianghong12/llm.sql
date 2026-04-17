import torch

from src.backends.sqlite_kvcache_engine import SqliteKVCacheGraphExecutor


class TinyLinear(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(8, 8, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _export_tiny_program() -> torch.export.ExportedProgram:
    module = TinyLinear().eval()
    example = (torch.randn(1, 8),)
    with torch.no_grad():
        return torch.export.export(module, example, strict=False)


def test_kvcache_file_compile_keeps_param_name_refs(tmp_path) -> None:
    db_path = tmp_path / "model.db"
    executor = SqliteKVCacheGraphExecutor(db_path=str(db_path))
    try:
        executor._compile_params(_export_tiny_program())

        assert executor._param_data
        assert all(
            isinstance(value, str) and value == name
            for name, value in executor._param_data.items()
        )

        stored_names = {
            name
            for (name,) in executor.connection.execute(
                "SELECT name FROM model_params ORDER BY name"
            )
        }
        assert stored_names == set(executor._param_data)
    finally:
        executor.connection.close()