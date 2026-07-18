from __future__ import annotations

import io
import math
import tarfile
from pathlib import Path
from typing import Any

from .native_readers import RestrictedNumpyUnpickler


def split_tar_uri(uri: str) -> tuple[Path, str] | None:
    if not uri.startswith("tar://") or "!" not in uri:
        return None
    archive, member = uri[len("tar://") :].split("!", 1)
    return Path(archive), member


class ParquetSourceReader:
    """Read an episode's tabular signals from Parquet, OXE pickle, or HDF5.

    The historical class name is retained for API compatibility. Native sources
    are exposed as Arrow tables so cleaning and materialization share exactly the
    same decode and derivation path.
    """

    def __init__(self) -> None:
        self._archive_path: Path | None = None
        self._archive: tarfile.TarFile | None = None

    def close(self) -> None:
        if self._archive is not None:
            self._archive.close()
        self._archive = None
        self._archive_path = None

    def __enter__(self) -> "ParquetSourceReader":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _member_bytes(self, archive_path: Path, member_name: str) -> bytes:
        if self._archive_path != archive_path:
            self.close()
            self._archive = tarfile.open(archive_path, mode="r:*")
            self._archive_path = archive_path
        assert self._archive is not None
        handle = self._archive.extractfile(member_name)
        if handle is None:
            raise FileNotFoundError(f"Archive member not found: {archive_path}!{member_name}")
        return handle.read()

    def read_table(self, uri: str, columns: list[str] | None = None):
        import pyarrow.parquet as pq

        tar_source = split_tar_uri(uri)
        if tar_source is None:
            path = Path(uri)
            if not path.is_file():
                raise FileNotFoundError(path)
            available = pq.ParquetFile(path).schema_arrow.names
            selected = [name for name in (columns or available) if name in available]
            return pq.read_table(path, columns=selected)

        archive_path, member_name = tar_source
        payload = self._member_bytes(archive_path, member_name)
        source = io.BytesIO(payload)
        available = pq.ParquetFile(source).schema_arrow.names
        selected = [name for name in (columns or available) if name in available]
        source.seek(0)
        return pq.read_table(source, columns=selected)

    @staticmethod
    def _source_uri(record: dict[str, Any]) -> str:
        references = record.get("references") or {}
        return str(references.get("data") or record.get("source_uri") or "")

    def source_kind(self, record: dict[str, Any]) -> str | None:
        configured = str(
            ((record.get("metadata") or {}).get("native_conversion") or {}).get(
                "source_format"
            )
            or ""
        )
        if configured:
            return configured
        uri = self._source_uri(record)
        lowered = uri.lower()
        tar_source = split_tar_uri(uri)
        member = tar_source[1].lower() if tar_source else ""
        if lowered.endswith(".parquet") or ".parquet" in lowered:
            return "parquet"
        if member.endswith(".data.pickle") or lowered.endswith(".data.pickle"):
            return "oxe_pickle"
        if lowered.endswith((".h5", ".hdf5")) or member.endswith((".h5", ".hdf5")):
            return "hdf5"
        return None

    def supports_record(self, record: dict[str, Any]) -> bool:
        return self.source_kind(record) in {"parquet", "oxe_pickle", "hdf5"}

    @staticmethod
    def _nested_value(root: Any, dotted_key: str) -> Any:
        value = root
        for token in dotted_key.split("."):
            if not isinstance(value, dict) or token not in value:
                raise KeyError(dotted_key)
            value = value[token]
        return value

    @staticmethod
    def _python_value(value: Any) -> Any:
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, (bool, int, float, str)):
            return value
        return value

    def _read_oxe_table(
        self, record: dict[str, Any], source_keys: list[str]
    ):
        import pyarrow as pa

        uri = self._source_uri(record)
        tar_source = split_tar_uri(uri)
        if tar_source is None:
            raise ValueError(f"OXE source must be a tar URI: {uri}")
        archive_path, member_name = tar_source
        payload = RestrictedNumpyUnpickler(
            io.BytesIO(self._member_bytes(archive_path, member_name))
        ).load()
        if not isinstance(payload, dict) or not isinstance(payload.get("steps"), list):
            raise ValueError("OXE pickle must contain a list-valued `steps` field")
        steps = payload["steps"]
        arrays: dict[str, Any] = {}
        for key in source_keys:
            if key in {"timestamp", "frame_index"}:
                continue
            values: list[Any] = []
            for step in steps:
                try:
                    value = self._nested_value(step, key)
                except KeyError:
                    values = []
                    break
                values.append(self._python_value(value))
            if values:
                arrays[key] = values
        fps = float(record.get("fps") or 0.0)
        if fps <= 0:
            raise ValueError("OXE native decode requires a verified or nominal source fps")
        arrays["timestamp"] = [index / fps for index in range(len(steps))]
        arrays["frame_index"] = list(range(len(steps)))
        return pa.table(arrays)

    def _read_hdf5_table(
        self, record: dict[str, Any], source_keys: list[str]
    ):
        import h5py
        import pyarrow as pa

        uri = self._source_uri(record)
        tar_source = split_tar_uri(uri)
        payload: io.BytesIO | None = None
        if tar_source is None:
            source: Any = Path(uri)
            if not source.is_file():
                raise FileNotFoundError(source)
        else:
            archive_path, member_name = tar_source
            payload = io.BytesIO(self._member_bytes(archive_path, member_name))
            source = payload

        conversion = (record.get("metadata") or {}).get("native_conversion") or {}
        valid_index_keys = [str(value) for value in conversion.get("valid_index_keys") or []]
        valid_index_policy = str(conversion.get("valid_index_policy") or "intersection")
        if valid_index_policy != "intersection":
            raise ValueError(f"Unsupported HDF5 valid_index_policy: {valid_index_policy}")

        arrays: dict[str, Any] = {}
        row_count: int | None = None
        valid_index_sets: list[set[int]] = []
        with h5py.File(source, mode="r") as handle:
            for key in source_keys:
                if key == "frame_index" or key not in handle:
                    continue
                dataset = handle[key]
                if not hasattr(dataset, "shape") or not dataset.shape:
                    continue
                values = dataset[...]
                count = int(values.shape[0])
                row_count = count if row_count is None else min(row_count, count)
                arrays[key] = values
            for key in valid_index_keys:
                if key not in handle:
                    raise KeyError(f"Configured HDF5 valid index dataset is missing: {key}")
                values = handle[key][...]
                if getattr(values, "ndim", 0) != 1:
                    raise ValueError(f"HDF5 valid index dataset must be 1D: {key}")
                valid_index_sets.append({int(value) for value in values.tolist()})
        if row_count is None:
            raise ValueError(f"No requested HDF5 signal datasets found in {uri}")
        if valid_index_sets:
            selected_indices = sorted(
                index
                for index in set.intersection(*valid_index_sets)
                if 0 <= index < row_count
            )
            if not selected_indices:
                raise ValueError(f"HDF5 valid index intersection is empty in {uri}")
        else:
            selected_indices = list(range(row_count))
        arrays = {
            key: self._python_value(value[selected_indices]) for key, value in arrays.items()
        }
        timestamp_key = str(
            ((record.get("metadata") or {}).get("native_conversion") or {}).get(
                "timestamp_key"
            )
            or ""
        )
        if timestamp_key and timestamp_key in arrays:
            timestamps = [float(value) for value in arrays[timestamp_key]]
            scale = 1.0
            if len(timestamps) >= 2:
                finite_deltas = [
                    right - left
                    for left, right in zip(timestamps, timestamps[1:])
                    if math.isfinite(left) and math.isfinite(right) and right > left
                ]
                if finite_deltas and sorted(finite_deltas)[len(finite_deltas) // 2] > 1e5:
                    scale = 1e-9
            origin = timestamps[0]
            arrays["timestamp"] = [(value - origin) * scale for value in timestamps]
        else:
            fps = float(record.get("fps") or 0.0)
            if fps <= 0:
                raise ValueError("HDF5 native decode requires timestamp_key or source fps")
            origin = selected_indices[0]
            arrays["timestamp"] = [
                (index - origin) / fps for index in selected_indices
            ]
        arrays["frame_index"] = selected_indices
        return pa.table(arrays)

    @staticmethod
    def _rpy_pose_to_rotvec(row: Any) -> list[float]:
        values = [float(value) for value in row]
        if len(values) != 6:
            raise ValueError(f"Expected xyz+rpy pose with 6 values, got {len(values)}")
        x, y, z, roll, pitch, yaw = values
        cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
        cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
        cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        vector_norm = math.sqrt(qx * qx + qy * qy + qz * qz)
        if vector_norm <= 1e-12:
            rotation = [0.0, 0.0, 0.0]
        else:
            angle = 2.0 * math.atan2(vector_norm, max(-1.0, min(1.0, qw)))
            if angle > math.pi:
                angle -= 2.0 * math.pi
            rotation = [
                angle * qx / vector_norm,
                angle * qy / vector_norm,
                angle * qz / vector_norm,
            ]
        return [x, y, z, *rotation]

    def _apply_derived_columns(self, table: Any, record: dict[str, Any]):
        import pyarrow as pa

        conversion = ((record.get("metadata") or {}).get("native_conversion") or {})
        derived = conversion.get("derived_columns") or {}
        for target_key, raw_spec in derived.items():
            spec = raw_spec if isinstance(raw_spec, dict) else {}
            source_key = str(spec.get("source_key") or "")
            if source_key not in table.column_names:
                raise KeyError(f"Derived column source is missing: {source_key}")
            values = table[source_key].to_pylist()
            indices = spec.get("indices")
            if isinstance(indices, list):
                selected: list[list[Any]] = []
                for row in values:
                    if not isinstance(row, (list, tuple)):
                        raise ValueError(f"Cannot index scalar source column: {source_key}")
                    selected.append([row[int(index)] for index in indices])
                values = selected
            operation = str(spec.get("operation") or "identity")
            if operation == "pose_rpy_to_rotvec":
                values = [self._rpy_pose_to_rotvec(row) for row in values]
            elif operation == "next_row_hold_last":
                values = [*values[1:], values[-1]] if values else []
            elif operation != "identity":
                raise ValueError(f"Unsupported derived-column operation: {operation}")
            array = pa.array(values)
            if target_key in table.column_names:
                table = table.set_column(
                    table.schema.get_field_index(target_key), target_key, array
                )
            else:
                table = table.append_column(target_key, array)
        return table

    def read_record(
        self, record: dict[str, Any], columns: list[str] | None = None
    ):
        kind = self.source_kind(record)
        if kind is None:
            raise ValueError(f"Unsupported native episode source: {self._source_uri(record)}")
        requested = list(dict.fromkeys(columns or []))
        derived = (
            ((record.get("metadata") or {}).get("native_conversion") or {}).get(
                "derived_columns"
            )
            or {}
        )
        source_keys = list(requested)
        timestamp_key = str(
            ((record.get("metadata") or {}).get("native_conversion") or {}).get(
                "timestamp_key"
            )
            or ""
        )
        if timestamp_key:
            source_keys.append(timestamp_key)
        for target_key, spec in derived.items():
            if not requested or target_key in requested:
                source_key = str((spec or {}).get("source_key") or "")
                if source_key:
                    source_keys.append(source_key)
        source_keys = list(dict.fromkeys(source_keys))
        uri = self._source_uri(record)
        if kind == "parquet":
            table = self.read_table(uri, columns=source_keys or None)
        elif kind == "oxe_pickle":
            table = self._read_oxe_table(record, source_keys)
        elif kind == "hdf5":
            table = self._read_hdf5_table(record, source_keys)
        else:
            raise AssertionError(kind)
        table = self._apply_derived_columns(table, record)
        if not requested:
            return table
        selected = [key for key in requested if key in table.column_names]
        return table.select(selected)
