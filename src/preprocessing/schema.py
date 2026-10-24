from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


CANONICAL_SCHEMA = pa.schema(
    [
        pa.field("series_id", pa.int64(), nullable=False),
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("start_timestamp", pa.timestamp("ns"), nullable=False),
        pa.field("frequency_seconds", pa.int32(), nullable=False),
        pa.field("length", pa.int32(), nullable=False),
        pa.field("n_channels", pa.int16(), nullable=False),
        pa.field("values", pa.list_(pa.float32()), nullable=False),
    ]
)

INDEX_SCHEMA = pa.schema(
    [
        pa.field("series_id", pa.int64(), nullable=False),
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("part_file", pa.string(), nullable=False),
        pa.field("row_index", pa.int64(), nullable=False),
        pa.field("row_group", pa.int32(), nullable=False),
        pa.field("row_in_group", pa.int32(), nullable=False),
        pa.field("length", pa.int32(), nullable=False),
        pa.field("n_channels", pa.int16(), nullable=False),
        pa.field("start_timestamp", pa.timestamp("ns"), nullable=False),
    ]
)


def canonical_table(
    series_ids: Iterable[int],
    source_ids: Iterable[str],
    start_timestamps: Iterable[datetime | np.datetime64],
    frequency_seconds: int | Iterable[int],
    lengths: Iterable[int],
    n_channels: int | Iterable[int],
    values: pa.Array,
) -> pa.Table:
    """Build an exact-schema canonical table without Pythonizing value arrays."""
    ids = np.asarray(list(series_ids), dtype=np.int64)
    n = len(ids)
    frequency = (
        np.full(n, frequency_seconds, dtype=np.int32)
        if np.isscalar(frequency_seconds)
        else np.asarray(list(frequency_seconds), dtype=np.int32)
    )
    channels = (
        np.full(n, n_channels, dtype=np.int16)
        if np.isscalar(n_channels)
        else np.asarray(list(n_channels), dtype=np.int16)
    )
    arrays = [
        pa.array(ids, type=pa.int64()),
        pa.array(list(source_ids), type=pa.string()),
        pa.array(np.asarray(list(start_timestamps), dtype="datetime64[ns]"), type=pa.timestamp("ns")),
        pa.array(frequency, type=pa.int32()),
        pa.array(list(lengths), type=pa.int32()),
        pa.array(channels, type=pa.int16()),
        values.cast(pa.list_(pa.float32())),
    ]
    return pa.Table.from_arrays(arrays, schema=CANONICAL_SCHEMA)


class CanonicalWriter:
    """Rotating Parquet writer with an incrementally written random-access index."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        dataset_name: str,
        source_files: list[str],
        frequency_seconds: int,
        channel_names: list[str],
        source_format: str,
        source_value_dtypes: list[str],
        compression: str = "zstd",
        target_values_per_part: int = 50_000_000,
        row_group_target_values: int = 2_000_000,
        row_group_max_rows: int = 1024,
        overwrite: bool = False,
        extra_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        if self.output_dir.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Output already exists: {self.output_dir}. Pass --overwrite explicitly."
                )
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=False)

        self.dataset_name = dataset_name
        self.source_files = source_files
        self.frequency_seconds = int(frequency_seconds)
        self.channel_names = channel_names
        self.source_format = source_format
        self.source_value_dtypes = source_value_dtypes
        self.compression = compression
        self.target_values_per_part = int(target_values_per_part)
        self.row_group_target_values = int(row_group_target_values)
        self.row_group_max_rows = int(row_group_max_rows)
        self.extra_metadata = extra_metadata or {}

        self._part_writer: pq.ParquetWriter | None = None
        self._part_number = -1
        self._part_file = ""
        self._part_rows = 0
        self._part_values = 0
        self._part_row_group = 0
        self._part_files: list[str] = []
        self._index_writer = pq.ParquetWriter(
            self.output_dir / "series_index.parquet",
            INDEX_SCHEMA,
            compression=compression,
            use_dictionary=["source_id", "part_file"],
            write_statistics=True,
        )

        self.num_series = 0
        self.total_time_steps = 0
        self.total_scalar_observations = 0
        self.missing_values = 0
        self.inserted_missing_time_steps = 0
        self.valid_scalar_observations = 0
        self._length_min: int | None = None
        self._length_max: int | None = None
        self._length_sum = 0
        self._started = False
        self._closed = False
        self._series_buffer: list[
            tuple[int, str, datetime | np.datetime64, int, np.ndarray, int]
        ] = []
        self._series_buffer_values = 0

    def __enter__(self) -> "CanonicalWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()

    def _open_part(self) -> None:
        self._part_number += 1
        self._part_file = f"part-{self._part_number:05d}.parquet"
        self._part_files.append(self._part_file)
        self._part_writer = pq.ParquetWriter(
            self.output_dir / self._part_file,
            CANONICAL_SCHEMA,
            compression=self.compression,
            use_dictionary=["source_id"],
            write_statistics=True,
        )
        self._part_rows = 0
        self._part_values = 0
        self._part_row_group = 0

    def _close_part(self) -> None:
        if self._part_writer is not None:
            self._part_writer.close()
            self._part_writer = None

    @staticmethod
    def _scalar_counts(table: pa.Table) -> np.ndarray:
        lengths = table.column("length").combine_chunks().to_numpy(zero_copy_only=False)
        channels = table.column("n_channels").combine_chunks().to_numpy(zero_copy_only=False)
        counts = lengths.astype(np.int64) * channels.astype(np.int64)
        value_lengths = pc.list_value_length(table.column("values").combine_chunks()).to_numpy(
            zero_copy_only=False
        )
        if not np.array_equal(counts, value_lengths):
            raise ValueError("Every row must satisfy len(values) == length * n_channels")
        return counts

    def add_table(
        self,
        table: pa.Table,
        *,
        missing_count: int | None = None,
        inserted_missing_time_steps: int = 0,
    ) -> None:
        if self._closed:
            raise RuntimeError("Writer is closed")
        if self._series_buffer:
            self._flush_series_buffer()
        if table.schema != CANONICAL_SCHEMA:
            table = table.cast(CANONICAL_SCHEMA)
        if len(table) == 0:
            return
        counts = self._scalar_counts(table)
        if np.any(counts < 0):
            raise ValueError("Negative length/channel count")

        if missing_count is None:
            flat = pc.list_flatten(table.column("values").combine_chunks())
            missing_count = int(pc.sum(pc.cast(pc.is_nan(flat), pa.int64())).as_py() or 0)
        self.missing_values += int(missing_count)
        self.inserted_missing_time_steps += int(inserted_missing_time_steps)

        offset = 0
        while offset < len(table):
            if self._part_writer is None:
                self._open_part()
            if self._part_rows and self._part_values + int(counts[offset]) > self.target_values_per_part:
                self._close_part()
                continue

            remaining_part = max(1, self.target_values_per_part - self._part_values)
            cumulative = np.cumsum(counts[offset:], dtype=np.int64)
            part_rows = int(np.searchsorted(cumulative, remaining_part, side="right"))
            if part_rows == 0:
                part_rows = 1
            part_rows = min(part_rows, len(table) - offset)

            # Keep row groups useful for random access even when series lengths are small.
            local = 0
            while local < part_rows:
                local_counts = counts[offset + local : offset + part_rows]
                cumulative_rg = np.cumsum(local_counts, dtype=np.int64)
                rg_rows = int(
                    np.searchsorted(cumulative_rg, self.row_group_target_values, side="right")
                )
                if rg_rows == 0:
                    rg_rows = 1
                rg_rows = min(rg_rows, self.row_group_max_rows, part_rows - local)
                chunk = table.slice(offset + local, rg_rows)
                chunk_counts = counts[offset + local : offset + local + rg_rows]
                assert self._part_writer is not None
                self._part_writer.write_table(chunk, row_group_size=rg_rows)
                self._write_index(chunk, self._part_row_group, self._part_rows)
                self._update_stats(chunk, chunk_counts)
                self._part_rows += rg_rows
                self._part_values += int(chunk_counts.sum())
                self._part_row_group += 1
                local += rg_rows
            offset += part_rows
            if offset < len(table):
                self._close_part()

    def add_series(
        self,
        *,
        series_id: int,
        source_id: str,
        start_timestamp: datetime | np.datetime64,
        frequency_seconds: int,
        values: np.ndarray,
        inserted_missing_time_steps: int = 0,
    ) -> None:
        matrix = np.asarray(values, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix[:, None]
        if matrix.ndim != 2:
            raise ValueError(f"Expected [T,C], got {matrix.shape}")
        matrix = np.ascontiguousarray(matrix)
        scalar_count = int(matrix.size)
        if self._series_buffer and (
            self._series_buffer_values + scalar_count > self.row_group_target_values
            or len(self._series_buffer) >= self.row_group_max_rows
        ):
            self._flush_series_buffer()
        self._series_buffer.append(
            (
                int(series_id),
                str(source_id),
                start_timestamp,
                int(frequency_seconds),
                matrix,
                int(inserted_missing_time_steps),
            )
        )
        self._series_buffer_values += scalar_count
        if self._series_buffer_values >= self.row_group_target_values:
            self._flush_series_buffer()

    def _flush_series_buffer(self) -> None:
        if not self._series_buffer:
            return
        records = self._series_buffer
        self._series_buffer = []
        self._series_buffer_values = 0
        flats = [record[4].reshape(-1) for record in records]
        offsets_np = np.zeros(len(records) + 1, dtype=np.int64)
        offsets_np[1:] = np.cumsum([len(flat) for flat in flats], dtype=np.int64)
        flat_values = np.concatenate(flats) if flats else np.empty(0, dtype=np.float32)
        lists = pa.ListArray.from_arrays(
            pa.array(offsets_np, type=pa.int64()), pa.array(flat_values, type=pa.float32())
        )
        table = canonical_table(
            [record[0] for record in records],
            [record[1] for record in records],
            [record[2] for record in records],
            [record[3] for record in records],
            [record[4].shape[0] for record in records],
            [record[4].shape[1] for record in records],
            lists,
        )
        self.add_table(
            table,
            missing_count=sum(int(np.isnan(record[4]).sum()) for record in records),
            inserted_missing_time_steps=sum(record[5] for record in records),
        )

    def _write_index(self, table: pa.Table, row_group: int, row_index: int) -> None:
        n = len(table)
        arrays = [
            table.column("series_id").combine_chunks(),
            table.column("source_id").combine_chunks(),
            pa.array([self._part_file] * n, type=pa.string()),
            pa.array(np.arange(row_index, row_index + n, dtype=np.int64)),
            pa.array(np.full(n, row_group, dtype=np.int32)),
            pa.array(np.arange(n, dtype=np.int32)),
            table.column("length").combine_chunks(),
            table.column("n_channels").combine_chunks(),
            table.column("start_timestamp").combine_chunks(),
        ]
        index = pa.Table.from_arrays(arrays, schema=INDEX_SCHEMA)
        self._index_writer.write_table(index, row_group_size=n)

    def _update_stats(self, table: pa.Table, counts: np.ndarray) -> None:
        lengths = table.column("length").combine_chunks().to_numpy(zero_copy_only=False)
        self.num_series += len(table)
        self.total_time_steps += int(lengths.astype(np.int64).sum())
        scalar_count = int(counts.sum())
        self.total_scalar_observations += scalar_count
        self.valid_scalar_observations = self.total_scalar_observations - self.missing_values
        self._length_sum += int(lengths.astype(np.int64).sum())
        batch_min = int(lengths.min())
        batch_max = int(lengths.max())
        self._length_min = batch_min if self._length_min is None else min(self._length_min, batch_min)
        self._length_max = batch_max if self._length_max is None else max(self._length_max, batch_max)
        self._started = True

    def close(self) -> dict[str, Any]:
        if self._closed:
            return self.metadata()
        self._flush_series_buffer()
        self._close_part()
        self._index_writer.close()
        self._closed = True
        metadata = self.metadata()
        with (self.output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        return metadata

    def abort(self) -> None:
        self._close_part()
        try:
            self._index_writer.close()
        finally:
            self._closed = True

    def metadata(self) -> dict[str, Any]:
        processed_bytes = 0
        if self.output_dir.exists():
            processed_bytes = sum(p.stat().st_size for p in self.output_dir.glob("*.parquet"))
        base = {
            "dataset_name": self.dataset_name,
            "source_files": self.source_files,
            "num_series": self.num_series,
            "frequency_seconds": self.frequency_seconds,
            "n_channels": len(self.channel_names),
            "channel_names": self.channel_names,
            "min_length": self._length_min,
            "max_length": self._length_max,
            "mean_length": self._length_sum / self.num_series if self.num_series else None,
            "total_time_steps": self.total_time_steps,
            "total_scalar_observations": self.total_scalar_observations,
            "nominal_scalar_positions": self.total_scalar_observations,
            "valid_scalar_observations": self.total_scalar_observations - self.missing_values,
            "missing_values": self.missing_values,
            "inserted_missing_time_steps": self.inserted_missing_time_steps,
            "value_dtype": "float32",
            "source_value_dtypes": self.source_value_dtypes,
            "value_layout": "time_major_flattened",
            "source_format": self.source_format,
            "compression": self.compression,
            "target_values_per_part": self.target_values_per_part,
            "row_group_target_values": self.row_group_target_values,
            "part_files": self._part_files,
            "part_file_count": len(self._part_files),
            "processed_parquet_bytes": processed_bytes,
        }
        base.update(self.extra_metadata)
        return base
