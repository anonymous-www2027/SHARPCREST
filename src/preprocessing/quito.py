from __future__ import annotations

import math
import shutil
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

from .schema import CanonicalWriter
from .io import file_hash, write_json


def inspect_quito_ids(
    parquet_file: pq.ParquetFile, batch_size: int = 1_048_576
) -> tuple[np.ndarray, dict[str, Any]]:
    unique_ids: set[int] = set()
    closed_ids: set[int] = set()
    last_item: int | None = None
    last_timestamp: int | None = None
    backward_timestamps = 0
    duplicate_timestamps = 0
    regrouped = False
    rows = 0
    min_timestamp: int | None = None
    max_timestamp: int | None = None
    for batch in parquet_file.iter_batches(
        batch_size=batch_size, columns=["item_id", "date_time"], use_threads=True
    ):
        ids = batch.column(0).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        timestamps = (
            batch.column(1).to_numpy(zero_copy_only=False).astype("datetime64[ns]").astype(np.int64)
        )
        rows += len(ids)
        unique_ids.update(map(int, np.unique(ids)))
        if not len(ids):
            continue
        batch_min, batch_max = int(timestamps.min()), int(timestamps.max())
        min_timestamp = batch_min if min_timestamp is None else min(min_timestamp, batch_min)
        max_timestamp = batch_max if max_timestamp is None else max(max_timestamp, batch_max)
        same = ids[1:] == ids[:-1]
        backward_timestamps += int(np.count_nonzero(same & (timestamps[1:] < timestamps[:-1])))
        duplicate_timestamps += int(np.count_nonzero(same & (timestamps[1:] == timestamps[:-1])))
        if last_item is not None:
            if int(ids[0]) != last_item:
                closed_ids.add(last_item)
            elif int(timestamps[0]) < int(last_timestamp):
                backward_timestamps += 1
            elif int(timestamps[0]) == int(last_timestamp):
                duplicate_timestamps += 1
            if int(ids[0]) in closed_ids:
                regrouped = True
        changes = np.flatnonzero(ids[1:] != ids[:-1]) + 1
        for index in changes:
            closed_ids.add(int(ids[index - 1]))
            if int(ids[index]) in closed_ids:
                regrouped = True
        last_item, last_timestamp = int(ids[-1]), int(timestamps[-1])
    ids_array = np.asarray(sorted(unique_ids), dtype=np.int64)
    return ids_array, {
        "source_row_count_scanned": rows,
        "source_unique_item_ids": len(ids_array),
        "source_items_regrouped": regrouped,
        "source_backward_timestamp_pairs": backward_timestamps,
        "source_duplicate_adjacent_timestamp_pairs": duplicate_timestamps,
        "source_grouped_and_timestamp_sorted": not regrouped and backward_timestamps == 0,
        "source_min_timestamp": np.datetime_as_string(np.datetime64(min_timestamp, "ns")),
        "source_max_timestamp": np.datetime_as_string(np.datetime64(max_timestamp, "ns")),
    }


def _partition_source(
    parquet_file: pq.ParquetFile,
    selected_ids: np.ndarray,
    temp_dir: Path,
    *,
    max_partition_rows: int = 2_000_000,
    batch_size: int = 262_144,
) -> tuple[list[Path], int]:
    source_rows = parquet_file.metadata.num_rows
    # Full conversions use enough partitions to cap the in-memory external-sort run.
    partition_count = max(1, math.ceil(source_rows / max_partition_rows))
    partition_count = min(partition_count, len(selected_ids))
    series_per_partition = max(1, math.ceil(len(selected_ids) / partition_count))
    partition_count = math.ceil(len(selected_ids) / series_per_partition)

    temp_dir.mkdir(parents=True, exist_ok=False)
    paths = [temp_dir / f"partition-{index:05d}.arrow" for index in range(partition_count)]
    sinks: dict[int, pa.OSFile] = {}
    writers: dict[int, ipc.RecordBatchFileWriter] = {}
    options = ipc.IpcWriteOptions(compression="zstd")
    columns = ["item_id", "date_time"] + sorted(
        [name for name in parquet_file.schema_arrow.names if name.startswith("ind_")],
        key=lambda value: int(value.split("_")[1]),
    )
    written_rows = 0
    try:
        for batch in parquet_file.iter_batches(
            batch_size=batch_size, columns=columns, use_threads=True
        ):
            ids = batch.column(0).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
            positions = np.searchsorted(selected_ids, ids)
            valid = positions < len(selected_ids)
            if not np.all(valid):
                safe = np.minimum(positions, len(selected_ids) - 1)
                valid &= selected_ids[safe] == ids
            else:
                valid &= selected_ids[positions] == ids
            selected_rows = np.flatnonzero(valid)
            if not len(selected_rows):
                continue
            positions = positions[selected_rows]
            buckets = positions // series_per_partition
            order = np.argsort(buckets, kind="stable")
            ordered_indices = selected_rows[order]
            ordered_buckets = buckets[order]
            ordered = batch.take(pa.array(ordered_indices, type=pa.int64()))
            boundaries = np.r_[0, np.flatnonzero(ordered_buckets[1:] != ordered_buckets[:-1]) + 1, len(order)]
            for start, stop in zip(boundaries[:-1], boundaries[1:]):
                bucket = int(ordered_buckets[start])
                if bucket not in writers:
                    sink = pa.OSFile(str(paths[bucket]), "wb")
                    sinks[bucket] = sink
                    writers[bucket] = ipc.new_file(sink, ordered.schema, options=options)
                writers[bucket].write_batch(ordered.slice(int(start), int(stop - start)))
                written_rows += int(stop - start)
    finally:
        for writer in writers.values():
            writer.close()
        for sink in sinks.values():
            sink.close()
    return [path for path in paths if path.exists()], written_rows


def _regularize_group(
    table: pa.Table, start: int, length: int, value_columns: list[str], frequency_seconds: int
) -> tuple[np.datetime64, np.ndarray, int]:
    group = table.slice(start, length)
    timestamps = (
        group.column("date_time")
        .combine_chunks()
        .to_numpy(zero_copy_only=False)
        .astype("datetime64[ns]")
        .astype(np.int64)
    )
    step_ns = frequency_seconds * 1_000_000_000
    differences = np.diff(timestamps)
    if np.any(differences == 0):
        raise ValueError("Duplicate timestamp within a Quito item_id")
    if np.any(differences < 0):
        raise AssertionError("Partition sort did not order timestamps")
    if np.any(differences % step_ns != 0):
        raise ValueError("Quito timestamp is not aligned to the declared regular frequency")
    values = np.column_stack(
        [
            group.column(name).combine_chunks().to_numpy(zero_copy_only=False)
            for name in value_columns
        ]
    ).astype(np.float32, copy=False)
    grid_length = int((int(timestamps[-1]) - int(timestamps[0])) // step_ns + 1)
    inserted = grid_length - len(timestamps)
    if inserted:
        regular = np.full((grid_length, len(value_columns)), np.nan, dtype=np.float32)
        positions = ((timestamps - timestamps[0]) // step_ns).astype(np.int64)
        regular[positions] = values
        values = regular
    return np.datetime64(int(timestamps[0]), "ns"), np.ascontiguousarray(values), inserted


def convert_quito(
    *,
    dataset_name: str,
    source: str | Path,
    output_dir: str | Path,
    compression: str = "zstd",
    target_values_per_part: int = 50_000_000,
    sample_series: int | None = None,
    max_partition_rows: int = 2_000_000,
    expected_source_sha256: str | None = None,
    source_version: str | None = None,
) -> dict[str, Any]:
    if dataset_name not in ("quitobench_hour", "quito_corpus_hour"):
        raise ValueError("Only Quito-Hour sources are supported")
    if pa.__version__ == "19.0.0":
        raise RuntimeError("Quito raw Parquet requires pyarrow>=19.0.1 (19.0.0 has a reader regression)")
    source = Path(source)
    if max_partition_rows < 1 or target_values_per_part < 1 or (sample_series is not None and sample_series < 1):
        raise ValueError("Partition, part and sample sizes must be positive")
    source_sha256 = file_hash(source)
    if expected_source_sha256 is not None and source_sha256 != expected_source_sha256:
        raise ValueError('Raw Quito source checksum mismatch')
    parquet_file = pq.ParquetFile(source)
    names = parquet_file.schema_arrow.names
    if "item_id" not in names or "date_time" not in names:
        raise ValueError("Quito source requires item_id and date_time")
    value_columns = sorted(
        [name for name in names if name.startswith("ind_")],
        key=lambda value: int(value.split("_")[1]),
    )
    if not value_columns:
        raise ValueError("Quito source contains no ind_* value columns")
    frequency_seconds = 3600
    all_ids, ordering = inspect_quito_ids(parquet_file)
    selected_ids = all_ids if sample_series is None else all_ids[: int(sample_series)]
    if not len(selected_ids):
        raise ValueError("No Quito series selected")

    writer = CanonicalWriter(
        output_dir,
        dataset_name=dataset_name,
        source_files=[str(source.resolve())],
        frequency_seconds=frequency_seconds,
        channel_names=value_columns,
        source_format="long_parquet",
        source_value_dtypes=[str(parquet_file.schema_arrow.field(name).type) for name in value_columns],
        compression=compression,
        target_values_per_part=target_values_per_part,
        overwrite=False,
        extra_metadata={
            **ordering,
            "source_row_count": parquet_file.metadata.num_rows,
            "source_row_group_count": parquet_file.metadata.num_row_groups,
            "sort_strategy": "disk_backed_id_range_partition_then_item_id_timestamp_sort",
        },
    )
    temp_dir = Path(output_dir).parent / f".{Path(output_dir).name}.sort-{uuid.uuid4().hex}"
    try:
        partitions, selected_rows = _partition_source(
            parquet_file,
            selected_ids,
            temp_dir,
            max_partition_rows=max_partition_rows if sample_series is None else max(1, parquet_file.metadata.num_rows),
        )
        writer.extra_metadata["selected_source_rows"] = selected_rows
        for partition_path in partitions:
            with pa.memory_map(str(partition_path), "r") as source_map:
                table = ipc.open_file(source_map).read_all()
            table = table.sort_by([("item_id", "ascending"), ("date_time", "ascending")])
            ids = table.column("item_id").combine_chunks().to_numpy(zero_copy_only=False)
            starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
            stops = np.r_[starts[1:], len(ids)]
            for start, stop in zip(starts, stops):
                item_id = int(ids[start])
                series_id = int(np.searchsorted(all_ids, item_id))
                if sample_series is not None:
                    series_id = int(np.searchsorted(selected_ids, item_id))
                first_timestamp, values, inserted = _regularize_group(
                    table, int(start), int(stop - start), value_columns, frequency_seconds
                )
                writer.add_series(
                    series_id=series_id,
                    source_id=str(item_id),
                    start_timestamp=first_timestamp,
                    frequency_seconds=frequency_seconds,
                    values=values,
                    inserted_missing_time_steps=inserted,
                )
        metadata = writer.close()
        metadata["source_sha256"] = source_sha256
        metadata["source_version"] = source_version
        write_json(Path(output_dir) / "metadata.json", metadata)
        files = sorted(Path(output_dir).glob("*.parquet")) + [Path(output_dir) / "metadata.json"]
        write_json(Path(output_dir) / "checksums.json", {p.name: file_hash(p) for p in files})
    except BaseException:
        writer.abort()
        raise
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
    return metadata
