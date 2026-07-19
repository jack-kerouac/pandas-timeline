from typing import Callable, Self
from collections.abc import Sequence

import pandas as pd
import numpy as np

class Timeline[T]:
    """
    Timeline<T> represents a value of type <T> as it changes over a continuous period.
    It behaves like a step function: the value remains constant for a segment of time
    and then instantly changes to a new value at the start of the next segment.

    Internally, Timeline is backed by a pandas DataFrame with columns:
    - 'start': pd.Timestamp, start of the segment (inclusive)
    - 'end': pd.Timestamp, end of the segment (exclusive)
    - 'value': value of type T for the segment

    Invariants:
    - Segments are back-to-back, non-overlapping, and contiguous.
    - For any point in time within the total duration, there is exactly one value defined.

    Key Operations:
    - cross_product: Combine multiple timelines into a new timeline with composite values for each segment.
    - merge_adjacent: Merge adjacent segments with identical values to simplify the timeline.
    - map: Transform the value in each segment using a function.
    - slice: Extract a sub-period as a new Timeline.

    The internal DataFrame is protected from external modification; only a copy is exposed via the 'df' property.
    """
    def __init__(self, df: pd.DataFrame):
        """
        Private constructor. Use Timeline.from_segments to create a Timeline.
        df: DataFrame with columns ['start', 'end', 'value']
        """
        self._df = df.copy()

    @staticmethod
    def from_segments(segments: Sequence[tuple[pd.Timestamp, pd.Timestamp, T]], /) -> 'Timeline[T]':
        """
        segments: List of (start, end, value) tuples.
        """
        df = pd.DataFrame(segments, columns=['start', 'end', 'value'])
        Timeline._validate(df)
        return Timeline(df)

    @staticmethod
    def from_dataframe(df: pd.DataFrame, /) -> 'Timeline':
        Timeline._validate(df)
        return Timeline(df.copy())

    @classmethod
    def _from_trusted(cls, df: pd.DataFrame, /) -> Self:
        """
        Internal constructor for a DataFrame that is freshly created, not shared
        with any external caller, and known to satisfy the invariants (e.g. a
        value-only transform of an already-validated frame). Skips validation
        and the defensive copy.
        """
        timeline = cls.__new__(cls)
        timeline._df = df
        return timeline

    @staticmethod
    def from_segments_with_gaps(
        segments: Sequence[tuple[pd.Timestamp, pd.Timestamp, T]], 
        /, 
        gap_value=pd.NA
    ) -> 'Timeline[T]':
        """
        Create Timeline from segments that may have gaps. Gaps are filled with gap_value.

        segments: List of (start, end, value) tuples, can be non-contiguous.
        gap_value: Value to use for gap segments (default: pd.NA).

        See pandas documentation about pd.NA semantics:
        https://pandas.pydata.org/docs/user_guide/missing_data.html#na-semantics
        """
        if not segments:
            raise ValueError("Timeline must have at least one segment")

        # Sort by start time
        segments = sorted(segments, key=lambda x: x[0])

        # Fill gaps with gap_value segments
        filled_segments = []
        for i, (start, end, value) in enumerate(segments):
            # Add gap-filling segment if needed
            if i > 0:
                prev_segment = filled_segments[-1]
                if prev_segment[1] < start:
                    filled_segments.append((prev_segment[1], start, gap_value))
            filled_segments.append((start, end, value))

        return Timeline.from_segments(filled_segments)

    @staticmethod
    def _validate(df: pd.DataFrame) -> None:
        if len(df) == 0:
            raise ValueError("Timeline must have at least one segment")
        if not df.columns.isin(['start', 'end', 'value']).all():
            raise ValueError("DataFrame must contain 'start', 'end', and 'value' columns")
        if df['start'].dtype != 'datetime64[ns]':
            raise TypeError("Start must be a datetime64 column")
        if df['end'].dtype != 'datetime64[ns]':
            raise TypeError("End must be a datetime64 column")
        if not df['start'].is_monotonic_increasing:
            raise ValueError("Segments must be sorted by start time")
        if len(df) > 1 and not (df['end'].iloc[:-1].values == df['start'].iloc[1:].values).all():
            raise ValueError("Segments must be contiguous")
        if not (df['start'] < df['end']).all():
            raise ValueError("Start must be before end")

    @property
    def df(self) -> pd.DataFrame:
        """
        Returns a copy of the internal DataFrame representing the timeline segments.

        Note:
            Direct modification of the returned DataFrame will not affect the Timeline object.
            To maintain class invariants, always use Timeline methods for mutation.
        """
        return self._df.copy()

    @property
    def start(self) -> pd.Timestamp:
        return self._df['start'].iloc[0]

    @property
    def end(self) -> pd.Timestamp:
        return self._df['end'].iloc[-1]

    def map[U](self, func: Callable[[T], U], /) -> 'Timeline[U]':
        new_df = self._df.assign(value=self._df['value'].apply(func))
        return Timeline._from_trusted(new_df)

    def filter(self, predicate: Callable[[T], bool], /, *, gap_value=pd.NA) -> Self:
        new_df = self._df.copy()
        mask = new_df['value'].apply(predicate)
        if not mask.all():
            # Cast to object first: assigning e.g. pd.NA into an int64 column
            # would upcast surviving values to float64.
            new_df['value'] = new_df['value'].astype(object)
            new_df.loc[~mask, 'value'] = gap_value
        return Timeline._from_trusted(new_df)

    def slice(self, start: pd.Timestamp, end: pd.Timestamp) -> Self:
        """Extract a sub-period as a new Timeline.

        Args:
            start: Start time for the slice (inclusive).
            end: End time for the slice (exclusive).

        Returns:
            A new Timeline containing only segments overlapping with [start, end).

        Raises:
            ValueError: If start is not before end, or if [start, end) does not
                overlap the timeline's duration.

        Note:
            Segments that partially overlap with the slice period will be
            truncated to fit within the specified bounds. Bounds extending
            beyond the timeline's duration are clamped to it.

        Performance:
            Slice as early as possible in a transformation chain. Per-segment
            operations such as map(), filter(), and cross_product() do work
            proportional to the number of segments, so segments discarded by a
            later slice() are processed for nothing. On a 100k-segment
            timeline, slicing 1% before three map() calls ran ~30x faster
            (~1.6 ms vs ~50 ms) than slicing after them.
        """
        if start >= end:
            raise ValueError(f"Slice start must be before end, got [{start}, {end})")
        if end <= self.start or start >= self.end:
            raise ValueError(
                f"Slice [{start}, {end}) does not overlap timeline duration [{self.start}, {self.end})"
            )
        # Extract segments overlapping with [start, end); at least one matches
        mask = (self._df['end'] > start) & (self._df['start'] < end)
        sliced = self._df[mask].copy()
        # Adjust the first segment's start if needed
        if sliced.iloc[0]['start'] < start:
            sliced.iat[0, sliced.columns.get_loc('start')] = start
        # Adjust the last segment's end if needed
        if sliced.iloc[-1]['end'] > end:
            sliced.iat[-1, sliced.columns.get_loc('end')] = end
        return Timeline._from_trusted(sliced)

    def merge_adjacent(self) -> Self:
        """Merge adjacent segments with identical values.

        Returns:
            A new Timeline with adjacent segments containing the same value merged together.
            This operation reduces the number of segments by combining consecutive
            segments that have identical values. Adjacent missing-value segments
            (e.g. gaps produced by filter()) are considered equal and merged.
        """
        values = self._df['value']
        prev = values.shift()
        # A run boundary is where the value differs from the previous row.
        # pd.NA comparisons return NA: treat NA-vs-NA as equal (merge gaps),
        # NA-vs-value as a change (fillna(True)).
        changed = ((values != prev) & ~(values.isna() & prev.isna())).fillna(True)
        changed = changed.astype(bool)
        changed.iloc[0] = True
        group_id = changed.cumsum()
        merged = self._df.groupby(group_id).agg(start=('start', 'first'), end=('end', 'last'))
        # Take each run's value by position: groupby 'first' would skip NA.
        merged['value'] = values[changed.values].values
        return Timeline.from_dataframe(merged.reset_index(drop=True))

    @staticmethod
    def cross_product(timelines: tuple['Timeline', ...], /) -> 'Timeline[tuple]':
        # All timelines must cover the same total duration
        starts = [tl.start for tl in timelines]
        ends = [tl.end for tl in timelines]
        assert all(s == starts[0] for s in starts), "All timelines must start at the same time"
        assert all(e == ends[0] for e in ends), "All timelines must end at the same time"

        # Collect all unique boundaries
        boundaries = set()
        for tl in timelines:
            boundaries.update(tl._df['start'])
            boundaries.update(tl._df['end'])
        boundaries = sorted(boundaries)

        # Precompute segment DataFrames and initialize pointers
        dfs = [tl._df.reset_index(drop=True) for tl in timelines]
        idxs = [0] * len(timelines)
        segments = []

        for i in range(len(boundaries) - 1):
            seg_start = boundaries[i]
            seg_end = boundaries[i + 1]
            values = []
            for j, df in enumerate(dfs):
                # Advance pointer if needed
                while idxs[j] + 1 < len(df) and df.at[idxs[j] + 1, 'start'] <= seg_start:
                    idxs[j] += 1
                values.append(df.at[idxs[j], 'value'])
            segments.append((seg_start, seg_end, tuple(values)))
        return Timeline.from_segments(segments)

    def __len__(self) -> int:
        """Return the number of segments in the timeline."""
        return len(self._df)

    def __iter__(self):
        """Iterate over timeline segments as (start, end, value) tuples."""
        for _, row in self._df.iterrows():
            yield (row['start'], row['end'], row['value'])

    def __eq__(self, other) -> bool:
        """Compare two timelines for equality."""
        if not isinstance(other, Timeline):
            return NotImplemented
        return self._df.equals(other._df)

    def at(self, timestamp: pd.Timestamp, /) -> T:
        """Get the value at a specific timestamp.

        Args:
            timestamp: The timestamp to query for a value.

        Returns:
            The value of type T that is active at the given timestamp.

        Raises:
            ValueError: If the timestamp is outside the timeline's duration.

        Note:
            This operation uses binary search for O(log n) performance.
            For timestamps exactly at segment boundaries, returns the value
            of the segment that starts at that timestamp.
        """
        if timestamp < self.start or timestamp >= self.end:
            raise ValueError(f"Timestamp {timestamp} is outside timeline duration [{self.start}, {self.end})")
        
        # Use binary search to find the correct segment
        # searchsorted with side='right' finds the insertion point after existing values
        # Subtracting 1 gives us the index of the segment containing the timestamp
        idx = self._df['start'].searchsorted(timestamp, side='right') - 1
        return self._df.iloc[idx]['value']

    def __repr__(self) -> str:
        return f"Timeline({self._df})"


class GridTimeline[T]:
    """
    GridTimeline<T> wraps a Timeline<T> whose segment boundaries all lie on a
    fixed time grid (e.g. quarter hours). Alignment is validated once at
    construction; every operation preserves it and returns a GridTimeline,
    so the guarantee holds through transformation chains.

    Use the 'timeline' property to unwrap the underlying Timeline.
    """
    def __init__(self, timeline: Timeline[T], freq: str | pd.Timedelta, /):
        """
        timeline: The Timeline to wrap.
        freq: Grid spacing, e.g. '15min'. Grid points are multiples of freq
            anchored at midnight.

        Raises:
            ValueError: If freq is not positive or any segment boundary does
                not lie on the grid.
        """
        freq = pd.Timedelta(freq)
        if freq <= pd.Timedelta(0):
            raise ValueError(f"freq must be positive, got {freq}")
        # Contiguity means checking every start plus the final end covers all boundaries.
        boundaries = pd.concat([timeline._df['start'], timeline._df['end'].iloc[-1:]])
        misaligned = boundaries[boundaries.dt.floor(freq) != boundaries]
        if not misaligned.empty:
            raise ValueError(f"Boundary {misaligned.iloc[0]} does not lie on the {freq} grid")
        self._timeline = timeline
        self._freq = freq

    @staticmethod
    def from_segments(segments: Sequence[tuple[pd.Timestamp, pd.Timestamp, T]], /, *, freq: str | pd.Timedelta) -> 'GridTimeline[T]':
        """
        segments: List of (start, end, value) tuples.
        freq: Grid spacing, e.g. '15min'.
        """
        return GridTimeline(Timeline.from_segments(segments), freq)

    @classmethod
    def _from_trusted(cls, timeline: Timeline, freq: pd.Timedelta, /) -> Self:
        """
        Internal constructor for a Timeline produced by an operation that
        cannot break grid alignment. Skips validation.
        """
        grid_timeline = cls.__new__(cls)
        grid_timeline._timeline = timeline
        grid_timeline._freq = freq
        return grid_timeline

    @property
    def timeline(self) -> Timeline[T]:
        """The wrapped Timeline."""
        return self._timeline

    @property
    def freq(self) -> pd.Timedelta:
        return self._freq

    @property
    def df(self) -> pd.DataFrame:
        return self._timeline.df

    @property
    def start(self) -> pd.Timestamp:
        return self._timeline.start

    @property
    def end(self) -> pd.Timestamp:
        return self._timeline.end

    def at(self, timestamp: pd.Timestamp, /) -> T:
        return self._timeline.at(timestamp)

    def map[U](self, func: Callable[[T], U], /) -> 'GridTimeline[U]':
        return GridTimeline._from_trusted(self._timeline.map(func), self._freq)

    def filter(self, predicate: Callable[[T], bool], /, *, gap_value=pd.NA) -> Self:
        return GridTimeline._from_trusted(self._timeline.filter(predicate, gap_value=gap_value), self._freq)

    def merge_adjacent(self) -> Self:
        return GridTimeline._from_trusted(self._timeline.merge_adjacent(), self._freq)

    def slice(self, start: pd.Timestamp, end: pd.Timestamp) -> Self:
        """Extract a sub-period as a new GridTimeline.

        In addition to Timeline.slice() semantics, both bounds must lie on
        the grid; off-grid bounds raise ValueError.
        """
        for bound in (start, end):
            if bound.floor(self._freq) != bound:
                raise ValueError(f"Slice bound {bound} does not lie on the {self._freq} grid")
        return GridTimeline._from_trusted(self._timeline.slice(start, end), self._freq)

    @staticmethod
    def cross_product(grid_timelines: tuple['GridTimeline', ...], /) -> 'GridTimeline[tuple]':
        """Combine multiple GridTimelines sharing the same grid into one with composite values.

        Semantics match Timeline.cross_product: the result has one segment per
        interval in the union of all input boundaries, with a tuple of the
        input values. All inputs must share the same freq and cover the same
        total duration.

        Note:
            Vectorized via dense per-grid-slot expansion, roughly an order of
            magnitude faster than Timeline.cross_product. Memory scales with
            total duration divided by freq.
        """
        freqs = {gt._freq for gt in grid_timelines}
        if len(freqs) != 1:
            raise ValueError(f"All timelines must share the same grid freq, got {sorted(freqs)}")
        freq = freqs.pop()
        starts = {gt.start for gt in grid_timelines}
        ends = {gt.end for gt in grid_timelines}
        if len(starts) != 1 or len(ends) != 1:
            raise ValueError("All timelines must cover the same duration")
        origin = starts.pop()

        # Per timeline: value for each grid slot, and boundary positions in slot units.
        dense_values = []
        cuts = []
        for gt in grid_timelines:
            df = gt._timeline._df
            lengths = ((df['end'] - df['start']) // freq).to_numpy()
            dense_values.append(np.repeat(df['value'].to_numpy(), lengths))
            cuts.append(np.concatenate([[0], np.cumsum(lengths)]))

        union_cuts = np.unique(np.concatenate(cuts))
        times = origin + freq * union_cuts
        values = list(zip(*(dense[union_cuts[:-1]] for dense in dense_values)))
        combined = pd.DataFrame({'start': times[:-1], 'end': times[1:], 'value': values})
        return GridTimeline._from_trusted(Timeline._from_trusted(combined), freq)

    def __len__(self) -> int:
        return len(self._timeline)

    def __iter__(self):
        return iter(self._timeline)

    def __eq__(self, other) -> bool:
        if not isinstance(other, GridTimeline):
            return NotImplemented
        return self._freq == other._freq and self._timeline == other._timeline

    def __repr__(self) -> str:
        return f"GridTimeline(freq='{self._freq}', {self._timeline._df})"
