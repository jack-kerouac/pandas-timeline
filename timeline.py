from typing import Callable, List, Tuple, Self

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
    def from_segments(segments: List[Tuple[pd.Timestamp, pd.Timestamp, T]], /) -> 'Timeline[T]':
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

    @staticmethod
    def from_segments_with_gaps(
        segments: List[Tuple[pd.Timestamp, pd.Timestamp, T]], 
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
    def _validate(df: pd.DataFrame):
        assert len(df) > 0, "Timeline must have at least one segment"
        assert df.columns.isin(['start', 'end', 'value']).all(), "DataFrame must contain 'start', 'end', and 'value' columns"
        assert df['start'].dtype == 'datetime64[ns]', "Start must be a datetime64 column"
        assert df['end'].dtype == 'datetime64[ns]', "End must be a datetime64 column"

        assert df['start'].is_monotonic_increasing, "Segments must be sorted by start time"
        assert len(df) <= 1 or (df['end'].iloc[:-1].values == df['start'].iloc[1:].values).all(), "Segments must be contiguous"
        assert (df['start'] < df['end']).all(), "Start must be before end"

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
        return Timeline.from_dataframe(new_df)

    def slice(self, start: pd.Timestamp, end: pd.Timestamp) -> Self:
        # Extract segments overlapping with [start, end)
        mask = (self._df['end'] > start) & (self._df['start'] < end)
        sliced = self._df[mask].copy()
        if not sliced.empty:
            # Adjust the first segment's start if needed
            if sliced.iloc[0]['start'] < start:
                sliced.iat[0, sliced.columns.get_loc('start')] = start
            # Adjust the last segment's end if needed
            if sliced.iloc[-1]['end'] > end:
                sliced.iat[-1, sliced.columns.get_loc('end')] = end
        return Timeline(sliced)

    def merge_adjacent(self) -> Self:
        # Merge adjacent segments with same value
        merged = []
        for _, row in self._df.iterrows():
            if merged and merged[-1][2] == row['value']:
                merged[-1] = (merged[-1][0], row['end'], row['value'])
            else:
                merged.append((row['start'], row['end'], row['value']))
        return Timeline.from_segments(merged)

    @staticmethod
    def cross_product(timelines: Tuple['Timeline', ...], /) -> 'Timeline[Tuple]':
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

    def __repr__(self):
        return f"Timeline({self._df})"
