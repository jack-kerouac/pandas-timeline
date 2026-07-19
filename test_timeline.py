import pytest
import pandas as pd
from timeline import GridTimeline, Timeline

# Helper to create pd.Timestamp datetime from HH:MM
dt = lambda t: pd.Timestamp(f"2023-01-01 {t}")

# Helper to create tuple representing a timeline segment. Start/end should be in HH:MM format.
ts = lambda start, end, value: (dt(start), dt(end), value)

def test_creation_succeeds_with_single_segment():
    timeline = Timeline.from_segments([
        ts('00:00', '01:00', 1)
    ])
    assert isinstance(timeline, Timeline)
    assert len(timeline.df) == 1
    assert timeline.start == dt('00:00')
    assert timeline.end == dt('01:00')

def test_creation_succeeds_with_valid_segments():
    timeline = Timeline.from_segments([
        ts('00:00', '01:00', 1),
        ts('01:00', '02:00', 2),
        ts('02:00', '03:00', 3)
    ])
    assert isinstance(timeline, Timeline)
    assert len(timeline.df) == 3
    assert timeline.start == dt('00:00')
    assert timeline.end == dt('03:00')

def test_creation_fails_with_no_segment():
    with pytest.raises(ValueError, match=r".*at least one segment.*"):
        Timeline.from_segments([])

def test_creation_fails_when_segment_start_after_end():
    with pytest.raises(ValueError, match=r".*Start must be before end.*"):
        Timeline.from_segments([
            ts('02:00', '01:00', 1)
        ])

def test_creation_fails_when_segments_overlap():
    with pytest.raises(ValueError, match=r".*contiguous.*"):
        Timeline.from_segments([
            ts('00:00', '03:00', 1),
            ts('02:00', '05:00', 2)
        ])

def test_creation_fails_when_segments_have_gaps():
    with pytest.raises(ValueError, match=r".*contiguous.*"):
        Timeline.from_segments([
            ts('00:00', '02:00', 1),
            ts('03:00', '05:00', 2)
        ])

def test_creation_fails_when_segments_not_sorted():
    with pytest.raises(ValueError, match=r".*sorted.*"):
        Timeline.from_segments([
            ts('02:00', '04:00', 2),
            ts('00:00', '02:00', 1)
        ])

def test_from_segments_with_gaps_no_gaps():
    timeline = Timeline.from_segments_with_gaps([
        ts('00:00', '01:00', 1),
        ts('01:00', '02:00', 2)
    ])
    assert len(timeline.df) == 2
    assert timeline.start == dt('00:00')
    assert timeline.end == dt('02:00')
    assert list(timeline.df['value']) == [1, 2]

def test_from_segments_with_gaps_single_gap():
    timeline = Timeline.from_segments_with_gaps([
        ts('00:00', '01:00', 1),
        ts('02:00', '03:00', 2)
    ])
    df = timeline.df.reset_index(drop=True)
    assert len(df) == 3
    assert df.iloc[0]['start'] == dt('00:00')
    assert df.iloc[0]['end'] == dt('01:00')
    assert df.iloc[0]['value'] == 1
    assert df.iloc[1]['start'] == dt('01:00')
    assert df.iloc[1]['end'] == dt('02:00')
    assert pd.isna(df.iloc[1]['value'])
    assert df.iloc[2]['start'] == dt('02:00')
    assert df.iloc[2]['end'] == dt('03:00')
    assert df.iloc[2]['value'] == 2

def test_from_segments_with_gaps_multiple_gaps():
    timeline = Timeline.from_segments_with_gaps([
        ts('00:00', '01:00', 1),
        ts('02:00', '03:00', 2),
        ts('05:00', '06:00', 3)
    ])
    df = timeline.df.reset_index(drop=True)
    assert len(df) == 5
    expected = [
        (dt('00:00'), dt('01:00'), 1),
        (dt('01:00'), dt('02:00'), pd.NA),
        (dt('02:00'), dt('03:00'), 2),
        (dt('03:00'), dt('05:00'), pd.NA),
        (dt('05:00'), dt('06:00'), 3)
    ]
    for i, (start, end, value) in enumerate(expected):
        assert df.iloc[i]['start'] == start
        assert df.iloc[i]['end'] == end
        if pd.isna(value):
            assert pd.isna(df.iloc[i]['value'])
        else:
            assert df.iloc[i]['value'] == value

def test_from_segments_with_gaps_empty_list():
    with pytest.raises(ValueError, match=r".*at least one segment.*"):
        Timeline.from_segments_with_gaps([])

def test_from_segments_with_gaps_unsorted_segments():
    timeline = Timeline.from_segments_with_gaps([
        ts('02:00', '03:00', 2),
        ts('00:00', '01:00', 1),
        ts('05:00', '06:00', 3)
    ])
    df = timeline.df.reset_index(drop=True)
    assert len(df) == 5
    assert df.iloc[0]['value'] == 1
    assert pd.isna(df.iloc[1]['value'])
    assert df.iloc[2]['value'] == 2
    assert pd.isna(df.iloc[3]['value'])
    assert df.iloc[4]['value'] == 3

def test_from_segments_with_gaps_custom_gap_value():
    timeline = Timeline.from_segments_with_gaps([
        ts('00:00', '01:00', 10),
        ts('02:00', '03:00', 20)
    ], gap_value=-999)
    df = timeline.df.reset_index(drop=True)
    assert len(df) == 3
    assert df.iloc[0]['value'] == 10
    assert df.iloc[1]['value'] == -999
    assert df.iloc[2]['value'] == 20

def test_slice_shrinks_single_segment():
    timeline = Timeline.from_segments([
        ts('00:00', '01:00', 1)
    ])
    sliced = timeline.slice(dt('00:30'), dt('00:45'))
    assert len(sliced.df) == 1
    assert sliced.start == dt('00:30')
    assert sliced.end == dt('00:45')

def test_slice_shrinks_start_and_end_segment():
    timeline = Timeline.from_segments([
        ts('00:00', '01:00', 1),
        ts('01:00', '02:00', 2)
    ])
    sliced = timeline.slice(dt('00:30'), dt('01:30'))
    assert len(sliced.df) == 2
    assert sliced.start == dt('00:30')
    assert sliced.end == dt('01:30')
    assert sliced.df.iloc[0]['start'] == dt('00:30')
    assert sliced.df.iloc[0]['end'] == dt('01:00')
    assert sliced.df.iloc[1]['start'] == dt('01:00')
    assert sliced.df.iloc[1]['end'] == dt('01:30')

def test_slice_with_start_before_timeline_start_leaves_start_unchanged():
    timeline = Timeline.from_segments([
        ts('00:30', '01:00', 1),
        ts('01:00', '02:00', 2)
    ])
    sliced = timeline.slice(dt('00:15'), dt('01:30'))
    assert sliced.start == dt('00:30')

def test_slice_with_end_after_timeline_end_leaves_end_unchanged():
    timeline = Timeline.from_segments([
        ts('00:30', '01:00', 1),
        ts('01:00', '02:00', 2)
    ])
    sliced = timeline.slice(dt('01:15'), dt('02:30'))
    assert sliced.end == dt('02:00')

def test_slice_aligned_with_segment_starts_and_ends():
    timeline = Timeline.from_segments([
        ts('00:00', '01:00', 1),
        ts('01:00', '02:00', 2),
        ts('02:00', '03:00', 3),
        ts('03:00', '04:00', 4),
    ])
    sliced = timeline.slice(dt('01:00'), dt('03:00'))
    assert len(sliced.df) == 2
    assert sliced.start == dt('01:00')
    assert sliced.end == dt('03:00')

def test_slice_with_start_not_before_end_raises():
    timeline = Timeline.from_segments([
        ts('00:00', '01:00', 1)
    ])
    with pytest.raises(ValueError, match=r".*start must be before end.*"):
        timeline.slice(dt('00:45'), dt('00:30'))
    with pytest.raises(ValueError, match=r".*start must be before end.*"):
        timeline.slice(dt('00:30'), dt('00:30'))

def test_slice_outside_timeline_duration_raises():
    timeline = Timeline.from_segments([
        ts('01:00', '02:00', 1)
    ])
    with pytest.raises(ValueError, match=r".*does not overlap timeline duration.*"):
        timeline.slice(dt('00:00'), dt('01:00'))
    with pytest.raises(ValueError, match=r".*does not overlap timeline duration.*"):
        timeline.slice(dt('02:00'), dt('03:00'))

def test_merge_adjacent_segments():
    timeline = Timeline.from_segments([
        ts('00:00', '01:00', 1),
        ts('01:00', '02:00', 1),
        ts('02:00', '03:00', 2),
        ts('03:00', '04:00', 3),
        ts('04:00', '05:00', 3)
    ])
    merged = timeline.merge_adjacent()
    assert len(merged.df) == 3
    assert merged.df.iloc[0]['start'] == dt('00:00')
    assert merged.df.iloc[0]['end'] == dt('02:00')
    assert merged.df.iloc[0]['value'] == 1
    assert merged.df.iloc[1]['start'] == dt('02:00')
    assert merged.df.iloc[1]['end'] == dt('03:00')
    assert merged.df.iloc[1]['value'] == 2
    assert merged.df.iloc[2]['start'] == dt('03:00')
    assert merged.df.iloc[2]['end'] == dt('05:00')
    assert merged.df.iloc[2]['value'] == 3

def test_merge_adjacent_merges_na_segments():
    timeline = Timeline.from_segments([
        ts('00:00', '01:00', 'a'),
        ts('01:00', '02:00', pd.NA),
        ts('02:00', '03:00', pd.NA),
        ts('03:00', '04:00', 'b'),
    ])
    merged = timeline.merge_adjacent()
    assert len(merged.df) == 3
    assert merged.df.iloc[1]['start'] == dt('01:00')
    assert merged.df.iloc[1]['end'] == dt('03:00')
    assert merged.df.iloc[1]['value'] is pd.NA

def test_merge_adjacent_starting_with_na():
    timeline = Timeline.from_segments([
        ts('00:00', '01:00', pd.NA),
        ts('01:00', '02:00', pd.NA),
        ts('02:00', '03:00', 1),
    ])
    merged = timeline.merge_adjacent()
    assert len(merged.df) == 2
    assert merged.df.iloc[0]['start'] == dt('00:00')
    assert merged.df.iloc[0]['end'] == dt('02:00')
    assert merged.df.iloc[0]['value'] is pd.NA
    assert merged.df.iloc[1]['value'] == 1

def test_merge_adjacent_does_not_merge_na_with_value():
    timeline = Timeline.from_segments([
        ts('00:00', '01:00', pd.NA),
        ts('01:00', '02:00', 1),
        ts('02:00', '03:00', pd.NA),
    ])
    merged = timeline.merge_adjacent()
    assert len(merged.df) == 3

def test_merge_adjacent_with_tuple_values():
    timeline = Timeline.from_segments([
        ts('00:00', '01:00', (1, 'x')),
        ts('01:00', '02:00', (1, 'x')),
        ts('02:00', '03:00', (2, 'y')),
    ])
    merged = timeline.merge_adjacent()
    assert len(merged.df) == 2
    assert merged.df.iloc[0]['end'] == dt('02:00')
    assert merged.df.iloc[0]['value'] == (1, 'x')
    assert merged.df.iloc[1]['value'] == (2, 'y')

def test_map_applies_function_to_all_segments():
    timeline = Timeline.from_segments([
        ts('00:00', '01:00', 1),
        ts('01:00', '02:00', 2),
        ts('02:00', '03:00', 3)
    ])
    mapped = timeline.map(lambda x: x * 10)
    expected = [10, 20, 30]
    assert list(mapped.df['value']) == expected
    # Start/end unchanged
    assert list(mapped.df['start']) == [dt('00:00'), dt('01:00'), dt('02:00')]
    assert list(mapped.df['end']) == [dt('01:00'), dt('02:00'), dt('03:00')]

def test_map_can_change_type():
    timeline = Timeline.from_segments([
        ts('00:00', '01:00', 1),
        ts('01:00', '02:00', 2)
    ])
    mapped = timeline.map(str)
    assert list(mapped.df['value']) == ['1', '2']
    assert mapped.df['value'].dtype == object

def test_map_identity_returns_same_values():
    timeline = Timeline.from_segments([
        ts('00:00', '01:00', 42)
    ])
    mapped = timeline.map(lambda x: x)
    assert list(mapped.df['value']) == [42]
    assert mapped.df['start'].iloc[0] == dt('00:00')
    assert mapped.df['end'].iloc[0] == dt('01:00')


def test_at_single_segment():
    timeline = Timeline.from_segments([
        ts('00:00', '01:00', 42)
    ])
    assert timeline.at(dt('00:00')) == 42
    assert timeline.at(dt('00:30')) == 42
    assert timeline.at(dt('00:59')) == 42

def test_at_multiple_segments():
    timeline = Timeline.from_segments([
        ts('00:00', '01:00', 1),
        ts('01:00', '02:00', 2),
        ts('02:00', '03:00', 3)
    ])
    assert timeline.at(dt('00:00')) == 1
    assert timeline.at(dt('00:30')) == 1
    assert timeline.at(dt('00:59')) == 1
    assert timeline.at(dt('01:00')) == 2
    assert timeline.at(dt('01:30')) == 2
    assert timeline.at(dt('01:59')) == 2
    assert timeline.at(dt('02:00')) == 3
    assert timeline.at(dt('02:30')) == 3
    assert timeline.at(dt('02:59')) == 3

def test_at_exact_segment_boundaries():
    timeline = Timeline.from_segments([
        ts('00:00', '01:00', 'A'),
        ts('01:00', '02:00', 'B'),
        ts('02:00', '03:00', 'C')
    ])
    # At exact start times, should return the value of the segment starting at that time
    assert timeline.at(dt('00:00')) == 'A'
    assert timeline.at(dt('01:00')) == 'B'
    assert timeline.at(dt('02:00')) == 'C'

def test_at_with_gaps():
    # Test with timeline that has gaps filled with pd.NA
    timeline = Timeline.from_segments_with_gaps([
        ts('00:00', '01:00', 1),
        ts('02:00', '03:00', 2)
    ])
    assert timeline.at(dt('00:30')) == 1
    assert pd.isna(timeline.at(dt('01:30')))
    assert timeline.at(dt('02:30')) == 2

def test_at_timestamp_before_timeline_start():
    timeline = Timeline.from_segments([ts('01:00', '02:00', 1)])
    with pytest.raises(ValueError, match=r"Timestamp .* is outside timeline duration"):
        timeline.at(dt('00:30'))

def test_at_timestamp_at_timeline_end():
    timeline = Timeline.from_segments([ts('00:00', '01:00', 1)])
    # At the exact end time should raise error (end is exclusive)
    with pytest.raises(ValueError, match=r"Timestamp .* is outside timeline duration"):
        timeline.at(dt('01:00'))

def test_at_timestamp_after_timeline_end():
    timeline = Timeline.from_segments([ts('00:00', '01:00', 1)])
    with pytest.raises(ValueError, match=r"Timestamp .* is outside timeline duration"):
        timeline.at(dt('02:00'))

def test_at_with_complex_timeline():
    # Test with a more complex timeline including gaps
    timeline = Timeline.from_segments_with_gaps([
        ts('00:00', '01:00', 'morning'),
        ts('03:00', '04:00', 'afternoon'),
        ts('06:00', '07:00', 'evening')
    ], gap_value='free_time')
    
    assert timeline.at(dt('00:30')) == 'morning'
    assert timeline.at(dt('01:30')) == 'free_time'
    assert timeline.at(dt('03:30')) == 'afternoon'
    assert timeline.at(dt('04:30')) == 'free_time'
    assert timeline.at(dt('06:30')) == 'evening'


def test_cross_product_fails_without_common_start_date():
    t1 = Timeline.from_segments([ts('00:00', '05:00', 1)])
    t2 = Timeline.from_segments([ts('01:00', '05:00', 2)])
    with pytest.raises(AssertionError) as e:
        Timeline.cross_product((t1, t2))
    assert 'start' in str(e.value).lower()

def test_cross_product_fails_without_common_end_date():
    t1 = Timeline.from_segments([ts('00:00', '04:00', 1)])
    t2 = Timeline.from_segments([ts('00:00', '05:00', 2)])
    with pytest.raises(AssertionError) as e:
        Timeline.cross_product((t1, t2))
    assert 'end' in str(e.value).lower()

def test_cross_product_single_timeline_single_segment():
    t1 = Timeline.from_segments([ts('00:00', '05:00', 1)])
    cp = Timeline.cross_product((t1,))
    df = cp.df
    assert len(df) == 1
    assert df.iloc[0]['start'] == dt('00:00')
    assert df.iloc[0]['end'] == dt('05:00')
    assert df.iloc[0]['value'] == (1,)

def test_cross_product_two_timelines_single_segment():
    t1 = Timeline.from_segments([ts('00:00', '05:00', 1)])
    t2 = Timeline.from_segments([ts('00:00', '05:00', 2)])
    cp = Timeline.cross_product((t1, t2))
    df = cp.df
    assert len(df) == 1
    assert df.iloc[0]['start'] == dt('00:00')
    assert df.iloc[0]['end'] == dt('05:00')
    assert df.iloc[0]['value'] == (1, 2)

def test_cross_product_two_timelines_multiple_segments():
    t1 = Timeline.from_segments([ts('00:00', '05:00', 1), ts('05:00', '10:00', 2)])
    t2 = Timeline.from_segments([ts('00:00', '03:00', 3), ts('03:00', '07:00', 4), ts('07:00', '10:00', 5)])
    cp = Timeline.cross_product((t1, t2))
    df = cp.df.reset_index(drop=True)
    expected = [
        (dt('00:00'), dt('03:00'), (1, 3)),
        (dt('03:00'), dt('05:00'), (1, 4)),
        (dt('05:00'), dt('07:00'), (2, 4)),
        (dt('07:00'), dt('10:00'), (2, 5)),
    ]
    assert len(df) == len(expected)
    for i, (start, end, value) in enumerate(expected):
        assert df.iloc[i]['start'] == start
        assert df.iloc[i]['end'] == end
        assert df.iloc[i]['value'] == value

def test_cross_product_three_timelines_two_segments_each():
    t1 = Timeline.from_segments([ts('00:00', '05:00', 1), ts('05:00', '10:00', 2)])
    t2 = Timeline.from_segments([ts('00:00', '05:00', 3), ts('05:00', '10:00', 4)])
    t3 = Timeline.from_segments([ts('00:00', '04:00', 5), ts('04:00', '10:00', 6)])
    cp = Timeline.cross_product((t1, t2, t3))
    df = cp.df.reset_index(drop=True)
    expected = [
        (dt('00:00'), dt('04:00'), (1, 3, 5)),
        (dt('04:00'), dt('05:00'), (1, 3, 6)),
        (dt('05:00'), dt('10:00'), (2, 4, 6)),
    ]
    assert len(df) == len(expected)
    for i, (start, end, value) in enumerate(expected):
        assert df.iloc[i]['start'] == start
        assert df.iloc[i]['end'] == end
        assert df.iloc[i]['value'] == value


def test_filter_keeps_matching_segments():
    timeline = Timeline.from_segments([
        ts('00:00', '01:00', 1),
        ts('01:00', '02:00', 2),
        ts('02:00', '03:00', 3),
    ])
    result = timeline.filter(lambda v: v > 1)
    df = result.df
    assert len(df) == 3
    assert pd.isna(df.iloc[0]['value'])
    assert df.iloc[1]['value'] == 2
    assert df.iloc[2]['value'] == 3


def test_filter_all_match():
    timeline = Timeline.from_segments([
        ts('00:00', '01:00', 10),
        ts('01:00', '02:00', 20),
    ])
    result = timeline.filter(lambda v: v > 0)
    df = result.df
    assert len(df) == 2
    assert df.iloc[0]['value'] == 10
    assert df.iloc[1]['value'] == 20


def test_filter_none_match():
    timeline = Timeline.from_segments([
        ts('00:00', '01:00', 1),
        ts('01:00', '02:00', 2),
    ])
    result = timeline.filter(lambda v: v > 100)
    df = result.df
    assert len(df) == 2
    assert pd.isna(df.iloc[0]['value'])
    assert pd.isna(df.iloc[1]['value'])


def test_filter_with_custom_gap_value():
    timeline = Timeline.from_segments([
        ts('00:00', '01:00', 'a'),
        ts('01:00', '02:00', 'b'),
        ts('02:00', '03:00', 'a'),
    ])
    result = timeline.filter(lambda v: v == 'a', gap_value='FILTERED')
    df = result.df
    assert df.iloc[0]['value'] == 'a'
    assert df.iloc[1]['value'] == 'FILTERED'
    assert df.iloc[2]['value'] == 'a'


def test_filter_preserves_int_values_exactly():
    timeline = Timeline.from_segments([
        ts('00:00', '01:00', 3),
        ts('01:00', '02:00', 4),
    ])
    result = timeline.filter(lambda v: v == 3)
    df = result.df
    surviving = df.iloc[0]['value']
    assert surviving == 3
    assert type(surviving) is int
    assert str(surviving) == '3'
    assert df.iloc[1]['value'] is pd.NA


def test_filter_preserves_time_structure():
    timeline = Timeline.from_segments([
        ts('09:00', '10:00', 5),
        ts('10:00', '11:00', 15),
    ])
    result = timeline.filter(lambda v: v > 10)
    df = result.df
    assert df.iloc[0]['start'] == dt('09:00')
    assert df.iloc[0]['end'] == dt('10:00')
    assert df.iloc[1]['start'] == dt('10:00')
    assert df.iloc[1]['end'] == dt('11:00')

# --- GridTimeline ---

def grid_timeline():
    return GridTimeline.from_segments([
        ts('00:00', '00:15', 1),
        ts('00:15', '01:00', 2),
        ts('01:00', '02:30', 3),
    ], freq='15min')

def test_grid_timeline_wraps_aligned_timeline():
    timeline = Timeline.from_segments([
        ts('00:00', '00:15', 1),
        ts('00:15', '01:00', 2),
    ])
    grid = GridTimeline(timeline, '15min')
    assert grid.freq == pd.Timedelta('15min')
    assert grid.timeline == timeline
    assert grid.start == dt('00:00')
    assert grid.end == dt('01:00')
    assert len(grid) == 2
    assert grid.at(dt('00:20')) == 2
    assert list(grid) == [ts('00:00', '00:15', 1), ts('00:15', '01:00', 2)]

def test_grid_timeline_from_segments():
    grid = grid_timeline()
    assert isinstance(grid, GridTimeline)
    assert len(grid) == 3

def test_grid_timeline_rejects_misaligned_segment_boundary():
    timeline = Timeline.from_segments([
        ts('00:00', '00:03', 1),
        ts('00:03', '00:15', 2),
    ])
    with pytest.raises(ValueError, match=r".*00:03.*does not lie on the.*grid"):
        GridTimeline(timeline, '15min')

def test_grid_timeline_rejects_misaligned_final_end():
    timeline = Timeline.from_segments([
        ts('00:00', '00:20', 1),
    ])
    with pytest.raises(ValueError, match=r".*00:20.*does not lie on the.*grid"):
        GridTimeline(timeline, '15min')

def test_grid_timeline_rejects_non_positive_freq():
    timeline = Timeline.from_segments([ts('00:00', '00:15', 1)])
    with pytest.raises(ValueError, match=r".*positive.*"):
        GridTimeline(timeline, '0min')

def test_grid_timeline_map_returns_grid_timeline():
    mapped = grid_timeline().map(lambda v: v * 10)
    assert isinstance(mapped, GridTimeline)
    assert mapped.freq == pd.Timedelta('15min')
    assert mapped.at(dt('00:00')) == 10

def test_grid_timeline_filter_returns_grid_timeline():
    filtered = grid_timeline().filter(lambda v: v == 2)
    assert isinstance(filtered, GridTimeline)
    assert filtered.at(dt('00:00')) is pd.NA
    assert filtered.at(dt('00:15')) == 2

def test_grid_timeline_merge_adjacent_returns_grid_timeline():
    merged = grid_timeline().map(lambda v: 'x').merge_adjacent()
    assert isinstance(merged, GridTimeline)
    assert len(merged) == 1

def test_grid_timeline_slice_with_aligned_bounds():
    sliced = grid_timeline().slice(dt('00:15'), dt('01:30'))
    assert isinstance(sliced, GridTimeline)
    assert sliced.freq == pd.Timedelta('15min')
    assert sliced.start == dt('00:15')
    assert sliced.end == dt('01:30')

def test_grid_timeline_slice_rejects_off_grid_bounds():
    with pytest.raises(ValueError, match=r".*00:03.*does not lie on the.*grid"):
        grid_timeline().slice(dt('00:03'), dt('01:00'))
    with pytest.raises(ValueError, match=r".*01:20.*does not lie on the.*grid"):
        grid_timeline().slice(dt('00:15'), dt('01:20'))

def test_grid_timeline_cross_product():
    a = GridTimeline.from_segments([
        ts('00:00', '01:00', 1),
        ts('01:00', '02:00', 2),
    ], freq='15min')
    b = GridTimeline.from_segments([
        ts('00:00', '00:30', 'x'),
        ts('00:30', '02:00', 'y'),
    ], freq='15min')
    combined = GridTimeline.cross_product((a, b))
    assert isinstance(combined, GridTimeline)
    assert combined.freq == pd.Timedelta('15min')
    assert combined.at(dt('00:45')) == (1, 'y')
    assert len(combined) == 3

def test_grid_timeline_cross_product_rejects_mixed_freqs():
    a = GridTimeline.from_segments([ts('00:00', '01:00', 1)], freq='15min')
    b = GridTimeline.from_segments([ts('00:00', '01:00', 2)], freq='30min')
    with pytest.raises(ValueError, match=r".*same grid.*"):
        GridTimeline.cross_product((a, b))

def test_grid_timeline_equality():
    assert grid_timeline() == grid_timeline()
    other_freq = GridTimeline(grid_timeline().timeline, '5min')
    assert grid_timeline() != other_freq
    assert grid_timeline() != grid_timeline().timeline
