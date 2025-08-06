import pytest
import pandas as pd
from timeline import Timeline

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
    with pytest.raises(Exception) as e:
        Timeline.from_segments([])
    assert 'at least one segment' in str(e.value).lower()

def test_creation_fails_when_segment_start_after_end():
    with pytest.raises(Exception) as e:
        Timeline.from_segments([
            ts('02:00', '01:00', 1)
        ])
    assert 'start must be before end' in str(e.value).lower()

def test_creation_fails_when_segments_overlap():
    with pytest.raises(Exception) as e:
        Timeline.from_segments([
            ts('00:00', '03:00', 1),
            ts('02:00', '05:00', 2)
        ])
    assert 'contiguous' in str(e.value).lower()

def test_creation_fails_when_segments_have_gaps():
    with pytest.raises(Exception) as e:
        Timeline.from_segments([
            ts('00:00', '02:00', 1),
            ts('03:00', '05:00', 2)
        ])
    assert 'contiguous' in str(e.value).lower()

def test_creation_fails_when_segments_not_sorted():
    with pytest.raises(Exception) as e:
        Timeline.from_segments([
            ts('02:00', '04:00', 2),
            ts('00:00', '02:00', 1)
        ])
    assert 'sorted' in str(e.value).lower()

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
