# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**pandas-timeline** is a Python library providing a `Timeline<T>` data structure for time-series data with discrete, irregular, step-wise changes. It represents step functions where values remain constant for time segments and change instantly at segment boundaries.

## Development Commands

### Testing
```bash
# Run all tests
python -m pytest test_timeline.py -v

# Run specific test
python -m pytest test_timeline.py::test_cross_product_two_timelines_multiple_segments -v

# Run tests for specific functionality
python -m pytest test_timeline.py -k "cross_product" -v
```

### Installation
```bash
# Development installation
pip install -e .
```

### Environment Setup
```bash
# Activate virtual environment (located in .venv)
source .venv/bin/activate
```

## Core Architecture

### Timeline<T> Data Structure

The `Timeline[T]` class is built on a pandas DataFrame with strict invariants:

- **Internal DataFrame**: 3 columns (`start`, `end`, `value`) where start/end are `pd.Timestamp`
- **Immutable**: All operations return new Timeline instances
- **Validated**: Constructor enforces contiguous, non-overlapping, sorted segments
- **Generic**: Type parameter `T` represents the value type in each segment

### Key Invariants (Critical for Development)

1. **Segments are contiguous**: `segment[i].end == segment[i+1].start`
2. **No overlaps**: Segments must be back-to-back
3. **Sorted by time**: `start` times are monotonic increasing
4. **Valid segments**: `start < end` for all segments
5. **Complete coverage**: Exactly one value defined for any point in total duration

Violations throw `ValueError` or `TypeError` (not assertions) for production safety.

### Construction Patterns

- `Timeline.from_segments(segments)`: Primary constructor for contiguous segments
- `Timeline.from_segments_with_gaps(segments, gap_value=pd.NA)`: Auto-fills gaps with specified value
- `Timeline.from_dataframe(df)`: Direct DataFrame construction (validates first)

### Core Operations

#### Cross Product (Most Complex Operation)
`Timeline.cross_product(timelines)` combines multiple timelines covering identical durations:
- Uses efficient boundary-collection algorithm with segment pointers
- Creates composite timeline with tuple values
- Critical for analyzing multiple time-varying factors simultaneously

#### Other Key Operations
- `at(timestamp)`: Get value at specific timestamp using O(log n) binary search
- `slice(start, end)`: Extract sub-periods with intelligent boundary adjustment
- `merge_adjacent()`: Consolidate consecutive segments with identical values  
- `map(func)`: Transform values while preserving time structure
- Dunder methods: `__len__`, `__iter__`, `__eq__` for Python integration

## Testing Architecture

### Test Helpers (Important for New Tests)
```python
dt(time_str)  # "HH:MM" → pd.Timestamp("2023-01-01 HH:MM") 
ts(start, end, value)  # Create (start, end, value) tuple from "HH:MM" strings
```

### Test Patterns
- Comprehensive validation testing (positive and negative cases)
- Error testing uses `pytest.raises(ExceptionType, match=r"regex")`
- Each operation tested with multiple scenarios and edge cases
- Tests serve as usage documentation

## Type System & Modern Python

### Requirements
- **Python ≥ 3.11**: Required for generic class syntax and modern type features
- Uses `class Timeline[T]:` syntax and `def method[U](...) -> Timeline[U]`
- Positional-only parameters: `method(arg, /)`

### Important Type Patterns
- `Sequence[tuple[pd.Timestamp, pd.Timestamp, T]]` for segment input
- `Self` return types for method chaining
- Generic transformations: `Timeline[T] → Timeline[U]` via `map[U]()`

## Development Guidelines

### Error Handling
- Use proper exceptions (`ValueError`, `TypeError`) never `assert` statements
- Validation happens in `_validate()` static method
- Specific error messages for each validation failure

### Missing Data Handling
- Default gap value is `pd.NA` (pandas universal missing value)
- Users can specify custom gap values in `from_segments_with_gaps()`
- See pandas NA semantics: https://pandas.pydata.org/docs/user_guide/missing_data.html#na-semantics

### Code Style
- Modern type hints using Python 3.11+ syntax (`list` not `List`)
- Comprehensive docstrings with Args/Returns sections
- Immutable operations (return new instances)
- Integration with pandas best practices

## Architecture Notes for Extensions

- Internal `_df` is protected; expose via `df` property (returns copy)
- New operations should preserve Timeline invariants
- Consider vectorized pandas operations for performance
- Cross-product algorithm is the most complex - study before modifying
- All constructors should validate via `Timeline._validate()`