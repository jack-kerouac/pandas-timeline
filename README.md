# pandas-timeline

## Installation

Install dependencies with:

```bash
pip install -e .
```

## Timeline

A `Timeline<T>` represents a value of type `<T>` as it changes over a continuous period. Conceptually, it behaves like a step function: the value remains constant for a segment of time and then instantly changes to a new value at the start of the next segment.

The structure consists of a series of back-to-back, non-overlapping time segments. This guarantees that for any single point in time within its total duration, there is one and only one value defined. While it can be constructed from various raw data sources, its core strength lies in its powerful and generic manipulation capabilities.

## Key Operations

The Timeline data structure provides several high-level methods for analysis and transformation:

- **Cross Product**: This is the standout feature. The cross-product operation allows you to combine two or more timelines that cover the exact same total duration. It works by overlaying them and creating a new, more granular timeline. The value in each new segment of the resulting timeline is a composite object containing the values from all the parent timelines for that specific time slice. This is incredibly useful for analyzing how different time-dependent variables interact or for calculating results based on multiple time-varying inputs (e.g., price, demand, and availability).

- **Merging & Simplification**: A Timeline can be simplified by merging any adjacent segments that happen to hold the same value. This operation cleans up the timeline by removing redundant boundaries, ensuring the most compact representation of the data.

- **Transformation & Slicing**:
  - You can map a function over a timeline to transform the value within each segment, producing a new Timeline with the same time structure but new data.
  - You can also cut or slice a timeline to extract a new, shorter timeline that represents a specific sub-period of the original.

