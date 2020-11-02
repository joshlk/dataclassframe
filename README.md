[![PyPI](https://img.shields.io/pypi/v/dataclassframe)](https://pypi.org/project/dataclassframe/)
![Python](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue)
[![Build Status](https://travis-ci.com/joshlk/dataclassframe.svg?branch=main)](https://travis-ci.com/joshlk/dataclassframe)
[![Documentation](https://readthedocs.org/projects/pip/badge/?version=latest&style=flat)](https://joshlk.github.io/dataclassframe)

# dataclassframe

A dataclass container with multi-indexing and bulk operations.
Provides the typed benefits and ergonomics of dataclasses while having the efficiency of [Pandas dataframes](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html).

The container is based on [data-oriented design][1] by optimising the memory layout of the stored data, providing fast
bulk operations and a smaller memory footprint for large collections.
Bulk operations are enabled using Pandas which has a rich set of vectorised methods for both numerical and string
data types.

Multi-indexing provides the ability to use multiple fields as keys to index the records.
This is suitable for bidirectional and inverse dictionary keys.

A DataClassFrame provides good ergonomics for production code as columns are immutable
and columns/data types are well defined by the dataclasses.
This makes it easier for users to understand the "shape" of the data in large projects and refactor when necessary.

## Installing

Get the latest version using pip/PyPi

```shell
pip install dataclassframe
```

## Feature comparison

| Container                                       | Positional indexing | Key indexing | Multi-key indexing | Data-oriented design | Column-wise opperations | Type hints | Use in prod |
|-------------------------------------------------|---------------------|--------------|--------------------|----------------------|-------------------------|------------|-------------|
| DataClassFrame                                  | ✅                   | ✅            | ✅                  | ✅                    | ✅                       | ✅          | ✅           |
| List                                            | ✅                   | ❌            | ❌                  | ❌                    | ❌                       | ✅          | ✅           |
| Dictionary                                      | ❌                   | ✅            | ❌                  | ❌                    | ❌                       | ✅          | ✅           |
| [MIDict](https://github.com/ShenggaoZhu/midict) | ✅                   | ✅            | ✅                  | ❌                    | ❌                       | ✅          | ✅           |
| [Pandas DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html)                               | ✅                   | ✅            | ✅                  | ✅                    | ✅                       | ❌          | ❌           |

## Show by example

A container data-type for dataclasses...
```python
from dataclasses import dataclass
from dataclassframe import DataClassFrame

@dataclass
class ExampleDC:
    field1: str
    field2: int

records = [
    ExampleDC('a', 1),
    ExampleDC('b', 2),
    ExampleDC('c', 3),
]

dcf = DataClassFrame(
        record_class=ExampleDC,
        data=records,
        index=['field1', 'field2']
)
```

Which acts like a ordered dictionary with multi-indexing...
```python
# Obtain record `ExampleDC('b', 2)`
row_idx = dcf.iat[1]    # Using positional index
row_f1 = dcf.at['b']    # Using index of `field1`
row_f2 = dcf.at[:, 2]   # Using index of `field2`
assert row_idx == row_f1 == row_f2
```

With bulk operations on the columns..
```python
assert dcf.cols.field2.sum() == 6
```

Works nicely with Python 3 type hints...
```python
dcf: DataClassFrame[ExampleDC]
dcf.iat[1]: ExampleDC
```

## Design

It's no secret that under the hood DataClassFrames are using Pandas DataFrames to store data.
The data is converted where possible to Pandas Series, which in turn use Numpy arrays. When the user accesses a record the data is then converted back into the dataclass provided at initialisation.

Pandas provides many advantages over of using a simple list of dataclasses or similar such as better memory
footprint and fast vectorised operations. However using Pandas DataFrames directly in production code is [considered by the author and others as an anti-pattern][2].
Specifically as DataFrames are column-wise mutable and therefore difficult to determine at code-time what columns
the dataframe contains i.e. its shape. It also does not provide any type-hinting benefits.

[1]: https://jamesmcm.github.io/blog/2020/07/25/intro-dod/
[2]: https://devanla.com/posts/do-not-create-that-dataframe.html

## Todo

- [ ] Slicing and dataclassframe views for accessing data and setting data
- [ ] Append and inserts
- [ ] Data-oriented design for Numpy fields

## Changelog

All notable changes to this project will be documented here.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

### [0.1.0] - 2020-10-22
#### Added
- Initial release of `dataclassframe`


## License

© Josh Levy-Kramer 2020. dataclassframe is released under the MIT license.