import io
from typing import Optional

from dataclassframe import DataClassFrame
from dataclasses import dataclass
import pandas as pd
import numpy as np
import pytest


@dataclass
class DataClassExample1:
    a: int
    b: str


def test_dataclass_str():
    # no index
    data = pd.DataFrame([[1, "a"], [2, "b"], [3, "c"]], columns=["a", "b"])
    dcf = DataClassFrame.from_dataframe(record_class=DataClassExample1, dataframe=data)
    print(dcf)

    # single index
    data = pd.DataFrame([[1, "a"], [2, "b"], [3, "c"]], columns=["a", "b"])
    dcf = DataClassFrame.from_dataframe(record_class=DataClassExample1, dataframe=data, index="b")
    print(dcf)

    # multi-index
    data = pd.DataFrame([[1, "a"], [2, "b"], [3, "c"]], columns=["a", "b"])
    dcf = DataClassFrame.from_dataframe(record_class=DataClassExample1, dataframe=data, index=["a", "b"])
    print(dcf)


def test_iat_indexing():
    data = pd.DataFrame([[1, "a"], [2, "b"], [3, "c"]], columns=["a", "b"])
    dcf = DataClassFrame.from_dataframe(record_class=DataClassExample1, dataframe=data, index="b")

    row = dcf.iat[0]
    assert isinstance(row, DataClassExample1)
    assert isinstance(row.a, int) and isinstance(row.b, str)
    assert row.a == 1 and row.b == "a"

    data_row = DataClassExample1(a=11, b="ZZ")
    dcf.iat[1] = data_row
    row = dcf.iat[1]
    assert row == data_row


def test_at_indexing():
    data = pd.DataFrame([[1, "a"], [2, "b"], [3, "c"]], columns=["a", "b"])
    dcf: DataClassFrame[DataClassExample1] = DataClassFrame.from_dataframe(
        record_class=DataClassExample1, dataframe=data, index="b")

    row = dcf.at['b']
    assert isinstance(row, DataClassExample1)
    assert isinstance(row.a, int) and isinstance(row.b, str)
    assert row.a == 2 and row.b == "b"

    data_row = DataClassExample1(a=11, b="ZZ")
    dcf.at['ZZ'] = data_row
    row = dcf.at['ZZ']
    assert row == data_row

    with pytest.raises(ValueError):
        # Index key is different to value in record
        dcf.at['A'] = DataClassExample1(a=0, b="B")


@dataclass
class ExampleDC:
    a: str
    b: int


def test_at_multiindexing_basic():
    records = [
        ExampleDC('a', 1),
        ExampleDC('b', 2),
        ExampleDC('c', 3),
    ]

    dcf = DataClassFrame(
        ExampleDC,
        records,
        index=['a', 'b'])

    a = dcf.iat[1]
    b = dcf.at['b']
    c = dcf.at['b', :]
    d = dcf.at[:, 2]
    assert a == b == c == d

def test_at_multiindexing_setting_value():
    records = [
        ExampleDC('a', 1),
        ExampleDC('b', 2),
        ExampleDC('c', 3),
    ]

    dcf = DataClassFrame(
        ExampleDC,
        records,
        index=['a', 'b'])

    new_rec = ExampleDC('d', 5)
    dcf.at['d', 5] = new_rec
    assert new_rec == dcf.at['d', 5]

    with pytest.raises(ValueError):
        # Key data miss-match
        dcf.at['e', 6] = ExampleDC('d', 5)

def test_from_records():
    data = [
        DataClassExample1(1, 'a'),
        DataClassExample1(2, 'b'),
        DataClassExample1(3, 'c'),
    ]

    dcf = DataClassFrame(record_class=DataClassExample1, data=data, index='b')

    rec = dcf.at['c']
    assert rec == DataClassExample1(3, 'c')

    with pytest.raises(ValueError):
        data = [
            DataClassExample1(1, 'a'),
            ExampleDC(0, 0)
        ]
        DataClassFrame(record_class=DataClassExample1, data=data, index='b')


def test_columns():
    data = pd.DataFrame([[1, "a"], [2, "b"], [3, "c"]], columns=["a", "b"])
    dcf = DataClassFrame.from_dataframe(record_class=DataClassExample1, dataframe=data, index="b")

    # Get sum of all columns
    assert 6 == dcf.cols.a.sum()

    # Set and then sum all columns
    dcf.cols.a = 1
    assert 3 == dcf.cols.a.sum()

    # String concatenation
    assert dcf.cols.b.sum() == 'abc'



@dataclass
class MIExample:
    A: str
    B: str
    C: str
    U: str
    bar: int
    foo: int
    bah: int
    foh: int


def test_multiindex():
    data = """
A	B	C	U	bah	bar	foh	foo
A0	B0	C0	U0	2	0	3	1
A0	B0	C1	U1	6	4	7	5
A0	B0	C2	U2	10	8	11	9
A0	B0	C3	U3	14	12	15	13
A0	B1	C0	U4	18	16	19	17
A0	B1	C1	U5	22	20	23	21
A0	B1	C2	U6	26	24	27	25
A0	B1	C3	U7	30	28	31	29
A1	B0	C0	U8	34	32	35	33
A1	B0	C1	U9	38	36	39	37
A1	B0	C2	U10	42	40	43	41
A1	B0	C3	U11	46	44	47	45
A1	B1	C0	U12	50	48	51	49
A1	B1	C1	U13	54	52	55	53
A1	B1	C2	U14	58	56	59	57
A1	B1	C3	U15	62	60	63	61
    """

    data = pd.read_csv(io.StringIO(data), sep='\t')
    dcf = DataClassFrame.from_dataframe(MIExample, data, index=['A', 'B', 'C', 'U'])

    row_0 = dcf.iat[0]
    row_U0 = dcf.at[:, :, :, 'U0']  # Index U0 provides a unique result
    assert row_0 == row_U0

    row_ABC0 = dcf.at['A0', 'B0', 'C0', :]  # A, B, C Combination provides a unique result
    assert row_0 == row_ABC0

    # Not a unique key combination
    with pytest.raises(KeyError):
        dcf.at['A0']

    with pytest.raises(KeyError):
        dcf.at['A0', 'B0', :, :]


@dataclass
class DataClassTestTypes:
    a: np.ndarray
    b: pd.Series
    c: pd.DataFrame
    d: list
    e: dict
    f: DataClassExample1

def test_non_basic_types():

    data = [
        DataClassTestTypes(
            a = np.ones((10,)),
            b = pd.Series(np.arange(10)),
            c = pd.DataFrame(np.zeros((3,3))),
            d = [1,2,3],
            e = {'a': 1, 'b': 2},
            f = DataClassExample1(1, 'a')
        )
    ]

    dcf = DataClassFrame(record_class=DataClassTestTypes, data=data)
    rec = dcf.iat[0]

    assert isinstance(rec.a, np.ndarray)
    assert np.array_equal(rec.a, np.ones((10,)))

    assert isinstance(rec.b, pd.Series)
    assert rec.b.equals(pd.Series(np.arange(10)))

    assert isinstance(rec.c, pd.DataFrame)
    assert rec.c.equals(pd.DataFrame(np.zeros((3,3))))

    assert isinstance(rec.d, list)
    assert rec.d == [1,2,3]

    assert isinstance(rec.e, dict)
    assert rec.e == {'a': 1, 'b': 2}

    assert isinstance(rec.f, DataClassExample1)
    assert rec.f == DataClassExample1(1, 'a')

@dataclass
class DataClassNoneValues:
    string: Optional[str]
    integer: Optional[int]
    floating: Optional[float]
    boolean: Optional[bool]


def test_none_values():
    data = [
        DataClassNoneValues(
            string=None,
            integer=None,
            floating=None,
            boolean=None,
        )
    ]

    dcf = DataClassFrame(record_class=DataClassNoneValues, data=data)
    rec = dcf.iat[0]

    assert rec.string is None
    assert rec.integer is None
    assert rec.floating is None
    assert rec.boolean is None

