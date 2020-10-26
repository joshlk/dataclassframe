
import pandas as pd
import numpy as np
from dataclasses import fields
from typing import Optional, List, Union, Type, TypeVar, Generic, Iterable
from copy import copy, deepcopy

RecordT = TypeVar("RecordT")

def to_basic_type(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj

class _IAtIndexer(Generic[RecordT]):
    def __init__(self, dcf: "DataClassFrame"):
        self.dcf = dcf

    def __getitem__(self, key: int) -> RecordT:
        row = self.dcf.df.iloc[key]

        if isinstance(row, pd.DataFrame):
            if len(row) > 1:
                raise KeyError("key combination is not unique. To slice use `iloc` method.")
            row = row.iloc[0]

        row = {k: to_basic_type(v) for k, v in row.to_dict().items()}
        row = self.dcf.record_class(**row)
        return row

    def __setitem__(self, key: int, value: RecordT):
        row = pd.Series(value.__dict__)
        self.dcf.df.iloc[key] = row


class _AtIndexer(Generic[RecordT]):
    def __init__(self, dcf: "DataClassFrame"):
        self.dcf = dcf

    def __getitem__(self, key) -> RecordT:
        idx = pd.IndexSlice
        row = self.dcf.df.loc[idx[key], :]

        if isinstance(row, pd.DataFrame):
            if len(row) > 1:
                raise KeyError("key combination is not unique. To slice use `loc` method.")
            row = row.iloc[0]

        row = {k: to_basic_type(v) for k, v in row.to_dict().items()}
        row = self.dcf.record_class(**row)
        return row

    def __setitem__(self, key, value: RecordT):
        # TODO: need to validate with multi indexing
        index_value_in_record = value.__dict__[self.dcf.index]
        if key != index_value_in_record:
            raise ValueError(
                "key {} must equal values in the record ({})".format(key, index_value_in_record))
        row = pd.Series(value.__dict__)
        self.dcf.df.loc[key] = row


class _ColumnsWrapper(object):
    def __init__(self, dcf: "DataClassFrame"):
        cols = set(dcf.df.columns)
        super().__setattr__('__dcf', dcf)
        super().__setattr__('__cols', cols)

    def __getattribute__(self, name):
        cols = super().__getattribute__('__cols')
        dcf = super().__getattribute__('__dcf')

        if name in cols:
            return dcf.df[name]
        else:
            return super().__getattribute__(name)

    def __setattr__(self, name, value):
        cols = super().__getattribute__('__cols')
        dcf = super().__getattribute__('__dcf')

        if name in cols:
            # TODO: verify data-type isn't changed
            dcf.df[name] = value
        else:
            super().__setattr__(name, value)


class DataClassFrame(Generic[RecordT]):
    def __init__(
            self,
            record_class: Type[RecordT],
            data: Optional[pd.DataFrame] = None,
            index: Union[None, str, List[str]] = None,
    ):
        self.record_class = record_class

        if data is not None:
            self.df = data
        else:
            self.df = self.dataclass_to_empty_dataframe(record_class)

        self.index = index
        if index is not None:
            self.df = self.df.set_index(index, drop=False, verify_integrity=True)
        else:
            self.df = self.df.reset_index(drop=True)

        self.cols = _ColumnsWrapper(self)

    @classmethod
    def from_records(
            cls,
            record_class: Type[RecordT],
            data: Iterable[RecordT],
            index: Union[None, str, List[str]] = None,
    ):
        def validate_and_to_dict(i, dc):
            if not isinstance(dc, record_class):
                raise ValueError(
                    "All data must be of type {}. Found type {} at index {}".format(record_class, dc, i))
            return dc.__dict__

        df_data = [validate_and_to_dict(i, dc) for i, dc in enumerate(data)]
        if len(df_data) < 1:
            raise ValueError("Data must contain at least one record")
        df = cls.dataclass_to_empty_dataframe(record_class)
        df = df.append(df_data)

        return DataClassFrame(record_class=record_class, data=df, index=index)

    @staticmethod
    def dataclass_to_empty_dataframe(record_class: Type[RecordT]):
        df = pd.DataFrame()
        for field in fields(record_class):
            try:
                df[field.name] = pd.Series(name=field.name, dtype=field.type)
            except TypeError:
                # If `TypeError` raised by `pandas_dtype` method. Just default to None i.e. list
                df[field.name] = pd.Series(name=field.name, dtype='object')
        return df

    @staticmethod
    def validate_dataframe(record_class: Type[RecordT], df: pd.DataFrame):
        # todo: need to deal with datatype conversion
        pass

    @property
    def iat(self) -> _IAtIndexer[RecordT]:
        return _IAtIndexer(self)

    @property
    def at(self) -> _AtIndexer[RecordT]:
        return _AtIndexer(self)

    def __repr__(self) -> str:
        record_class_name = self.record_class.__name__
        header = f'DataClassFrame[{record_class_name}]\n'

        # Get df info
        df_repr = self.df.__repr__()
        return header + df_repr

    def head(self, n: int = 5) -> 'DataClassFrame[RecordT]':
        new_dcf = self.copy(deep=False)
        new_dcf.df = self.df.head(n=n)
        return new_dcf

    def copy(self, deep: bool = True) -> 'DataClassFrame[RecordT]':
        if deep:
            return deepcopy(self)
        else:
            return copy(self)

    def to_dataframe(self) -> pd.DataFrame:
        return self.df.copy(deep=True)
