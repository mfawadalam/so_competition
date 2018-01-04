import pytest

from src.preprocess.read import JsonDataReader


@pytest.fixture
def data_reader():
    return JsonDataReader("/Users/fawadalam/Documents/Kaggle/statoil_ccore/test/resources/sample.json")

def test_data_reader(data_reader):
    data = data_reader.data
    actual = data["glossary"]["title"]
    expected = "example glossary"
    assert actual == expected


