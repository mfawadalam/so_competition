import pytest

from src.preprocess.converter import TrainData

raw_dict = [
    {
        "inc_angle": 42.3,
        "is_iceberg": 0,
        "band_1": [1.0,2.0,10.0],
        "band_2": [13.0,12.0,12.0]
    },
    {
        "inc_angle": 41.6,
        "is_iceberg": 1,
        "band_1": [10.0,20.0,10.0],
        "band_2": [13.5,1.2,10.6]

    },
    {
        "inc_angle": 32.4,
        "is_iceberg": 1,
        "band_1": [110.0,120.0,105.0],
        "band_2": [133.5,102.2,105.6]
    },
    {
        "inc_angle": 30.4,
        "is_iceberg": 0,
        "band_1": [7.0,20.0,5.0],
        "band_2": [3.5,2.2,5.6]
    }
]


@pytest.fixture
def train_data():
    return TrainData(raw_dict)

def test_train_data(train_data):
    x_train = train_data.X_train
    y_train = train_data.Y_train
    x_test = train_data.X_test
    y_test = train_data.Y_test
    assert x_train.shape == (3,7)
    assert y_train.shape == (3,1)
    assert x_test.shape == (1,7)
    assert y_test.shape == (1,1)