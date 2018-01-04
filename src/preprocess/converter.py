
import numpy as np
from sklearn.model_selection import train_test_split


class TrainData:
    def __init__(self, data):
        flattened = self._flatten_data(data)
        train, test = self._randomize_train_test(flattened, 0.2)
        self._Y_train = np.reshape(np.array([record.pop() for record in train],dtype=np.float64),(len(train),1))
        self._X_train = np.array(train,dtype=np.float64)
        self._Y_test = np.reshape(np.array([record.pop() for record in test],dtype=np.float64),(len(test),1))
        self._X_test = np.array(test,dtype=np.float64)

    def _flatten_data(self, data):
        flattened = []
        for example in data:
            flat_list=[]
            feature_list = [[example["inc_angle"]], example["band_1"], example["band_2"], [example["is_iceberg"]]]
            flat_list = [float(item) if not isinstance(item, basestring) else 0.0 for sub_list in feature_list for item in sub_list]
            flattened.append(flat_list)
        return flattened

    def _randomize_train_test(self, data, test_proportion):
        train ,test = train_test_split(data,test_size=test_proportion)
        return train, test

    @property
    def Y_train(self):
        return self._Y_train

    @property
    def Y_test(self):
        return self._Y_test

    @property
    def X_train(self):
        return self._X_train

    @property
    def X_test(self):
        return self._X_test