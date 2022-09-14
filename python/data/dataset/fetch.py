
class BaseFetcher:
    def __init__(self):
        pass

    def fetch(self, batch_indices):
        raise NotImplementedError("'{}' not implement in class " \
                                  "{}".format('fetch', self.__class__.__name__))


class MapDatasetFetcher(BaseFetcher):
    def __init__(self, dataset, collection_fn):
        super().__init__()
        self.dataset = dataset
        self.collection_fn = collection_fn

    def fetch(self, batch_indices):
        data = [self.dataset[idx] for idx in batch_indices]
        return self.collection_fn(data)


class IterDatasetFetcher(BaseFetcher):
    def __init__(self, dataset, collection_fn):
        super(IterDatasetFetcher, self).__init__()
        self.dataset_iter = iter(dataset)
        self.collection_fn = collection_fn

    def fetch(self, batch_indices):
        data = []
        for _ in batch_indices:
            data.append(next(self.dataset_iter))
        return self.collection_fn(data)
