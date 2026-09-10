from lilypond import Basin

class BasinWithTrainingData:

    def __init__(self, dataset_name, basin:Basin, X_train):
        self.dataset_name = dataset_name
        self.basin = basin
        self.X_train = X_train

    def export(self, export_dir):
        import pickle, os
        os.makedirs(export_dir, exist_ok=True)
        pickle.dump(self, open(f"{export_dir}/dBasin_{self.dataset_name}.pkl", "wb"))
        return self
