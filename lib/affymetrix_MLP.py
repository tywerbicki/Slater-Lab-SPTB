# Paper guiding architecture decisions.
# https://www.frontiersin.org/articles/10.3389/fgene.2020.00402/full
# https://www.sciencedirect.com/science/article/pii/S0933365717305067

import os
import shutil
from functools import partial
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim import optimizer
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter

class _affy_data(Dataset):
    """
    Helper class to format data for torch data loader.
    """
    def __init__(self, X : pd.DataFrame, y : pd.Series):
        scaler = StandardScaler()
        X = scaler.fit_transform(X.to_numpy())
        self.__scaler = scaler
        self.__X = torch.from_numpy(X).type(torch.FloatTensor)
        self.__y = torch.from_numpy(y.to_numpy()).type(torch.FloatTensor)

    def __len__(self):
        return len(self.__y)

    def __getitem__(self, idx):
        return self.__X[idx], self.__y[idx]

    def pass_scaler(self) -> StandardScaler:
        scaler = self.__scaler
        self.__scaler = None
        return scaler

class _mlp_bc_network(nn.Module):
    """
    Helper class to hold neural network + a few methods.
    """
    def __init__(self, n_features, l1, l2, p1, p2):
        super(_mlp_bc_network, self).__init__()

        self.__layer_stack = nn.Sequential(
            nn.Linear(n_features, l1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(l1),
            nn.Dropout(p = p1),
            nn.Linear(l1, l2),
            nn.LeakyReLU(),
            nn.Dropout(p = p2),
            nn.Linear(l2, 1)
        )

    def forward(self, X : torch.FloatTensor) -> torch.FloatTensor:
        """
        Returns raw network output. This then needs to be transformed via
        the sigmoid function to enforce it exists in the interval (0, 1).
        """
        return self.__layer_stack(X).squeeze()

    def val_forward(self, X : torch.FloatTensor) -> torch.FloatTensor:
        """
        Perform forward pass on validation / test 2-D Torch FloatTensor.
        """
        self.eval()
        with torch.no_grad():
            return self(X)

    def predict(self, X : torch.FloatTensor, raw_outputs : torch.FloatTensor = None) -> np.ndarray:
        """
        Make predictons on a 2-D Torch FloatTensor.
        """
        if raw_outputs == None:
            raw_outputs = self.val_forward(X)

        preds = torch.sigmoid(raw_outputs).round()
        return preds.numpy().astype("int32")


class MLP_Binary_Classifier:

    def __init__(self):
        self.__tr_data = None
        self.__mlp_opt = None

    def fit(self, X_tr : pd.DataFrame, y_tr : pd.Series, val_size : float = 0.2):

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_tr, y_tr,
            test_size = val_size,
            random_state = 9999
            )

        self.__n_features = X_tr.shape[1]
        self.__tr_data = _affy_data(X_tr, y_tr)
        self.__scaler = self.__tr_data.pass_scaler()
        X_val = self.__scaler.transform(X_val.to_numpy())
        self.__X_val = torch.from_numpy(X_val).type(torch.FloatTensor)
        self.__y_val = torch.from_numpy(y_val.to_numpy()).type(torch.FloatTensor)

    def train(self, config : dict, max_epochs : int, reduction_factor : int,
        n_hyper_samples : int, n_cpus : int = 1, verbosity : int = 0,
        discard_logs : bool = True):
        """
        Optimize weights, biases, and hyperparameters.
        """
        if not self.__tr_data:
            raise AttributeError("This network has no data for training.")
        needed_config_params = ["l1", "l2", "p1", "p2", "lr", "weight_decay", "batch_size"]
        if any( hp not in config.keys() for hp in needed_config_params ):
            raise ValueError(f"{needed_config_params} not all keys in config.")

        def experiment(config, checkpoint_dir = None):

            mlp_bc = _mlp_bc_network(
                n_features = self.__n_features,
                l1 = int( config["l1"] ),
                l2 = int( config["l2"] ),
                p1 = float( config["p1"] ),
                p2 = float( config["p2"] )
                )

            loss_function = nn.BCEWithLogitsLoss()

            # https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
            optimizer = torch.optim.AdamW(
                mlp_bc.parameters(),
                lr = float( config["lr"] ),
                weight_decay = float( config["weight_decay"] )
                )

            tr_loader = DataLoader(
                self.__tr_data,
                batch_size = int( config["batch_size"] ),
                shuffle = True,
                drop_last = True # Prevent single sample batches.
                )

            if checkpoint_dir:
                model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
                mlp_bc.load_state_dict(model_state)
                optimizer.load_state_dict(optimizer_state)

            for epoch in range(max_epochs):

                # Train network.
                mlp_bc.train()
                for X_tr, y_tr in iter(tr_loader):

                    optimizer.zero_grad()
                    loss = loss_function(mlp_bc(X_tr), y_tr)
                    loss.backward()
                    optimizer.step()

                # Acquire validation metrics.
                raw_outputs = mlp_bc.val_forward(self.__X_val)
                val_loss = loss_function(raw_outputs, self.__y_val).item()
                val_auroc = roc_auc_score(
                    y_true = self.__y_val.numpy().astype("int32"),
                    y_score = mlp_bc.predict(self.__X_val, raw_outputs)
                    )

                # Store network state.
                with tune.checkpoint_dir(step = epoch) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    torch.save((mlp_bc.state_dict(), optimizer.state_dict()), path)

                # Report validation metrics to Ray Tune for early stopping and network selection.
                tune.report(val_loss = val_loss, val_auroc = val_auroc)

        scheduler = ASHAScheduler(
            time_attr = "training_iteration",
            metric = "val_loss",
            mode = "min",
            max_t = max_epochs,
            grace_period = 5,
            reduction_factor = reduction_factor)

        reporter = CLIReporter(
            parameter_columns = needed_config_params,
            metric_columns = ["val_loss", "val_auroc", "training_iteration"]
            )

        try:
            result = tune.run(
                experiment,
                resources_per_trial = {"cpu": n_cpus},
                config = config,
                num_samples = n_hyper_samples,
                scheduler = scheduler,
                progress_reporter = reporter,
                local_dir = os.path.join(os.getcwd(), "ray_results"),
                verbose = verbosity
                )

            # Locate the best trial.
            best_trial = result.get_best_trial(
                metric = "val_loss",
                mode = "min",
                scope = "last"
                )

            # Create new network with optimal structure.
            mlp_opt = _mlp_bc_network(
                n_features = self.__n_features,
                l1 = best_trial.config["l1"],
                l2 = best_trial.config["l2"],
                p1 = best_trial.config["p1"],
                p2 = best_trial.config["p2"]
            )

            best_checkpoint_dir = best_trial.checkpoint.value
            # Load best model from checkpoint data.
            model_state, optimizer_state = torch.load(
                os.path.join(best_checkpoint_dir, "checkpoint"))
            # Load new network with optimal weights and biases.
            mlp_opt.load_state_dict(model_state)
            # Store new network in object.
            self.__mlp_opt = mlp_opt

        finally:
            if discard_logs:
                # Remove all checkpointing data.
                shutil.rmtree(
                    os.path.join(os.getcwd(), "ray_results"),
                    ignore_errors = True
                    )

    def predict(self, X : pd.DataFrame) -> np.ndarray:
        """
        Generate predictions on pandas dataframe.
        """
        if not self.__mlp_opt:
            raise AttributeError("This network has not been optimized.")
        if X.shape[1] != self.__n_features:
            raise ValueError(f"X has {X.shape[1]} features. Expecting {self.__n_features}.")

        # Scale data, convert to Torch FloatTensor, and then predict.
        X = self.__scaler.transform(X.to_numpy())
        X = torch.from_numpy(X).type(torch.FloatTensor)
        return self.__mlp_opt.predict(X)


# TO DO:
# Add support for PBT scheduler. 
