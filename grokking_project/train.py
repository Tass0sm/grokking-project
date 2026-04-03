import os
import argparse
import numpy as np

import jax
import jax.numpy as jnp
from flax import nnx
import optax

import mlflow

from grokking_project.models import Transformer


def generate_data(p: int, train_fraction: float = 0.5, seed: int = 42):
    # Generate all pairs (a, b) in [0, p) x [0, p)
    pairs = [(a, b) for a in range(p) for b in range(0, p)]

    # labels
    results = [(a + b) % p for (a, b) in pairs]

    pairs = np.array(pairs, dtype=int)
    results = np.array(results, dtype=int)

    # Encode input sequences [a, op_token, b, equals_token]
    eq_token = p  # ID for '='
    seqs = np.stack([
        pairs[:, 0],                     # a
        pairs[:, 1],                     # b
        np.full(len(pairs), eq_token)    # '='
    ], axis=1)

    # Shuffle and split
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(seqs))
    n_train = int(train_fraction * len(seqs))
    train_idx, test_idx = indices[:n_train], indices[n_train:]

    X_train, y_train = seqs[train_idx], results[train_idx]
    X_test,  y_test  = seqs[test_idx],  results[test_idx]

    # Convert to JAX arrays (int32)
    X_train = jnp.array(X_train, dtype=jnp.int32)
    y_train = jnp.array(y_train, dtype=jnp.int32)
    X_test  = jnp.array(X_test,  dtype=jnp.int32)
    y_test  = jnp.array(y_test,  dtype=jnp.int32)

    return X_train, y_train, X_test, y_test


def main(model, divisor, n_epochs, seed):

    ###########################################################################
    #                               create model                              #
    ###########################################################################

    model_classes = {
        "transformer": Transformer
    }

    rngs = nnx.Rngs(seed)

    n_tokens=divisor+1
    model = model_classes[model](
        n_tokens=n_tokens, # [0, divisor) for numbers and +1 for equals
        dim=256,
        heads=1,
        dropout=0.2,
        depth=1,
        rngs=rngs
    )

    ###########################################################################
    #                               create data                               #
    ###########################################################################

    X_train, y_train, X_val, y_val = generate_data(
        divisor, seed=seed
    )

    ###########################################################################
    #                              train with SGD                             #
    ###########################################################################

    tx = optax.sgd(1e-4)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    @nnx.jit
    def train_step(model: nnx.Module, optimizer: nnx.Optimizer, x, y):
        def loss_fn(model):
            logits = model(x)
            one_hot = jax.nn.one_hot(y, n_tokens)
            loss = optax.softmax_cross_entropy(logits, one_hot).mean()
            return loss

        # nnx.value_and_grad handles model being stateful?
        loss, grads = nnx.value_and_grad(loss_fn)(model)

        optimizer.update(model, grads)
        return loss

    @nnx.jit
    def eval_step(model: nnx.Module, x, y):
        logits = model(x, training=False)
        one_hot = jax.nn.one_hot(y, n_tokens)
        loss = optax.softmax_cross_entropy(logits, one_hot).mean()
        preds = jnp.argmax(logits, axis=-1)
        acc = jnp.mean((preds == y).astype(jnp.float32))
        return loss, acc

    batch_size = 512
    num_train = X_train.shape[0]
    num_batches = int(np.ceil(num_train / batch_size))
    real_batch_size = num_train // num_batches

    with mlflow.start_run() as run:

        for epoch in range(1, n_epochs + 1):
            # Shuffle each epoch
            perm = np.random.permutation(num_train)
            X_train = X_train[perm]
            y_train = y_train[perm]
    
            epoch_loss = 0.0
            epoch_acc_count = 0
            for i in range(num_batches):
                batch_X = X_train[i * batch_size : (i+1) * batch_size]
                batch_y = y_train[i * batch_size : (i+1) * batch_size]
    
                loss = train_step(model, optimizer, batch_X, batch_y)
                epoch_loss += float(loss) * batch_X.shape[0]
    
            train_loss = epoch_loss / num_train
    
            # Validation set
            val_loss, val_acc = eval_step(model, X_val, y_val)
            val_loss = float(val_loss)
            val_acc  = float(val_acc)

            mlflow.log_metrics({
                "train_loss": epoch_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, epoch)

            if i % 100 == 0:
                print(f"Finished epoch {i}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="transformer")
    parser.add_argument("--divisor", type=int, default=10)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--seed", default=0)
    args = parser.parse_args()

    mlflow.set_tracking_uri("sqlite:////home/tassos/.local/share/mlflow/runs.db")
    if os.getenv("MLFLOW_EXPERIMENT_NAME") is None:
        mlflow.set_experiment("grokking")

    main(model=args.model,
         divisor=args.divisor,
         n_epochs=args.n_epochs,
         seed=args.seed)
