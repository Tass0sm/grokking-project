import argparse

from flax import nnx

from grokking_project.models import Transformer

def main(model, seed):

    model_classes = {
        "transformer": Transformer
    }

    rngs = nnx.Rngs(seed)

    model = model_classes[model](
        n_tokens=5,
        dim=256,
        heads=1,
        dropout=0.2,
        depth=1,
        rngs=rngs
    )

    breakpoint()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="transformer")
    parser.add_argument("--seed", default=0)
    args = parser.parse_args()

    main(model=args.model, seed=args.seed)
