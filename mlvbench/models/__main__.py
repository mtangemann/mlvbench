"""A script to print the information about models."""

import argparse

from tabulate import tabulate

from mlvbench.models import build_model, list_models


def main():
    """Run the model CLI."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list")
    list_parser.add_argument("patterns", type=str, nargs="*")
    list_parser.add_argument(
        "--bundle",
        type=str,
        default=None,
        metavar="NAME",
        help="List models from a named model bundle (e.g. standard10).",
    )
    list_parser.set_defaults(func=list_command)

    show_parser = subparsers.add_parser("show")
    show_parser.add_argument("model_name", type=str)
    show_parser.set_defaults(func=show_command)

    args = parser.parse_args()
    args.func(args)


def list_command(args: argparse.Namespace):
    """Print the list of available models."""
    patterns = args.patterns if len(args.patterns) > 0 else None
    models = list_models(patterns, bundle=args.bundle)
    for model_name in models:
        print(model_name)


def show_command(args: argparse.Namespace):
    """Print the information about a model."""
    model = build_model(args.model_name)
    print(model)
    print()

    if hasattr(model, "_transform"):
        print("transform:")
        print(model._transform)
        print()

    print("info:")
    info = model.info()
    table = []
    for key, value in info.items():
        if key == "num_parameters":
            value = human_readable_value(value)
        table.append((key, value))
    print(tabulate(table, headers=["Key", "Value"], tablefmt="simple"))


def human_readable_value(value: int) -> str:
    """Return a human-readable value."""
    if value < 1000:
        return str(value)
    elif value < 1000000:
        return f"{value / 1000:.1f}k"
    elif value < 1000000000:
        return f"{value / 1000000:.1f}M"
    else:
        return f"{value / 1000000000:.1f}B"


if __name__ == "__main__":
    main()
