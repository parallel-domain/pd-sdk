import argparse

from paralleldomain.utilities.logging import setup_loggers

from paralleldomain.visualization.statistics.dash.dash_viewer import DashViewer
from paralleldomain.visualization.statistics.rerun.rerun_viewer import RerunViewer


setup_loggers(logger_names=["__main__", "paralleldomain", "pd"])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Data lab example showing how to compute and visualize dataset statistics"
    )
    parser.add_argument(
        "-p", "--path", required=False, default=None, help="Path to a folder storing pre-computed statistics"
    )
    parser.add_argument(
        "-b", "--backend", required=False, type=str, default="dash", help="Desired backend", choices=["dash", "rerun"]
    )
    parser.add_argument(
        "-l", "--live", action="store_true", help="Watch folder for file modifications and update visualization"
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.backend == "dash":
        viewer, _ = DashViewer.create_from_filepath(path=args.path, watch_changes=args.live)
        viewer.launch(in_background=False)

    elif args.backend == "rerun":
        viewer, _ = RerunViewer.create_from_filepath(path=args.path, watch_changes=args.live)
        viewer.launch()
    else:
        raise TypeError(f"{args.backend} not in supported backends [rerun, dash]")


if __name__ == "__main__":
    main()
