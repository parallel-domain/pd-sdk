import argparse
from tqdm import tqdm
from time import perf_counter

from paralleldomain.decoding.helper import decode_dataset, known_decoders
from paralleldomain.utilities.logging import setup_loggers
from paralleldomain.visualization.statistics.dash.dash_viewer import DashViewer
from paralleldomain.visualization.statistics.rerun.rerun_viewer import RerunViewer

setup_loggers(logger_names=["__main__", "paralleldomain", "pd"])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Data lab example showing how to compute and visualize dataset statistics"
    )
    parser.add_argument("-d", "--dataset_path", help="Path to a dgp dataset path", required=False)
    parser.add_argument(
        "-f",
        "--dataset_format",
        help="Dataset format",
        choices=[decoder.get_format() for decoder in known_decoders],
        required=False,
    )
    parser.add_argument(
        "-p", "--precomputed", required=False, default=None, help="Path to a folder storing pre-computed statistics"
    )
    parser.add_argument(
        "-b", "--backend", required=False, type=str, default="dash", help="Desired backend", choices=["dash", "rerun"]
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.backend == "dash":
        viewer, model = DashViewer.create_with_default_components()
        viewer.launch()
    elif args.backend == "rerun":
        viewer, model = RerunViewer.create_with_default_components()
        viewer.launch()
    else:
        raise TypeError(f"{args.backend} not in supported backends [rerun, dash]")

    if args.precomputed is None:
        # Compute dataset statistics and update visualization live
        dataset = decode_dataset(
            dataset_path=args.dataset_path,
            dataset_format=args.dataset_format,
        )

        t_start = perf_counter()
        for sensor_frame, _, scene in tqdm(
            dataset.sensor_frame_pipeline(only_cameras=True, concurrent=True, shuffle=True)
        ):
            if len(sensor_frame.available_annotation_types) == 0:
                continue

            model.update(scene=scene, sensor_frame=sensor_frame)

        print(f"Elapsed time {perf_counter() - t_start}")
    else:
        # Load and show pre-computed statistics
        model.load(args.precomputed)


if __name__ == "__main__":
    main()
