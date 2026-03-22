import argparse
import json
import os
import numpy as np
import torch


SUMMERTIME_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
DEFAULT_MEGADEPTH_ROOT = '/mnt/sda/datasets/MegaDepth'
DEFAULT_NPZ_DIR = os.path.join(SUMMERTIME_ROOT, 'third_party/efficientloftr/assets/megadepth_test_1500_scene_info')
DEFAULT_OUTPUT = os.path.join(SUMMERTIME_ROOT, 'projects/Frontier/reports/dkm/dkm_megadepth_1500_native_results.json')


def test_mega1500(model, megadepth_root, npz_dir, max_pairs=None, output=None, dump_dir=None):
    from dkm.benchmarks import Megadepth1500Benchmark
    model.h_resized = 660
    model.w_resized = 880
    model.upsample_preds = True
    model.upsample_res = (1152, 1536)
    model.use_soft_mutual_nearest_neighbours = False

    benchmark = Megadepth1500Benchmark(npz_dir)
    benchmark.data_root = megadepth_root

    results = benchmark.benchmark(model, max_pairs=max_pairs, dump_dir=dump_dir)

    print(f'\n=== DKMv3 outdoor — MegaDepth-1500 ===')
    for k, v in results.items():
        print(f'  {k}: {v:.4f}')

    if output:
        os.makedirs(os.path.dirname(output), exist_ok=True)
        with open(output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'\nSaved to {output}')

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--megadepth_root', default=DEFAULT_MEGADEPTH_ROOT)
    parser.add_argument('--npz_dir', default=DEFAULT_NPZ_DIR)
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    parser.add_argument('--checkpoint', default=None,
                        help='Path to DKMv3 outdoor checkpoint (skips torch.hub download)')
    parser.add_argument('--max_pairs', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dump_dir', default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    from dkm.models.model_zoo import DKMv3_outdoor
    model = DKMv3_outdoor(path_to_weights=args.checkpoint)
    test_mega1500(model, args.megadepth_root, args.npz_dir,
                  max_pairs=args.max_pairs, output=args.output, dump_dir=args.dump_dir)
