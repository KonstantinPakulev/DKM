import argparse
import json
import os
import torch


SUMMERTIME_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
DEFAULT_SCANNET_ROOT = '/mnt/sda/datasets/scannet1500'
DEFAULT_NPZ_PATH = os.path.join(SUMMERTIME_ROOT, 'third_party/efficientloftr/assets/scannet_test_1500/test.npz')
DEFAULT_OUTPUT = os.path.join(SUMMERTIME_ROOT, 'projects/Frontier/reports/dkm/dkm_scannet_1500_native_results.json')


def test_scannet(model, scannet_root, npz_path, max_pairs=None, output=None, dump_dir=None, shuffle=True):
    from dkm.benchmarks import ScanNetBenchmark
    model.h_resized = 480
    model.w_resized = 640
    model.upsample_preds = False

    benchmark = ScanNetBenchmark(npz_path=npz_path, scans_dir=scannet_root)
    results = benchmark.benchmark(model, max_pairs=max_pairs, dump_dir=dump_dir, shuffle=shuffle)

    print(f'\n=== DKMv3 indoor — ScanNet-1500 ===')
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
    parser.add_argument('--scannet_root', default=DEFAULT_SCANNET_ROOT)
    parser.add_argument('--npz_path', default=DEFAULT_NPZ_PATH)
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    parser.add_argument('--max_pairs', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dump_dir', default=None)
    parser.add_argument('--no_shuffle', action='store_true')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    from dkm.models.model_zoo import DKMv3_indoor
    model = DKMv3_indoor()
    test_scannet(model, args.scannet_root, args.npz_path,
                 max_pairs=args.max_pairs, output=args.output, dump_dir=args.dump_dir,
                 shuffle=not args.no_shuffle)
