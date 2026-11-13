"""Download original FMP checkpoints from GitHub Release and verify SHA256."""
import argparse
import hashlib
import tempfile
import urllib.request
from pathlib import Path

MODELS = {'early': {'name': 'model_epoch_8_low_dim_v15_success_0.58.pth', 'url': 'https://github.com/lakshitadodeja/q2rl/releases/download/fmp-v15-20pct-seed1-8957390/model_epoch_8_low_dim_v15_success_0.58.pth', 'size': 1052511949, 'sha256': 'a1643e5f7c421401903946b1912552186cbd3aa8ac8a733aabb019329820b2e2'}, 'final': {'name': 'model_epoch_2000.pth', 'url': 'https://github.com/lakshitadodeja/q2rl/releases/download/fmp-v15-20pct-seed1-8957390/model_epoch_2000.pth', 'size': 1052495409, 'sha256': '2da94efa94d01e2ff12d45d40a529bd8f7cd50619a924a7403f63cf71b2768b2'}}


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', choices=['early', 'final', 'all'], default='all')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    selected = list(MODELS) if args.checkpoint == 'all' else [args.checkpoint]
    for key in selected:
        model = MODELS[key]
        target = out / model['name']
        if target.exists():
            assert sha256(target) == model['sha256'], f'Existing file differs; not overwritten: {target}'
            print('Already verified:', target)
            continue
        print('Downloading:', model['name'], flush=True)
        with tempfile.TemporaryDirectory(prefix='.fmp-download-', dir=out) as tmp:
            part = Path(tmp) / model['name']
            h = hashlib.sha256()
            received = 0
            milestone = 0
            request = urllib.request.Request(model['url'], headers={'User-Agent':'q2rl-fmp-model-downloader'})
            with urllib.request.urlopen(request, timeout=60) as response, part.open('wb') as stream:
                for block in iter(lambda: response.read(1024 * 1024), b''):
                    stream.write(block)
                    h.update(block)
                    received += len(block)
                    if received // (32 * 1024**2) > milestone:
                        milestone = received // (32 * 1024**2)
                        print(f"  {received / 1024**2:.0f}/{model['size'] / 1024**2:.0f} MiB", flush=True)
            assert received == model['size'], 'Downloaded size mismatch'
            assert h.hexdigest() == model['sha256'], 'Downloaded SHA256 mismatch'
            assert not target.exists(), f'Destination appeared during download: {target}'
            part.rename(target)
        print('SHA256 verified:', target)


if __name__ == '__main__':
    main()
