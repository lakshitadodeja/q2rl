"""Download original FMP checkpoints from GitHub Release and verify SHA256."""
import argparse
import hashlib
import tempfile
import urllib.request
from pathlib import Path

MODELS = {'early': {'name': 'model_epoch_8.pth', 'url': 'https://github.com/lakshitadodeja/q2rl/releases/download/fmp-v15-full-seed1-8946875/model_epoch_8.pth', 'size': 1052493171, 'sha256': 'bb1b4e794e9724440e82ab5bc8f8c72ddfb6ca7fbe9ac405f9f8b4e81ba6daf6'}, 'final': {'name': 'model_epoch_2000.pth', 'url': 'https://github.com/lakshitadodeja/q2rl/releases/download/fmp-v15-full-seed1-8946875/model_epoch_2000.pth', 'size': 1052495409, 'sha256': '0306e5bad3e0c4e22cf4ec05ddb407e11a49dc4d9487be442a0d8bb825b4ed5f'}}


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
