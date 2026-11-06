"""Run an archived paper configuration with portable data/output paths."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=int, choices=[5], default=5)
    parser.add_argument('--data_dir', default='data/LPEM')
    parser.add_argument('--output_dir', default='results')
    parser.add_argument('--oracle_best_f1', action='store_true')
    cli = parser.parse_args()
    root = Path(__file__).resolve().parent
    config = json.loads((root / 'configs' / f'scene{cli.scene}.json').read_text())
    args = config['args']
    args.update(root_dir=str(Path(cli.data_dir).resolve()), load_data_dir='.',
                result_folder=str(Path(cli.output_dir).resolve() / f'scene{cli.scene}'))
    command = [sys.executable, str(root / 'train.py')]
    for name, value in args.items():
        if isinstance(value, bool):
            if value:
                command.append('--' + name)
        elif value is not None:
            command.extend(['--' + name, str(value)])
    if cli.oracle_best_f1:
        command.append('--oracle_best_f1')
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
