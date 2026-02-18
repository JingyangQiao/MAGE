import os
import argparse
import json


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-file', type=str, default='./results/HQEdit/merge.jsonl')
    parser.add_argument('--output-dir', type=str)
    return parser.parse_args()


def eval_single(result_file):
    results = [json.loads(line) for line in open(result_file)]
    total, score = 0, 0

    for result in results:
        total += 1
        score += float(result['score'])

    print('Samples: {}\nAccuracy: {:.2f}%\n'.format(total, 100. * score / total))

    if args.output_dir is not None:
        output_file = os.path.join(args.output_dir, 'Result.text')
        with open(output_file, 'w') as f:
            f.write('Samples: {}\nAccuracy: {:.2f}%\n'.format(total, 100. * score / total))


if __name__ == "__main__":
    args = get_args()

    if args.result_file is not None:
        eval_single(args.result_file)
