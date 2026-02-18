import os
import argparse
import json
from collections import Counter


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-file', type=str, default='./OCRVQA/val.json')
    parser.add_argument('--result-file', type=str, default='./results/OCRVQA/merge.jsonl')
    parser.add_argument('--output-dir', type=str)
    return parser.parse_args()


def has_word_repeat(s):
    words = s.split()
    word_counts = Counter(words)

    return any(count > 2 for count in word_counts.values())


def eval_single(annotation_file, result_file):
    annotations = json.load(open(annotation_file))
    annotations = {annotation['question_id']: annotation for annotation in annotations}
    results = [json.loads(line) for line in open(result_file)]

    total = len(results)
    right = 0
    for result in results:
        annotation = annotations[result['question_id']]
        pred = result['text']
        ground_truth = annotation['answer']
        if 'Unanswerable' in pred:
            continue
        if has_word_repeat(pred):
            continue
        if pred.lower().strip() == ground_truth.lower().strip():
            right += 1
        elif ground_truth.lower().strip() in pred.lower().strip() or pred.lower().strip() in ground_truth.lower().strip():
            right += 1
        else:
            continue
        
    print('Samples: {}\nAccuracy: {:.2f}%\n'.format(total, 100. * right / total))

    if args.output_dir is not None:
        output_file = os.path.join(args.output_dir, 'Result.text')
        with open(output_file, 'w') as f:
            f.write('Samples: {}\nAccuracy: {:.2f}%\n'.format(total, 100. * right / total))

if __name__ == "__main__":
    args = get_args()

    if args.result_file is not None:
        eval_single(args.annotation_file, args.result_file)
