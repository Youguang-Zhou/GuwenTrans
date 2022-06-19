import argparse

from nltk.translate.bleu_score import corpus_bleu


def get_args():
    parser = argparse.ArgumentParser('GuwenTrans')
    parser.add_argument('--test_gold')  # ground truth
    parser.add_argument('--test_pred')  # model output
    parser.add_argument('--bleu-output', default='bleu.txt')
    return parser.parse_args()


def main(args):
    test_gold = args.test_gold
    test_pred = args.test_pred
    bleu_output = args.bleu_output

    with open(test_gold, 'r') as f:
        references = [[list(ref)] for ref in f.read().splitlines()]
    with open(test_pred, 'r') as f:
        candidates = [list(cand) for cand in f.read().splitlines()]

    assert len(references) == len(candidates)

    bleu = corpus_bleu(references, candidates)
    bleu_1_gram = corpus_bleu(references, candidates, weights=(1, 0, 0, 0))
    bleu_2_gram = corpus_bleu(references, candidates, weights=(0, 1, 0, 0))
    bleu_3_gram = corpus_bleu(references, candidates, weights=(0, 0, 1, 0))
    bleu_4_gram = corpus_bleu(references, candidates, weights=(0, 0, 0, 1))

    with open(bleu_output, 'w') as f:
        out = f'BLEU = {bleu*100:.2f}, {bleu_1_gram*100:.1f}/{bleu_2_gram*100:.1f}/{bleu_3_gram*100:.1f}/{bleu_4_gram*100:.1f}'
        print(out)
        f.write(out)


if __name__ == '__main__':
    args = get_args()
    main(args)
