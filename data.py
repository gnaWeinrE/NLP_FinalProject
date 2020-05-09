import Preprocess

import spacy
import time

if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")

    t0 = time.perf_counter()

    Preprocess.load_files(
        ['fce/m2/fce.train.gold.bea19.m2', 'wi+locness\m2\A.train.gold.bea19.m2',
         'wi+locness\m2\B.train.gold.bea19.m2', 'wi+locness\m2\C.train.gold.bea19.m2'], nlp,
        'train.txt')

    Preprocess.load_files(['fce/m2/fce.dev.gold.bea19.m2', 'wi+locness\m2\ABCN.dev.gold.bea19.m2'], nlp, 'dev.txt')

    Preprocess.load_files(['fce/m2/fce.test.gold.bea19.m2'], nlp, 'test.txt')

    print(time.perf_counter() - t0)
    print('train/dev/test data loaded')
    t0 = time.perf_counter()
