from spellchecker import SpellChecker
import spacy
import inflect
from pattern.en import lexeme


def wordpool(word, pos, former_pos, dep, spell):
    pool = []

    title = word.istitle()

    word = word.lower()

    # misspelled
    misspelled = spell.candidates(word)

    pool.extend(list(misspelled))

    if pos == 'PREP':
        pool.extend(
            ['', 'aboard', 'about', 'above', 'across', 'after', 'against', 'ahead ', 'along', 'amid', 'among', 'around',
             'as', 'as far as', 'as of', 'aside from', 'at', 'because', 'because of',
             'before', 'behind', 'below', 'beneath', 'beside', 'between', 'beyond', 'but', 'by', 'despite', 'down',
             'during', 'except', 'far from', 'for', 'from', 'in', 'in front of', 'inside', 'instead of', 'into',
             'like', 'minus', 'near', 'next to', 'of', 'off', 'on', 'onto', 'opposite', 'out', 'outside', 'over',
             'past', 'plus',
             'regarding', 'since', 'than', 'through', 'throughout', 'till', 'to', 'towards', 'under', 'underneath',
             'unlike', 'up', 'upon',
             'via', 'with', 'within', 'without'
             ])

    if pos == 'DET':
        p1 = ['', 'a', 'an', 'other', 'another']
        p2 = ['this', 'that', 'the']
        p3 = ['my', 'your', 'his', 'her', 'its', 'our', 'their']
        p4 = ['these','those']

        if word in p1:
            pool.extend(p1)

        if word in p2:
            pool.extend(p2)

        if word in p3:
            pool.extend(p3)

        if word in p4:
            pool.extend(p4)

    if pos == 'SCONJ' or pos == 'CCONJ':
        pool.extend([word])

    if pos == 'PRON':
        pool_me = ['', 'I', 'me']

        pool_you = ['', 'you']

        pool_he = ['he']

        pool_she = ['she']

        if word in pool_me:
            pool.extend(pool_me)

        if word in pool_you:
            pool.extend(pool_you)

        if word in pool_he:
            pool.extend(pool_he)

        if word in pool_she:
            pool.extend(pool_she)

    if pos == 'AUX':
        if word in ['has', 'have']:
            pool.extend(['has', 'have'])

        if word in ['will', 'shall', 'can']:
            pool.extend(['will', 'shall', 'can'])

    if pos == 'NOUN':
        p = inflect.engine()

        if p.singular_noun(word):
            if former_pos != 'DET':
                pool.append('a ' + p.singular_noun(word))
                pool.append('an ' + p.singular_noun(word))

            pool.append(p.singular_noun(word))

        if p.plural(word):
            pool.append(p.plural(word))
            if former_pos != 'DET':
                pool.append('a ' + word)
                pool.append('an ' + word)

    if pos == 'VERB':
        vpool = []

        while True:
            try:
                vpool = lexeme(word)
            except:
                print('pattern error')
            else:
                break

        pool.extend(vpool)

    if not title:
        return pool

    else:
        return [i.title() for i in pool]


if __name__ == "__main__":
    spell = SpellChecker()
    sent = 'I am sad to read abouut Richard not being at his best .'

    nlp = spacy.load("en_core_web_sm")

    ss = nlp(sent)

    for token in ss:
        print(wordpool(token.text, token.pos_, token.dep_, spell))
