from importlib.resources import files
from enum import Enum

import numpy as np
from pandas import read_csv

try:
    resource_path = files("spellitout.resources").joinpath("phonItalia-1.10-wordforms.tsv")
    with resource_path.open("r", encoding="utf-8") as f:
        phonitalia = read_csv(f, delimiter="\t", low_memory=False)[['word', 'PhoneSyll']]
except Exception as e:
    print(f"Could not load phonetics database: {e}. Please check that the package is correctly installed.")

class PhoneticCategory(Enum):
    # name is the name of the category
    # value is the value of the category
    CONSONANT = 'CONSONANT'
    VOWEL = 'VOWEL'

    NASAL = 'NASAL'
    OCCLUSIVE = 'OCCLUSIVE'
    AFFRICATE = 'AFFRICATE'
    FRICATIVE = 'FRICATIVE'
    VIBRANT = 'VIBRANT'
    LATERAL = 'LATERAL'
    APPROXIMANT = 'APPROXIMANT'

    BILABIAL = 'BILABIAL'
    LABIODENTAL = 'LABIODENTAL'
    DENTIALVEOLAR = 'DENTIALVEOLAR'
    POSTALVEOLAR = 'POSTALVEOLAR'
    PALATAL = 'PALATAL'
    VELAR = 'VELAR'

    HIGH = 'HIGH'
    MEDIUM_HIGH = 'MEDIUM_HIGH'
    MEDIUM_LOW = 'MEDIUM_LOW'
    LOW = 'LOW'

    ANTERIOR = 'ANTERIOR'
    CENTRAL = 'CENTRAL' 
    POSTERIOR = 'POSTERIOR'

    def __repr__(self):
        return self.value
    
    def __str__(self):
        return self.value
    

class Phoneme:
    def __init__(self,name,categories):
       self.name = name
       self.categories = categories

    def has_category(self,category):
        return category in self.categories
    def __repr__(self):
        return f"Phoneme('{self.name}',{self.categories})"
    def __str__(self):
        return self.name
   
class ItalianPhonemes(Enum):
    a = Phoneme('a',[PhoneticCategory.VOWEL,PhoneticCategory.LOW,PhoneticCategory.CENTRAL])
    b = Phoneme('b',[PhoneticCategory.CONSONANT,PhoneticCategory.BILABIAL,PhoneticCategory.OCCLUSIVE])
    c = Phoneme('c',[PhoneticCategory.CONSONANT,PhoneticCategory.POSTALVEOLAR,PhoneticCategory.AFFRICATE])
    d = Phoneme('d',[PhoneticCategory.CONSONANT,PhoneticCategory.DENTIALVEOLAR,PhoneticCategory.OCCLUSIVE])
    e = Phoneme('e',[PhoneticCategory.VOWEL,PhoneticCategory.MEDIUM_HIGH,PhoneticCategory.ANTERIOR])
    E = Phoneme('E',[PhoneticCategory.VOWEL,PhoneticCategory.MEDIUM_LOW,PhoneticCategory.ANTERIOR])
    f = Phoneme('f',[PhoneticCategory.CONSONANT,PhoneticCategory.LABIODENTAL,PhoneticCategory.FRICATIVE])
    G = Phoneme('G',[PhoneticCategory.CONSONANT,PhoneticCategory.VELAR,PhoneticCategory.OCCLUSIVE])
    g = Phoneme('g',[PhoneticCategory.CONSONANT,PhoneticCategory.POSTALVEOLAR,PhoneticCategory.AFFRICATE])
    i = Phoneme('i',[PhoneticCategory.VOWEL,PhoneticCategory.HIGH,PhoneticCategory.ANTERIOR])
    j = Phoneme('j',[PhoneticCategory.CONSONANT,PhoneticCategory.PALATAL,PhoneticCategory.APPROXIMANT])
    k = Phoneme('k',[PhoneticCategory.CONSONANT,PhoneticCategory.VELAR,PhoneticCategory.OCCLUSIVE])
    L = Phoneme('L',[PhoneticCategory.CONSONANT,PhoneticCategory.PALATAL,PhoneticCategory.LATERAL])
    l = Phoneme('l',[PhoneticCategory.CONSONANT,PhoneticCategory.DENTIALVEOLAR,PhoneticCategory.LATERAL])
    m = Phoneme('m',[PhoneticCategory.CONSONANT,PhoneticCategory.BILABIAL,PhoneticCategory.NASAL])
    n = Phoneme('n',[PhoneticCategory.CONSONANT,PhoneticCategory.DENTIALVEOLAR,PhoneticCategory.NASAL])
    N = Phoneme('N',[PhoneticCategory.CONSONANT,PhoneticCategory.PALATAL,PhoneticCategory.NASAL])
    O = Phoneme('O',[PhoneticCategory.VOWEL,PhoneticCategory.MEDIUM_LOW,PhoneticCategory.POSTERIOR])
    o = Phoneme('o',[PhoneticCategory.VOWEL,PhoneticCategory.MEDIUM_HIGH,PhoneticCategory.POSTERIOR])
    p = Phoneme('p',[PhoneticCategory.CONSONANT,PhoneticCategory.BILABIAL,PhoneticCategory.OCCLUSIVE])
    r = Phoneme('r',[PhoneticCategory.CONSONANT,PhoneticCategory.DENTIALVEOLAR,PhoneticCategory.VIBRANT])
    s = Phoneme('s',[PhoneticCategory.CONSONANT,PhoneticCategory.DENTIALVEOLAR,PhoneticCategory.FRICATIVE])
    S = Phoneme('S',[PhoneticCategory.CONSONANT,PhoneticCategory.POSTALVEOLAR,PhoneticCategory.FRICATIVE])
    t = Phoneme('t',[PhoneticCategory.CONSONANT,PhoneticCategory.DENTIALVEOLAR,PhoneticCategory.OCCLUSIVE])
    u = Phoneme('u',[PhoneticCategory.VOWEL,PhoneticCategory.HIGH,PhoneticCategory.POSTERIOR])
    v = Phoneme('v',[PhoneticCategory.CONSONANT,PhoneticCategory.LABIODENTAL,PhoneticCategory.FRICATIVE])
    w = Phoneme('w',[PhoneticCategory.CONSONANT,PhoneticCategory.VELAR,PhoneticCategory.APPROXIMANT])
    z = Phoneme('z',[PhoneticCategory.CONSONANT,PhoneticCategory.DENTIALVEOLAR,PhoneticCategory.AFFRICATE])
    Z = Phoneme('Z',[PhoneticCategory.CONSONANT,PhoneticCategory.DENTIALVEOLAR,PhoneticCategory.AFFRICATE])
    
def get_phoneme(ch):
    return ItalianPhonemes[ch].value

def string2phonlist(text):
    if type(text) == str:
        text = text.replace('.','').replace(' ','')
    return [ItalianPhonemes[phoneme].value for phoneme in text]

def calc_phonesyll_freq(sentence_list):
    all_phones = [p for s in sentence_list for w in split_into_phonemes(s).split() for p in w.split('.')]
    # all_phones = np.concatenate([split_into_phonemes(sentence) for sentence in sentence_list])
    phones,counts = np.unique(all_phones,return_counts=True)
    sorted_idx = np.argsort(counts)[::-1]
    phones = phones[sorted_idx]
    counts = counts[sorted_idx]
    return phones,counts

def calc_phone_freq(sentence_list):
    all_phones = [p for s in sentence_list for w in split_into_phonemes(s).split() for p in w.replace('.','').replace(' ','')]
    # all_phones = np.concatenate([split_into_phonemes(sentence) for sentence in sentence_list])
    phones,counts = np.unique(all_phones,return_counts=True)
    return phones,counts

def split_into_phonemes(text):
    # Replace punctuation with space
    text = text.replace('.',' ').replace(',',' ').replace(';',' ').replace(':',' ').replace('!',' ')\
               .replace('?',' ').replace('(',' ').replace(')',' ').replace('-',' ').replace('\'',' ').replace('’',' ')
    # Remove numbers and non-standard characters
    text = ''.join([i for i in text if not i.isdigit() and i.isalpha() or i == ' '])
    text = text.replace('à','a').replace('è','e').replace('é','e').replace('ì','i').replace('ò','o').replace('ù','u')
    words = text.lower().split()

    syllables = []
    for word in words:
        if word in phonitalia.word.values:
            syllables.append(phonitalia.loc[phonitalia.word==word,'PhoneSyll'].iloc[0])
        else: 
            raise Exception(f'Could not split sentence due to word not in database: {word}')
    return ' '.join(syllables)