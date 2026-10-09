from enum import Enum
from functools import lru_cache
from importlib.resources import files
from typing import Any

import pandas as pd


def _load_phonitalia_resource(
    filename: str,
    *,
    default_header: int,
    **read_csv_kwargs: Any,
) -> pd.DataFrame:
    """Load one packaged phonItalia TSV using overridable Pandas options."""
    options: dict[str, Any] = {
        "sep": "\t",
        "header": default_header,
        "low_memory": False,
    }
    options.update(read_csv_kwargs)
    resource = files("spellitout.resources").joinpath(filename)
    with resource.open("r", encoding="utf-8") as stream:
        return pd.read_csv(stream, **options)


def load_phonitalia_wordforms(**read_csv_kwargs: Any) -> pd.DataFrame:
    """Load ``phonItalia-1.10-wordforms.tsv`` as a DataFrame.

    Additional keyword arguments are forwarded to :func:`pandas.read_csv` and
    override the defaults. For example, ``usecols=["word", "PhoneSyll"]`` can
    be used to load only the orthographic and phonological syllable columns.
    """
    return _load_phonitalia_resource(
        "phonItalia-1.10-wordforms.tsv",
        default_header=0,
        **read_csv_kwargs,
    )


def load_phonitalia_phones(**read_csv_kwargs: Any) -> pd.DataFrame:
    """Load ``phonItalia-1.10-phones.tsv`` as a DataFrame.

    Its second physical row contains the tabular column names, so ``header=1``
    is used by default. Pandas disambiguates repeated names with suffixes such
    as ``.1`` and ``.2``. Pass a different ``header`` value to override this.
    """
    return _load_phonitalia_resource(
        "phonItalia-1.10-phones.tsv",
        default_header=1,
        **read_csv_kwargs,
    )


def load_phonitalia_syllables(**read_csv_kwargs: Any) -> pd.DataFrame:
    """Load ``phonItalia-1.10-syllables.tsv`` as a DataFrame.

    Its second physical row contains the tabular column names, so ``header=1``
    is used by default. Pandas disambiguates repeated names with suffixes such
    as ``.1`` and ``.2``. Pass a different ``header`` value to override this.
    """
    return _load_phonitalia_resource(
        "phonItalia-1.10-syllables.tsv",
        default_header=1,
        **read_csv_kwargs,
    )


@lru_cache(maxsize=1)
def _word_pronunciations() -> pd.DataFrame:
    """Lazily cache only the columns needed for word transcription."""
    return load_phonitalia_wordforms(usecols=["word", "PhoneSyll"])

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

def split_into_phonemes(text):
    # Replace punctuation with space
    text = text.replace('.',' ').replace(',',' ').replace(';',' ').replace(':',' ').replace('!',' ')\
               .replace('?',' ').replace('(',' ').replace(')',' ').replace('-',' ').replace('\'',' ').replace('’',' ')
    # Remove numbers and non-standard characters
    text = ''.join([i for i in text if not i.isdigit() and i.isalpha() or i == ' '])
    text = text.replace('à','a').replace('è','e').replace('é','e').replace('ì','i').replace('ò','o').replace('ù','u')
    words = text.lower().split()

    phonitalia = _word_pronunciations()
    syllables = []
    for word in words:
        if word in phonitalia.word.values:
            syllables.append(phonitalia.loc[phonitalia.word==word,'PhoneSyll'].iloc[0])
        else: 
            raise Exception(f'Could not split sentence due to word not in database: {word}')
    return ' '.join(syllables)
