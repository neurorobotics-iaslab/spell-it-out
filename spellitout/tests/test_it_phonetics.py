import pandas as pd

from spellitout import it_phonetics


def test_phonitalia_resource_loaders():
    wordforms = it_phonetics.load_phonitalia_wordforms(
        usecols=["word", "PhoneSyll"]
    )
    phones = it_phonetics.load_phonitalia_phones()
    syllables = it_phonetics.load_phonitalia_syllables()

    assert isinstance(wordforms, pd.DataFrame)
    assert list(wordforms.columns) == ["word", "PhoneSyll"]
    assert not wordforms.empty
    assert phones.columns[0] == "Phone"
    assert not phones.empty
    assert syllables.columns[0] == "phones"
    assert not syllables.empty


def test_word_pronunciations_are_loaded_lazily_and_cached():
    it_phonetics._word_pronunciations.cache_clear()
    assert it_phonetics._word_pronunciations.cache_info().currsize == 0

    result = it_phonetics.split_into_phonemes("casa")

    assert isinstance(result, str)
    assert it_phonetics._word_pronunciations.cache_info().currsize == 1
    it_phonetics.split_into_phonemes("casa")
    assert it_phonetics._word_pronunciations.cache_info().hits == 1
