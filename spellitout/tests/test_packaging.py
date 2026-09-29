from importlib import resources

from spellitout import EventCode, RunData, SpellItOutDataset


def test_package_imports_and_includes_resources():
    assert SpellItOutDataset.__name__ == "SpellItOutDataset"
    assert RunData.__name__ == "RunData"
    assert int(EventCode.REPETITION) == 782

    resource_path = resources.files("spellitout.resources").joinpath("phonItalia-1.10-wordforms.tsv")
    assert resource_path.is_file()
