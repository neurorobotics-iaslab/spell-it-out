from importlib import resources

from spellitout import RunData, SpellItOutDataset


def test_package_imports_and_includes_resources():
    assert SpellItOutDataset.__name__ == "SpellItOutDataset"
    assert RunData.__name__ == "RunData"

    resource_path = resources.files("spellitout.resources").joinpath("phonItalia-1.10-wordforms.tsv")
    assert resource_path.is_file()



def test_core_exports_only_dataset_access():
    import spellitout

    assert set(spellitout.__all__) == {"RunData", "SpellItOutDataset"}
    assert not hasattr(spellitout, "build_syllable_datasets")
