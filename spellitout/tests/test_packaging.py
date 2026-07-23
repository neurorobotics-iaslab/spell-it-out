from importlib import resources

from spellitout import SpellItOut


def test_package_imports_and_includes_resources():
    assert SpellItOut.__name__ == "SpellItOut"

    resource_path = resources.files("spellitout.resources").joinpath("phonItalia-1.10-wordforms.tsv")
    assert resource_path.is_file()
