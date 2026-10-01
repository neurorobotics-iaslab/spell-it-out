from importlib import resources
import json

from spellitout import EventCode, RunData, SpellItOutDataset
from spellitout.publishing import write_derivative_scaffold


def test_package_imports_and_includes_resources():
    assert SpellItOutDataset.__name__ == "SpellItOutDataset"
    assert RunData.__name__ == "RunData"
    assert int(EventCode.REPETITION) == 782

    resource_path = resources.files("spellitout.resources").joinpath("phonItalia-1.10-wordforms.tsv")
    assert resource_path.is_file()


def test_preprocessing_derivative_description_has_generated_by(tmp_path):
    raw_root = tmp_path / "raw"
    output_root = raw_root / "derivatives" / "preprocessing"
    raw_root.mkdir()
    (raw_root / "dataset_description.json").write_text(
        json.dumps(
            {
                "Name": "Raw dataset",
                "BIDSVersion": "1.11.1",
                "DatasetType": "raw",
            }
        ),
        encoding="utf-8",
    )

    write_derivative_scaffold(output_root, raw_root, pipeline_version="test")

    description = json.loads(
        (output_root / "dataset_description.json").read_text(encoding="utf-8")
    )
    assert description["DatasetType"] == "derivative"
    assert description["GeneratedBy"][0]["Name"] == "preprocessing"
    assert description["GeneratedBy"][0]["Version"] == "test"
    assert description["SourceDatasets"] == [{"URL": "../.."}]
