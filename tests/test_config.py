from pathlib import Path

from rag_mapreduce.config import load_pipeline_config


def test_load_pipeline_config(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
        dataset_path: data/samples/mini_corpus.txt
        artifact_dir: artifacts/vectorstores
        index_name: demo
        pipeline_type: sequential
        retrieval:
          top_k: 3
        """
    )
    config = load_pipeline_config(config_path)
    assert config.index_name == "demo"
    assert config.retrieval.top_k == 3
    assert config.dataset.exists()
