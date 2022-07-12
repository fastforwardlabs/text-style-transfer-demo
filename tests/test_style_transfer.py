import pytest
import transformers

from src.style_transfer import StyleTransfer


@pytest.fixture
def subjectivity_styletransfer():
    MODEL_PATH = "cffl/bart-base-styletransfer-subjective-to-neutral"
    return StyleTransfer(model_identifier=MODEL_PATH, max_gen_length=200)


def test_StyleTransfer_init(subjectivity_styletransfer):
    assert isinstance(
        subjectivity_styletransfer.pipeline,
        transformers.pipelines.text2text_generation.Text2TextGenerationPipeline,
    )


def test_StyleTransfer_transfer(subjectivity_styletransfer):
    examples = [
        """there is an iconic roadhouse, named "spud's roadhouse", which sells fuel and general shop items , has great meals and has accommodation.""",
        "chemical abstracts service (cas), a prominent division of the american chemical society, is the world's leading source of chemical information.",
        "the most serious scandal was the iran-contra affair.",
        "another strikingly elegant four-door saloon for the s3 continental came from james young.",
        "other ambassadors also sent their messages of condolence following her passing.",
    ]

    ground_truth = [
        'there is a roadhouse, named "spud\'s roadhouse", which sells fuel and general shop items and has accommodation.',
        "chemical abstracts service (cas), a division of the american chemical society, is a source of chemical information.",
        "one controversy was the iran-contra affair.",
        "another four-door saloon for the s3 continental came from james young.",
        "other ambassadors also sent their messages of condolence following her death.",
    ]

    assert ground_truth == subjectivity_styletransfer.transfer(examples)
