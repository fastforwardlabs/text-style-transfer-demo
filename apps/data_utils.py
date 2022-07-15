import os
from typing import List
from collections import defaultdict

import numpy as np
from dataclasses import dataclass


@dataclass
class StyleAttributeData:
    source_attribute: str
    target_attribute: str
    examples: List[str]
    cls_model_path: str
    seq2seq_model_path: str
    sbert_model_path: str = "sentence-transformers/all-MiniLM-L6-v2"
    hf_base_url: str = "https://huggingface.co/"

    def __post_init__(self):
        self._make_attribute_selection_string()
        self._make_attribute_AND_string()
        self._make_attribute_THAN_string()

    def _make_attribute_selection_string(self):
        self.attribute_selecting_string = (
            f"{self.source_attribute}-{self.target_attribute}"
        )

    def _make_attribute_AND_string(self):
        self.attribute_AND_string = (
            f"**{self.source_attribute}** and **{self.target_attribute}**"
        )

    def _make_attribute_THAN_string(self):
        self.attribute_THAN_string = (
            f"**{self.source_attribute}** than **{self.target_attribute}**"
        )

    def build_model_url(self, model_type: str):
        """
        Build a complete HuggingFace url for the given `model_type`.

        Args:
            model_type (str): "cls", "seq2seq", "sbert"
        """
        attr_name = f"{model_type}_model_path"
        return os.path.join(self.hf_base_url, getattr(self, attr_name))


# instantiate data classes & collect all data class instances
DATA_PACKET = {
    "subjective-to-neutral": StyleAttributeData(
        source_attribute="subjective",
        target_attribute="neutral",
        examples=[
            "another strikingly elegant four-door saloon for the s3 continental came from james young.",
            "the most serious scandal was the iran-contra affair.",
            "chemical abstracts service (cas), a prominent division of the american chemical society, is the world's leading source of chemical information.",
            "this is an objective statement.",
        ],
        cls_model_path="cffl/bert-base-styleclassification-subjective-neutral",
        seq2seq_model_path="cffl/bart-base-styletransfer-subjective-to-neutral",
    ),
    "informal-to-formal": StyleAttributeData(
        source_attribute="informal",
        target_attribute="formal",
        examples=[
            "I am quitting my job",
            "That was funny LOL",
            "i loooooooooooooooooooooooove going to the movies.",
            "It's piece of cake, we can do it",
        ],
        cls_model_path="cointegrated/roberta-base-formality",
        seq2seq_model_path="prithivida/informal_to_formal_styletransfer",
    ),
}


def format_classification_results(id2label: dict, cls_result):
    """
    Formats classification output to be plotted using Altair.

    Args:
        id2label (dict): Transformer model's label dictionary
        cls_result (List): Classification pipeline output
    """

    labels = [v for k, v in id2label.items()]

    format_cls_result = []

    for i in range(len(labels)):
        temp = defaultdict()
        temp["type"] = labels[i].capitalize()
        temp["value"] = round(cls_result[0]["distribution"][i], 4)

        if i == 0:
            temp["percentage_start"] = 0
            temp["percentage_end"] = temp["value"]
        else:
            temp["percentage_start"] = 1 - temp["value"]
            temp["percentage_end"] = 1

        format_cls_result.append(temp)

    return format_cls_result


def string_to_list_string(text: str):
    return np.expand_dims(np.array(text), axis=0).tolist()
