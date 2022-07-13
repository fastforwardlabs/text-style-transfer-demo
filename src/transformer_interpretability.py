import torch
from transformers_interpret import SequenceClassificationExplainer
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


class InterpretTransformer:
    """
    Utility for visualizing word attribution scores from Transformer models.

    This class utilizes the [Transformers Interpret](https://github.com/cdpierse/transformers-interpret)
    libary to calculate word attributions using a techinique called Integrated Gradients.

    Attributes:
        cls_model_identifier (str)

    """

    def __init__(self, cls_model_identifier: str):

        self.cls_model_identifier = cls_model_identifier
        self.device = (
            torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        )

        self._initialize_hf_artifacts()

    def _initialize_hf_artifacts(self):
        """
        Initialize a HuggingFace artifacts (tokenizer and model) according
        to the provided identifiers for both SBert and the classification model.
        Then initialize the word attribution explainer with the HF model+tokenizer.

        """

        # classifer
        self.cls_tokenizer = AutoTokenizer.from_pretrained(self.cls_model_identifier)
        self.cls_model = AutoModelForSequenceClassification.from_pretrained(
            self.cls_model_identifier
        )
        self.cls_model.to(self.device)

        # transformers interpret
        self.explainer = SequenceClassificationExplainer(
            self.cls_model, self.cls_tokenizer
        )

    def visualize_feature_attribution_scores(self, text: str, class_index: int = 0):
        """
        Calculates and visualizes feature attributions using integrated gradients.

        Args:
            text (str) - text to get attributions for
            class_index (int) - Optional output index to provide attributions for

        """
        self.explainer(text, index=class_index)
        return self.explainer.visualize()
