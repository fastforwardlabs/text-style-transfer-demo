from turtle import st
from apps.data_utils import DATA_PACKET
from src.style_transfer import StyleTransfer
from src.style_classification import StyleIntensityClassifier
from src.content_preservation import ContentPreservationScorer
from src.transformer_interpretability import InterpretTransformer


def load_and_cache_HF_models(style_data_packet):
    """
    This utility function is used to download and cache models needed for all style
    attributes in `apps.data_utils.DATA_PACKET`

    Args:
        style_data_packet (dict)
    """

    for style_data in style_data_packet.keys():
        try:
            st = StyleTransfer(model_identifier=style_data.seq2seq_model_path)
            sic = StyleIntensityClassifier(style_data.cls_model_path)
            cps = ContentPreservationScorer(
                cls_model_identifier=style_data.cls_model_path,
                sbert_model_identifier=style_data.sbert_model_path,
            )

            del st, sic, cps
        except Exception as e:
            print(e)

if __name__=="__main__":
    load_and_cache_HF_models(DATA_PACKET)