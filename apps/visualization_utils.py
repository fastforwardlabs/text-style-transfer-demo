from typing import Iterable

import altair as alt
from captum.attr._utils.visualization import (
    VisualizationDataRecord,
    format_classname,
    format_word_importances,
    _get_color,
)

try:
    from IPython.display import display, HTML

    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False


def visualize_text(
    datarecords: Iterable[VisualizationDataRecord], legend: bool = True
) -> "HTML":  # In quotes because this type doesn't exist in standalone mode
    assert HAS_IPYTHON, (
        "IPython must be available to visualize text. "
        "Please run 'pip install ipython'."
    )
    dom = [
        """<table width:100; style='font-family:"Source Sans Pro", sans-serif'; border-collapse:collapse;>"""
    ]
    rows = [
        '<tr style="text-align:left; font-size: 20; font-weight: bold; color: 00a2ad;"><td>Predicted Label</td>'
        "<td>Attribution Score</td>"
        "<td>Word Importance</td>"
    ]
    for datarecord in datarecords:
        rows.append(
            "".join(
                [
                    "<tr>",
                    format_classname(
                        "{0} ({1:.2f})".format(
                            datarecord.pred_class, datarecord.pred_prob
                        )
                    ),
                    format_classname("{0:.2f}".format(datarecord.attr_score)),
                    format_word_importances(
                        datarecord.raw_input_ids, datarecord.word_attributions
                    ),
                    "<tr>",
                ]
            )
        )

    dom.append("".join(rows))
    dom.append("</table>")

    if legend:
        dom.append(
            '<div style="border-top: 1px solid; margin-top: 5px; \
            padding-top: 5px; display: inline-block">'
        )
        dom.append("<b>Legend: </b>")

        for value, label in zip([-1, 0, 1], ["Negative", "Neutral", "Positive"]):
            dom.append(
                '<span style="display: inline-block; width: 10px; height: 10px; \
                border: 1px solid; background-color: \
                {value}"></span> {label}  '.format(
                    value=_get_color(value), label=label
                )
            )
        dom.append("</div>")

    html = HTML("".join(dom))
    display(html)

    return html


def build_altair_classification_plot(format_cls_result):
    """
    Builds Altair bar chart for classification results.

    Args:
        format_cls_result (List): Output from `format_classification_results()`
    """
    source = alt.pd.DataFrame(format_cls_result)

    color_scale = alt.Scale(
        domain=["Subjective", "Neutral"], range=["#94c6da", "#1770ab"]
    )

    c = (
        alt.Chart(source)
        .mark_bar(size=50)
        .encode(
            x=alt.X(
                "percentage_start:Q", axis=alt.Axis(title="Style Distribution (%)")
            ),
            x2=alt.X2("percentage_end:Q"),
            color=alt.Color(
                "type:N",
                legend=alt.Legend(title="Attribute"),
                scale=color_scale,
            ),
        )
        .properties(height=150)
    )

    return c
