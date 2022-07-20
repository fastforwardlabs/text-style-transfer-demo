from typing import Iterable

import altair as alt
from captum.attr._utils.visualization import (
    VisualizationDataRecord,
    format_word_importances,
    _get_color,
)

try:
    from IPython.display import display, HTML

    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False

def format_classname(classname):
    return f'<td>{classname}</td>'

def visualize_text(
    datarecords: Iterable[VisualizationDataRecord], legend: bool = True
) -> "HTML":  # In quotes because this type doesn't exist in standalone mode
    assert HAS_IPYTHON, (
        "IPython must be available to visualize text. "
        "Please run 'pip install ipython'."
    )

    dom = []
    dom.append(
        '<head><link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous"></head>'
    )
    dom.append("""<table width:100; class="table">""")
    rows = [
        "<thead>"
        "<tr>"
        "<th scope='col'><span class='text-nowrap'>Predicted Label</span></th>"
        "<th scope='col'><span class='text-nowrap'>Attribution Score</span></th>"
        "<th scope='col'><span class='text-nowrap'>Feature Importance</span></th>"
        "</tr>"
        "</thead>"
    ]
    for datarecord in datarecords:
        rows.append(
            "".join(
                [
                    "<tbody>",
                    "<tr>",
                    format_classname(
                        f"{datarecord.pred_class.capitalize()}"
                    ),
                    format_classname(f"{round(datarecord.attr_score.item(), 2)}"),
                    format_word_importances(
                        datarecord.raw_input_ids, datarecord.word_attributions
                    ),
                    "<tr>",
                    "</tbody>",
                ]
            )
        )

    dom.append("".join(rows))
    dom.append("</table>")

    if legend:
        dom.append("<div class='row'>")
        dom.append("<div class='col-6'>")
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
        dom.append("<div class='col-6'></div>")

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
        domain=[record["type"] for record in format_cls_result],
        range=["#00A3AF", "#F96702"],
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
