from emulator.utils_metric.metric import Metric, metric_to_str, metric_to_function, is_metric_with_percentage
from emulator.utils_metric.utils_metric_function import condition_for_climatological_metrics


def add_metric_box(ax, y, y_predicted, target_label:str, split_name: str,
                   add_climatological_metrics=False):
    assert y.ndim == 1
    assert y_predicted.ndim == 1
    x_and_y_location = (0.5, 0.02)
    coef = 0.95
    metrics = [Metric.MRAE, Metric.RMSE, Metric.COR]
    if add_climatological_metrics and condition_for_climatological_metrics(y):
        metrics += [Metric.MRE_AVERAGE_FIRST_20_YEARS, Metric.MRE_AVERAGE_LAST_20_YEARS,
                    Metric.MRE_TREND_BETWEEN_FIRST_AND_LAST_20_YEARS]
    summary = []
    for metric in metrics:
        text = f'{metric_to_str[metric]}: {round(metric_to_function[metric](y, y_predicted), 2)}'
        if is_metric_with_percentage(metric):
            text += ' (%)'
        if metric == Metric.RMSE:
            if '(' in target_label:
                text += ' (' + target_label.split('(')[-1]
        summary.append(text)
    text_to_annotate = f'{split_name.capitalize()} metrics\n'
    text_to_annotate += '\n'.join(summary)
    ax.annotate(text_to_annotate, xy=x_and_y_location, xycoords='axes fraction', textcoords='offset points',
                xytext=x_and_y_location,
                bbox=dict(boxstyle="round", fc=(coef, coef, coef), ec="none"))





