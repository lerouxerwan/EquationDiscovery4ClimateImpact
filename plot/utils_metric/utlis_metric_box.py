from plot.utils_metric.metric import Metric, metric_to_str, metric_to_function, is_metric_with_percentage, \
    is_metric_with_target_unit


def add_metric_box(ax, y, y_predicted, target_label:str, split_name: str, x_and_y_location=None):
    assert y.ndim == 1
    assert y_predicted.ndim == 1
    if x_and_y_location is None:
        x_and_y_location = (0.65, 0.05)
    metrics = [Metric.MRAE, Metric.RMSE, Metric.MEDAE, Metric.COR, Metric.SPREADRATIO]
    metrics = [Metric.RMSE, Metric.MRAE, Metric.COR_DETRENDED]
    # metrics = [Metric.RMSE, Metric.MRAE, Metric.COR_YBY_RE]
    summary = []
    for metric in metrics:
        text = f'{metric_to_str[metric]}: {round(metric_to_function[metric](y, y_predicted), 2)}'
        if is_metric_with_percentage(metric):
            text += ' (%)'
        if is_metric_with_target_unit(metric):
            if '(' in target_label:
                text += ' (' + target_label.split('(')[-1]
        summary.append(text)
    # text_to_annotate = f'{split_name.capitalize()} metrics\n'
    text_to_annotate = f''
    text_to_annotate += '\n'.join(summary)
    coef = 0.95
    ax.annotate(text_to_annotate, xy=x_and_y_location, xycoords='axes fraction', textcoords='offset points',
                xytext=x_and_y_location,
                bbox=dict(boxstyle="round", fc=(coef, coef, coef), ec="none"))





