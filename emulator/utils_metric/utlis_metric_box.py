from emulator.utils_metric.metric import Metric, metric_to_str, metric_to_function


def add_metric_box(ax, y, y_predicted, prefix, x_and_y_location, coef=0.95):
    metrics = [Metric.MRAE, Metric.MSE, Metric.COR]
    summary = [f'{metric_to_str[metric]}: {round(metric_to_function[metric](y, y_predicted), 2)}' for metric in metrics]
    text_to_annotate = f'{prefix} metrics\n'
    text_to_annotate += '\n'.join(summary)
    ax.annotate(text_to_annotate, xy=x_and_y_location, xycoords='axes fraction', textcoords='offset points',
                xytext=x_and_y_location,
                bbox=dict(boxstyle="round", fc=(coef, coef, coef), ec="none"))




