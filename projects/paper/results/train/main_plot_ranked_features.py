import pandas as pd

from utils.utils_dataset import load_dataset
from projects.paper.results.train.utils_ranked_features import compute_sorted_features
from projects.paper.utils_paper import filename_dataset_paper
from utils.utils_latex import print_df_latex, plot_df_latex


def main_plot_ranked_features():
    (X_train, y_train, _, _, _, _, years_train, _, rcp_name_train, _, variable_names, _, ind_validation) = load_dataset(filename_dataset_paper)
    sorted_importance, sorted_names = compute_sorted_features(X_train, y_train, variable_names, ind_validation, select_k_features=10)

    df_latex = pd.DataFrame(data={"Feature name": sorted_names, "Feature importance": sorted_importance},
                            index=[f'#{i + 1}' for i in range(len(sorted_names))])
    df_latex['Feature importance'] = df_latex['Feature importance'].apply(lambda x: round(x, 3))
    df_latex.index.name = "Rank"
    df_latex.reset_index(inplace=True)
    print_df_latex(df_latex)
    plot_df_latex(df_latex, show=True)



if __name__ == '__main__':
    main_plot_ranked_features()