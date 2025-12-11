from copy import deepcopy
import numpy as np


def add_colorcell_to_highest_value_multicolumn(
    df, cols, suffix=["\cellcolor{ForestGreen}", "\cellcolor{SpringGreen}"], mode="max"
):
    """
    Add suffix before the highest value in the given columns. Used to highlight the best value in a row. See GLOMAP paper tables for reference.
    Args:
        df: dataframe
        cols: list of columns
        suffix: suffix to add to the highest value. The number of suffixes determins the lengh of the ranks and the colors.
                E.g. suffix=["\cellcolor{ForestGreen}", "\cellcolor{SpringGreen}"] will highlight the highest value with ForestGreen and the second highest with SpringGreen.
        mode: "max" or "min" to select the highest or lowest value in the given columns
    Returns:
        df: dataframe with the highest value in the given columns highlighted
    Examples:
        suffix = ["\cellcolor{ForestGreen}", "\cellcolor{SpringGreen}"]
        cols = ["col1", "col2", "col3"]
        mode = "max"
        out: a pandas dataframe that for each row has the highest value colored in ForestGreen and the second highest in SpringGreen only in the columns col1, col2, col3.
    """

    assert mode in ["max", "min"], "mode must be max or min"

    # remove invalid cols from cols
    index_invalid_cols = df[cols].isnull().all().values
    non_nan_cols = [col for idx, col in enumerate(cols) if index_invalid_cols[idx] == 0]
    invalid = index_invalid_cols.sum()
    assert len(suffix) <= (
        len(cols) - invalid
    ), f"suffix must be less or equal to the number of non nan-columns, suffix: {len(suffix)}, non nans-columns: {len(cols)-invalid}"

    # copy the dataframe
    df_color_cells = deepcopy(df[non_nan_cols])

    # init change list
    to_change = {i: [] for i in range(len(suffix))}

    # find index and col of highest values and store it
    for i in range(len(df_color_cells)):
        if mode == "max":
            idxs = df_color_cells.iloc[i].argsort().values[::-1][: len(suffix)]
        elif mode == "min":
            idxs = (
                df_color_cells.iloc[i].argsort().values[: len(suffix)]
            )  # invert order first

        # save as {0:[(col, idx), ...], 1:[(col, idx), ...], ...}
        for j in range(len(idxs)):
            to_change[j].append(idxs[j])

    # set to string
    df_color_cells = df_color_cells.astype(str)

    # add suffix to the highest value according to to_change list
    for k in to_change.keys():  # 0,1
        for row, col in enumerate(to_change[k]):  # row:[0,25], col, len(suffix)
            df_color_cells[df_color_cells.columns[col]][[row]] = (
                suffix[k] + df_color_cells[df_color_cells.columns[col]][row]
            )

    # readd invalid cols
    for idx, col in enumerate(cols):
        if index_invalid_cols[idx]:
            df_color_cells[col] = np.nan

    return df_color_cells


def sort_columns(columns, first_part_order, second_part_order):
    """
    Sort columns based on the first and second part of the column name.
    Args:
        columns: list of column names
        first_part_order: list of first part of the column name in the desired order
        second_part_order: list of second part of the column name in the desired order
    Returns:
        sorted_column_names: list of sorted column names
    Examples:
        cols = ["model1_test1", "model1_test2", "model2_test1", "model2_test2"]
        first_part_order = ["model1", "model2"]
        second_part_order = ["test1", "test2"]
        out: ["model1_test1", "model2_test1", "model1_test2", "model2_test2"]
    """
    split_columns = []

    # Split the column into first and second parts
    for col in columns:
        if "_" in col:
            prefix, suffix = col.split("_", 1)
        else:
            prefix, suffix = col, ""

        split_columns.append((prefix, suffix))

    # Sort first based on the second part, then the first part
    sorted_columns = sorted(
        split_columns,
        key=lambda x: (
            (
                second_part_order.index(x[1])
                if x[1] in second_part_order
                else len(second_part_order)
            ),
            first_part_order.index(x[0]),
        ),
    )

    # Join the sorted parts back into column names
    sorted_column_names = ["_".join(filter(None, col)) for col in sorted_columns]

    return sorted_column_names


def concat_columns(df, columns, new_column_name, sep="$\pm$", drop=True):
    """
    Concatenate columns into a new column.
    Args:
        df: dataframe
        columns: list of columns to concatenate
        new_column_name: name of the new column
        sep: separator between the columns
        drop: drop the original columns
    Returns:
        df: dataframe with the new column
    Examples:
        columns = [mean, std]
        sep = "$\pm$"
        new_column_name = e
        out: a pandas dataframe with a new column "new_col": mean $\pm$ std.
    """

    df[new_column_name] = df[columns].round(1).astype("str").agg(sep.join, axis=1)

    # remove columns
    if drop:
        df.drop(columns=columns, inplace=True)

    return df


############################################
def add_colorcell_to_highest_value(df, cols, suffix, mode="max"):
    """
    Add suffix before the highest value in the given columns. Used to highlight the best value in a row. See GLOMAP paper tables for reference.
    Args:
        df: dataframe
        cols: list of columns
        suffix: suffix to add to the highest value
        mode: "max" or "min" to select the highest or lowest value in the given columns
    Returns:
        df: dataframe with the highest value in the given columns highlighted
    Notes:
        Use add_colorcell_to_highest_value_multicolumn as generale case.
    """

    to_change = []
    df_color_cells = deepcopy(df[cols])
    # find index and col of highest value and store it
    for i in range(len(df_color_cells)):
        if mode == "max":
            idx = df_color_cells.iloc[i].argmax()
        elif mode == "min":
            idx = df_color_cells.iloc[i].argmin()
        to_change.append((cols[idx], i))

    # set to string
    df_color_cells = df_color_cells.astype(str)

    # add suffix to the highest value according to to_cange list
    for col, row in to_change:
        df_color_cells.loc[df_color_cells.index[row], col] = (
            suffix + df_color_cells.loc[df_color_cells.index[row], col]
        )

    # update cols
    for col in cols:
        df[col] = df_color_cells[col]

    return df
