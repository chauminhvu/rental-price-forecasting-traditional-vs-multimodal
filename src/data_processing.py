import sys
sys.path.append("/home/vchau/.local/lib/python3.10/site-packages")
import collections
# import sqlparse
import math
import cupy as cp
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from missingpy import MissForest
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("whitegrid")
sns.set_palette("deep")
sns.set(rc={'figure.figsize':(8,6)})


def count_missing_values(data):
    """
    Analyze and report the missing values in a pandas DataFrame or Series.
    Calculates the number of missing values for each column or the single column 
    along with their percentages relative to total rows.

    Parameters:
        data (pandas DataFrame or Series): Input DataFrame or Series to be analyzed.

    Returns:
        pandas DataFrame: A report containing the sorted missing value count and percentage
        for each column.

    Raises:
        TypeError: If input is not a pandas DataFrame or Series.
        ValueError: If input DataFrame has no columns.
    """

    if not isinstance(data, (pd.DataFrame, pd.Series)) or data.empty:
        raise ValueError(
            'Input must be a non-empty pandas DataFrame or a non-empty pandas Series')

    if isinstance(data, pd.Series):
        data = data.to_frame()
    
    missing_values = data.isnull().sum()
    missing_percentage = (missing_values / len(data)) * 100

    report = pd.DataFrame({
        'Missing count': missing_values,
        'Percentage': missing_percentage
        }).sort_values(by='Percentage', ascending=False)
    return report


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the Haversine distance between two sets of latitude and longitude.

    Args:
        lat1 (float): Latitude of the first point in degrees.
        lon1 (float): Longitude of the first point in degrees.
        lat2 (float): Latitude of the second point in degrees.
        lon2 (float): Longitude of the second point in degrees.

    Returns:
        float: The distance between the two points in kilometers.
    """
    # Radius of the Earth
    R = 6371

    # convert to radian
    lat1 = cp.deg2rad(lat1)
    lon1 = cp.deg2rad(lon1)
    lat2 = cp.deg2rad(lat2)
    lon2 = cp.deg2rad(lon2)

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = cp.sin(dlat / 2)**2 + cp.cos(lat1) * cp.cos(lat2) * cp.sin(dlon / 2)**2
    d = 2 * cp.arcsin(cp.sqrt(a)) * R
    return d


def compute_mad_1d(x, k=[2.5, 2.5]):
    """
    References:
        https://doi.org/10.1016/j.jesp.2013.03.013
    """
    x = cp.asarray(x)
    isnan = cp.isnan(x)

    if x.size == 0:
        raise ValueError("Input is an empty array.")

    if isnan.any():
        raise ValueError("Input contains NaN.")

    med = cp.median(x)
    mad = 1.486 * cp.median(cp.abs(x - med))
    lower = med - k[0] * mad
    upper = med + k[1] * mad
    return lower, upper


def plot_bar_charts(df, figsize=(15, 10)):
    """
    Plot bar charts

    Parameters:
        df (DataFrame): DataFrame containing the data to plot.
        figsize (tuple, optional): Figure size
    """
    # Get the number of rows and columns for the subplot grid
    num_rows = math.ceil(len(df.columns) / 3)
    num_cols = min(len(df.columns), 3)

    # Create a grid of subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

    # Flatten the axes array if necessary
    axes = axes.flatten()

    # Get the color palette from seaborn
    colors = sns.color_palette("deep")

    # Loop through the columns
    for i, column in enumerate(df.columns):
        counts = df[column].value_counts()
        percent = 100 * (counts.values / df[column].shape[0])
        
        # Create a bar plot
        axes[i].bar(counts.index, percent, color=colors, alpha=0.8)
        axes[i].set_title(column, fontsize=16, fontweight='bold')
        axes[i].tick_params(axis='x', rotation=45, labelsize=12)

    # Hide any remaining empty subplots
    for j in range(len(df.columns), num_rows * num_cols):
        axes[j].axis('off')

    # Adjust layout
    plt.tight_layout()
    plt.show()


def plot_pie_charts(df, figsize=(15, 10), pie_size=1.5):
    """
    Plot pie charts

    Parameters:
        df (DataFrame): DataFrame containing the data to plot.
        figsize (tuple, optional): Figure size (width, height)
        pie_size (float, optional): Size of the pie chart.
    """
    # Get the number of rows and columns for the subplot grid
    num_rows = math.ceil(len(df.columns) / 3)
    num_cols = min(len(df.columns), 3)
    colors = sns.color_palette('Set2')
    # Create a grid of subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, facecolor='white')
    axes = axes.flatten()

    # Loop through the columns
    for i, column in enumerate(df.columns):
        counts = df[column].value_counts()
        counts_np = counts.to_numpy()

        # Create a pie plot on the current axis
        axes[i].pie(counts_np, labels=counts.index, startangle=0, autopct='%1.1f%%',
                radius=pie_size, colors=colors,)
        axes[i].set_title(column, fontsize=16)

    # Hide any remaining empty subplots
    for j in range(len(df.columns), num_rows * num_cols):
        axes[j].axis('off')

    # Adjust layout
    fig.subplots_adjust(wspace=.1, hspace=0.1)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_donut_charts(df, figsize=(15, 10), pie_size=1.2, title=""):
    """
    Plot pie charts for categorical columns in a DataFrame.

    Parameters:
        df (DataFrame): DataFrame containing the data to plot.
        figsize (tuple, optional): Figure size (width, height)
        pie_size (float, optional): Size of the pie chart.
    """
    # Get the number of rows and columns for the subplot grid
    num_rows = math.ceil(len(df.columns) / 3)
    num_cols = min(len(df.columns), 3)

    # Create a grid of subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()

    # Loop through the columns
    for i, column in enumerate(df.columns):
        counts = df[column].value_counts()

        # Create a pie plot on the current axis
        wedges, _ = axes[i].pie(counts, radius=pie_size, wedgeprops=dict(width=0.3))
        axes[i].set_title(column)

        # Add a legend
        axes[i].legend(wedges, counts.index,
                       title=title,
                       loc="center left",
                       bbox_to_anchor=(1, 0, 0.5, 1))

    # Hide any remaining empty subplots
    for j in range(len(df.columns), num_rows * num_cols):
        axes[j].axis('off')

    # Adjust layout
    plt.tight_layout()
    plt.show()


def plot_custom_histogram(data, x, bins=30, xlabel="", ylabel="", title="", palette="hls"):
    """
    Plot a histogram with custom colours.

    Parameters:
        data (DataFrame): DataFrame containing the data to plot.
        x (str): Column name for the x-axis.
        bins (int or sequence, optional): Specification of hist bins. Default is 30.
        xlabel (str, optional): Label for the x-axis. Default is "".
        ylabel (str, optional): Label for the y-axis. Default is "".
        title (str, optional): Title of the plot. Default is "".
        palette (str or sequence, optional): Palette to use for colouring the histogram bars. Default is "hls".
    """
    # Set the color palette
    cm = sns.color_palette(palette, bins)

    # Plot the histogram
    ax = sns.histplot(data, x=x, bins=bins)

    # Set labels and title
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)

    # Customise the histogram colours
    for bin_, color in zip(ax.patches, cm):
        bin_.set_facecolor(color)

    plt.show()


def label_encoding(df, columns):
    """
    Label encodes the set of the features to be used for imputation
    Args:
        df: data frame (processed data)
        columns: list (features to be encoded)
    Returns: dictionary
    """
    encoders = dict()
    for col_name in columns:
        series = df[col_name]
        label_encoder = LabelEncoder()
        df[col_name] = pd.Series(label_encoder.fit_transform(series[series.notna()]),
                                  index=series[series.notnull()].index)
        encoders[col_name] = label_encoder
    return encoders


def categorical_imputation(df, columns_to_impute, n_estimators=10, max_depth=70):
    """
    Perform categorical imputation using MissForest algorithm.
    
    Args:
        df (DataFrame): The DataFrame containing the data.
        columns_to_impute (list): List of columns to impute.
        n_estimators (int, optional): Number of trees in the forest. Defaults to 10.
        max_depth (int, optional): Maximum depth of the trees. Defaults to 70.
    
    Returns:
        DataFrame: DataFrame with imputed values.
    """

    # Label encode the specified columns
    encoders = label_encoding(df, columns_to_impute)

    # Initialize the MissForest imputer
    imp_cat = MissForest(n_estimators=n_estimators, max_depth=max_depth)

    # Perform categorical imputation using MissForest
    df[columns_to_impute] = imp_cat.fit_transform(df[columns_to_impute])

    # Decode the label encoded columns
    for column in columns_to_impute:
        df[column] = encoders[column].inverse_transform(df[column].astype(int))
    
    return df


def sql2csv(sqlfile):
    """
    This function parses an SQL dump file and converts each table in the dump into a separate CSV file.

    Parameters:
    sqlfile (str): The path to the SQL dump file.

    Returns:
    None
    """
    try:
        with open(sqlfile, 'r', encoding="utf-8") as sqldump:
            parser = sqlparse.parsestream(sqldump)
            headers = {}
            contents = collections.defaultdict(list)

            for statement in parser:
                if statement.get_type() == 'INSERT':
                    sublists = statement.get_sublists()
                    table_info = next(sublists)
                    table_name = table_info.get_name()

                    headers[table_name] = [
                        col.get_name()
                        for col in table_info.get_parameters()
                    ]

                    contents[table_name].extend(
                        tuple(
                            s.value.strip('"\'')
                            for s in next(rec.get_sublists()).get_identifiers()
                        )
                        for rec in next(sublists).get_sublists()
                    )

        data = {
            name: pd.DataFrame.from_records(table, columns=headers[name])
            for name, table in contents.items()
        }

        # Save each table to a separate CSV file
        for name, df in data.items():
            df.to_csv(f'{name}.csv', index=False)

    except FileNotFoundError:
        print(f"The file {sqlfile} does not exist.")
