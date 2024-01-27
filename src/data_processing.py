import collections
# import sqlparse
import cupy as cp
import pandas as pd


def missing_values(df):
    """
    Analyze and report the missing values in a pandas DataFrame.

    This function takes a pandas DataFrame as input, validates if it is a valid dataframe 
    with at least one column, calculates the number of missing values for each column 
    along with their percentages relative to total rows and returns this information in a new dataframe.

    Parameters:
        df (pandas DataFrame): Input DataFrame to be analyzed.

    Returns:
        pandas DataFrame: A report containing the missing value count and percentage
        for each column, sorted by count descending.

    Raises:
        TypeError: If input is not a pandas DataFrame.
        ValueError: If input DataFrame has no columns.
    """

    # Validate inputs
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Input must be a pandas DataFrame')

    if df.shape[1] == 0:
        raise ValueError('Input DataFrame has no columns')

    # Calculate missing values
    missing_count = df.isnull().sum().sort_values(ascending=False)
    percentage_missing = (missing_count / df.shape[0]) * 100

    report = pd.concat([missing_count, percentage_missing],
                       axis=1, keys=['Missing count', 'Percentage'])
    return report


def compute_mad_1d(x):
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
    mad = cp.median(cp.abs(x - med))
    # constant b using Ley (2023)
    # b = 1. / cp.quantile(x, 0.75)
    return mad * 1.486


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

