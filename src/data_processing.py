import collections
# import sqlparse
import cupy as cp
import pandas as pd


def count_missing_values(data):
    """
    Analyze and report the missing values in a pandas DataFrame or Series.

    This function calculates the number of missing values for each column or the single column 
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

    data = data.fillna(cp.nan)

    if not isinstance(data, (pd.DataFrame, pd.Series)) or data.empty:
        raise ValueError(
            'Input must be a non-empty pandas DataFrame or a non-empty pandas Series')

    # Calculate missing values
    if isinstance(data, pd.Series):
        missing_count = data.isnull().sum()
        percentage_missing = (missing_count / data.shape[0]) * 100
        report = pd.DataFrame(
            {'Missing count': [missing_count], 'Percentage': [percentage_missing]})
    else:
        missing_count = data.isnull().sum().sort_values(ascending=False)
        percentage_missing = (missing_count / data.shape[0]) * 100
        report = pd.DataFrame(
            {'Missing count': missing_count, 'Percentage': percentage_missing})

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


def compute_mad_1d(x, k=[2.5,2.5]):
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

