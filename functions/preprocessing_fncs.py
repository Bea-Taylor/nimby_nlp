import pandas as pd
import numpy as np
import re

def format_borough(df, borough_col_name='borough'):
    """Standardizes borough names by removing redundant text and normalizing naming.
    
    Args:
        df (DataFrame): Dataframe containing borough data.
        borough_col_name (str, optional): Name of the borough column. Defaults to 'borough'.
        
    Returns:
        DataFrame: Dataframe with formatted borough names.
    """

    # Ensure values are strings to avoid type chaos
    df[borough_col_name] = df[borough_col_name].astype(str)

    # Replacement rules, folded together
    replacements = {
        'London Borough of ': '',
        'Royal Borough of ': '',
        '(LA Code)': '',
        'Bromley Custodian Code': 'Bromley',
        'Council': '',
        '&': 'and',
        'Custodian code': 'nan',
        'Out of Borough': 'Haringey',
        'And': 'and',
        'City Of London': 'City of London',
        'Richmond': 'Richmond upon Thames',
        'Kingston': 'Kingston upon Thames'
    }

    # Apply all replacements in a vectorized way
    for old, new in replacements.items():
        df[borough_col_name] = df[borough_col_name].str.replace(old, new, regex=False)

    # Str cleanup: - remove trailing spaces and fix double spaces formed by replacement
    df[borough_col_name] = df[borough_col_name].str.strip().str.replace(r'\s+', ' ', regex=True)

    return df



def _format_dateime(df):
    """Formats the date columns in the dataframe to datetime format.

    Args:
        df (_type_): Dataframe to be processed.

    Returns:
        _type_: Processed dataframe.
    """

    for col in df.columns:
        if re.findall(r'_date', col):
            df[col] = pd.to_datetime(df[col], format='%d/%m/%Y', errors='coerce')

    return df



def format_df(df, drop_cols=True, borough_col_name='borough'):
    """Cleans dataframe column names, formats date fields, and standardizes borough naming.
    
    Args:
        df (DataFrame): Dataframe to be processed.
        drop_cols (bool, optional): Whether to drop metadata columns. Defaults to True.
        borough_col_name (str, optional): Name of the borough column. Defaults to 'borough'.
        
    Returns:
        DataFrame: Processed dataframe.
    """

    # Vectorized cleanup of column names
    remove_prefixes = ['_source.', 'application_details.', 'residential_details.']
    for prefix in remove_prefixes:
        df.columns = df.columns.str.replace(prefix, '', regex=False)

    # Apply datetime formatter if exists 
    df = _format_dateime(df)

    # Format borough column if present
    if borough_col_name in df.columns:
        df = format_borough(df, borough_col_name=borough_col_name)

    # Drop metadata columns if requested
    if drop_cols:
        drop_columns = ['_index', '_type', '_id', '_score', '_ignored']
        df.drop(columns=[c for c in drop_columns if c in df.columns], inplace=True)

    return df



def __match_decisions(df, decision_col_name='decision', status_col_name='status'):
    """ Matches the decision column to the status column in the dataframe.

    Args:
        df (df): dataframe to be processed.
        decision_col_name (str, optional): decision column. Defaults to 'decision'.
        status_col_name (str, optional): status column. Defaults to 'status'.

    Returns:
        df: Same dataframe, with decision column updated using information from the status column.
    """

    expected = ['Approved', 'Refused', 'Withdrawn', 'Superseded', 'Closed']

    for i, row in df.iterrows():
        if row[status_col_name] in expected and row[decision_col_name] not in expected:
            df.at[i, decision_col_name] = row[status_col_name]
    
    return df



def format_decisions(df, decision_col_name='decision', status_col_name='status'):
    """ Formats the decision and status columns in the dataframe to remove unnecessary words and characters.

    Args:
        df (df): dataframe to be processed.
        decision_col_name (str, optional): decision column. Defaults to 'decision'.
        status_col_name (str, optional): status column. Defaults to 'status'.   

    Returns:
        df: Same dataframe, with decision column and status column updated.
    """

    decisions = df[decision_col_name].unique()
    decisions = decisions.astype(str)

    new_d = {}
    for d in decisions:
        n = d.lower()
        n = n.rstrip()
        if re.findall('\Aapprov|\Aaprov|\Aappprov', n):
            n = 'approved'
        if re.findall('\Arefu', n):
            n = 'refused'
        if re.findall('\Awith', n):
            n = 'withdrawn'

        n = n.replace('allowed', 'approved')
        n = n.capitalize()

        new_d[d] = n

    df[decision_col_name] = df[decision_col_name].map(new_d)

    if status_col_name in df.columns:
        df[status_col_name] = df[status_col_name].map(new_d)

    df = __match_decisions(df, decision_col_name, status_col_name)

    return df



def create_decision_outcomes(df, outcome_col_name='outcome', decision_col_name='decision', status_col_name='status', completion_date_col_name='actual_completion_date'):
    """ Creates a new column in the dataframe with the outcome of the decision. Specifically this column records whether each application was 
    permitted or not permitted, by taking into account the decision, status and completion date columns.

    Args:
        df (df): dataframe to be processed.
        outcome_col_name (str, optional): outcome column. Defaults to 'outcome'.
        decision_col_name (str, optional): decision column. Defaults to 'decision'.
        status_col_name (str, optional): status column. Defaults to 'status'.
        completion_date_col_name (str, optional): completion date column. Defaults to 'actual_completion_date'.

    Returns:
        df: Same dataframe, with a new column outcome_col_name added
    """

    df[outcome_col_name] = df[decision_col_name]

    # format decision and status columns
    df['d_format'] = df[decision_col_name].replace(np.nan, 'nan')
    df['s_format'] = df[status_col_name].replace(np.nan, 'nan')
    df['d_format'] = df['d_format'].fillna('nan')
    df['s_format'] = df['s_format'].fillna('nan')

    # lower case all strings
    decision_lower = [d.lower() for d in df['d_format']]
    status_lower = [d.lower() for d in df['s_format']]

    df['d_format'] = decision_lower
    df['s_format'] = status_lower

    # define categories
    permitted = ['approved', 'not required', 'commenced', 'allowed', 'granted', 'permitted', 'consent', 'permit']
    ongoing = ['opinion issued', 'comment issued', 'called in by secretary of state', 'pre-application advice case completed', 'referred to mayor']
    superseded = ['superseded', 'superseded by new application']
    not_permitted = ['refused']
    withdrawn = ['withdrawn', 'withdrawn by applicant']

    # assign outcome to each application
    for i, row in df.iterrows():
        if row['d_format'] in superseded or row['s_format'] in superseded:
            df.at[i, outcome_col_name] = 'Superseded'
        elif row['d_format'] in permitted or row['s_format'] in permitted:
            df.at[i, outcome_col_name] = 'Permitted'
        elif pd.isnull(row[completion_date_col_name]) == False: 
            df.at[i, outcome_col_name] = 'Permitted'
        elif row['d_format'] in not_permitted or row['s_format'] in not_permitted:
            df.at[i, outcome_col_name] = 'Not Permitted'
        elif row['d_format'] in ongoing or row['s_format'] in ongoing:
            df.at[i, outcome_col_name] = 'Ongoing'
        elif row['d_format'] in withdrawn or row['s_format'] in withdrawn:
            df.at[i, outcome_col_name] = 'Withdrawn'
        else:
            df.at[i, outcome_col_name] = 'Other'

    # drop temporary columns used for formatting
    df.drop(columns=['d_format', 's_format'], inplace=True)

    return df



def add_housing_type(df):
    """ Adds a new column to the dataframe indicating the type of housing proposed in each application.
    Args:
        df (df): dataframe to be processed. 

    Returns:
        df: Same dataframe, with a new column 'housing_type' added.
    """

    # Define housing type categories based on the number of proposed residential units in the PLD 
    self_build = 'total_no_proposed_residential_units_self_build_and_custom_build'

    social_housing = 'total_no_proposed_residential_units_social_rent'

    affordable_rent = ['total_no_proposed_residential_units_london_living_rent',
                    'total_no_proposed_residential_units_discount_market_rent_charged_at_london_rents',
                    'total_no_proposed_residential_units_london_affordable_rent',
                    'total_no_proposed_residential_units_discount_market_rent']

    affordable_sale = ['total_no_proposed_residential_units_shared_equity',
                    'total_no_proposed_residential_units_london_shared_ownership',
                    'total_no_proposed_residential_units_discount_market_sale',
                    'total_no_proposed_residential_units_starter_homes']

    market_rent = 'total_no_proposed_residential_units_market_for_rent'

    market_sale = 'total_no_proposed_residential_units_market_for_sale'

    df['self_build'] = pd.to_numeric(df[self_build])
    df['social_housing'] = pd.to_numeric(df[social_housing])
    df['affordable_rent'] = pd.to_numeric(df[affordable_rent].sum(axis=1))
    df['affordable_sale'] = pd.to_numeric(df[affordable_sale].sum(axis=1))
    df['market_rent'] = pd.to_numeric(df[market_rent])
    df['market_sale'] = pd.to_numeric(df[market_sale])

    # Create 'housing_type' column
    for i, row in df.iterrows():
        if row['social_housing'] > 0:
            if row['total_no_proposed_residential_units'] == row['social_housing']:
                df.at[i, 'housing_type'] = 'Social housing'
            else:
                df.at[i, 'housing_type'] = 'Mixed social housing'
        elif row['affordable_rent'] > 0 or row['affordable_sale'] > 0:
            df.at[i, 'housing_type'] = 'Mixed affordable housing'
        elif row['market_rent'] > 0 or row['market_sale'] > 0:
            df.at[i, 'housing_type'] = 'Market housing'
        elif row['self_build'] > 0:
            df.at[i, 'housing_type'] = 'Self-build housing'
        else:
            df.at[i, 'housing_type'] = 'Other'
    
    return df