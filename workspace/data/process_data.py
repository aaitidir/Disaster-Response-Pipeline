import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Load the data from the csv's and return a combined dataframe
    Params:
      messages_filepath(string): File path of the messages csv
      categories_filepath(string) : File path of the categories csv
    Return:
      df(dataframe):  dataframe of the combined files
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    categories_id = categories["id"]

    df = messages.join(categories.set_index('id'), on='id')    
    return df

def clean_data(df):
    '''
    Clean the df dataframe and return a dataframe with the messages and numerical values for the categories
    
    Params:
      df(dataframe): Messages and categories
    Return:
      df(dataframe): cleaned dataframe
    '''
<<<<<<< HEAD
    # create a dataframe of the 36 individual category columns
    categories = df["categories"].str.split(";",expand=True)
    row = categories.loc[:0]
    # use this row to extract a list of new column names for categories
    category_colnames = row.apply(lambda x: x.str[0:-2], axis=0)
    # rename the columns of `categories`
    categories.columns = category_colnames.values[0]
    # set each value to be the last character of the string
    # and convert column from string to numeric
=======
    categories = df["categories"].str.split(";",expand=True)
    row = categories.loc[:0]
    category_colnames = row.apply(lambda x: x.str[0:-2], axis=0)
    categories.columns = category_colnames.values[0]
    
>>>>>>> bcff0b7d86df792fae0b6e8cd5b7643ee8041129
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1])
        categories[column] = categories[column].astype('int32')
    
<<<<<<< HEAD
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    #Drop the categories column from the df dataframe since it is no longer needed.
    df.drop(columns=['categories'],inplace=True)
    # drop duplicates
=======
    df = pd.concat([df, categories], axis=1)
    df.drop(columns=['categories'],inplace=True)
>>>>>>> bcff0b7d86df792fae0b6e8cd5b7643ee8041129
    df.drop_duplicates(inplace=True)
    return df

def save_data(df, database_filename):
    '''
    Save the df dataframe and to a database
    
    Params:
      df(dataframe): dataframe
      database_filename(string) : File path for the database
    Return:
    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('etl_pipeline', engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
<<<<<<< HEAD
    main()
=======
    main()
>>>>>>> bcff0b7d86df792fae0b6e8cd5b7643ee8041129
