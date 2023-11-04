from pyspark.sql import SparkSession # Spark SQL
spark = SparkSession.builder.master("local[*]").getOrCreate() # Spark Session - to create Spark DataFrames

# import libraries
from scipy import stats
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline
from pyspark.sql.functions import *
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType, DoubleType, LongType
import random
import time 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns
import warnings
import json
from datetime import datetime
from statistics import mode

# [RQ1] Function to obtain the type of the columns
def count_columns_type(data):
    categorial = 0
    numeric = 0
    other = 0
    for col in data.columns:
        c_type = data.schema[col].dataType
        #print(c_type)
        if isinstance(c_type, StringType):
            categorial += 1
            
        elif isinstance(c_type, LongType) or isinstance(c_type,DoubleType):
            numeric += 1
        else:
            other += 1
    print("The number of categorial columns is: ", categorial)
    print("The number of numeric columns is: ", numeric)
    print("The number of other types of columns is: ", other)

# [RQ2] Function divide the data in 4 parts
def division_data(data,n):
    interval_len = len(data) // n
    divisions = []
    start = 0
    for i in range(n):
        end = start + interval_len
        divisions.append(data.iloc[start:end])
        start = end
    # if there are any rows left, concat them with the last chunk
    if start < len(data):
        divisions[-1] = pd.concat([divisions[-1], data.iloc[start:]])
    return divisions

# [RQ3.1] `yearly_stats` function to get yearly stats for each year in the `df_lighter_books` dataframe
def yearly_stats(df_lighter_books,year):
    # Filter books for given year
    df_year = df_lighter_books.filter(df_lighter_books['year'] == year)

    # Number of books published
    num_books = df_year.count()

    # Total number of pages
    total_pages = df_year.agg(F.sum('num_pages')).first()[0]

    # Most prolific month
    prolific_month = df_year.groupBy('month').count().orderBy(F.desc('count')).first()[0]

    # Longest book
    longest_book = df_year.orderBy(F.desc('num_pages')).first()['title']

    return (year, num_books, total_pages, prolific_month, longest_book)

# [RQ3.2] ChatGPT implementation of the `yearly_stats` function to get yearly stats for each year in the `df_lighter_books` dataframe
def gpt_yearly_stats(df_lighter_books_pd,year):
    # Filter books for given year
    df_year = df_lighter_books_pd[df_lighter_books_pd['year'] == year]

    # Number of books published
    num_books = df_year.shape[0]

    # Total number of pages
    total_pages = df_year['num_pages'].sum()

    # Most prolific month
    prolific_month = df_year['month'].value_counts().idxmax()

    # Longest book
    longest_book = df_year.loc[df_year['num_pages'].idxmax()]['title']

    return (year, num_books, total_pages, prolific_month, longest_book)

# [RQ4.2] A function that, given a list of `author_id`, outputs a dictionary where each `author_id` is a key, and the related value is a list with the names of all the books the author has written.
def get_author_books(df_lighter_books,author_ids):

    # Filter df_authors on the provided author_ids
    df_author_books = df_lighter_books.filter(df_lighter_books['author_id'].isin(author_ids))
    
    # Group by author id and collect all book titles
    df_author_books = df_author_books.groupBy("author_id").agg(collect_list("title").alias("books"))
    
    # Convert to Python dictionary
    author_books_dict = {row['author_id']: row['books'] for row in df_author_books.collect()}
    
    return author_books_dict

# [RQ7.1] A function that, given the `df_rating_dist` dataframe, preprocesses it, and returns a new dataframe where each row contains the ratings distribution for a specific book.`
def preprocess(df):
    # Split the rating distribution into separate columns
    df = df['rating_dist'].str.split('|', expand=True)

    # For each column, split on ':' and keep only the count (index 1)
    for col in df.columns:
        df[col] = df[col].str.split(':', expand=True)[1]

    # Rename the columns to match the ratings they represent
    df.columns = ['rating_5', 'rating_4', 'rating_3', 'rating_2', 'rating_1', 'total']
    
    # Convert string to numeric
    df = df.apply(pd.to_numeric, errors='coerce')
    
    return df