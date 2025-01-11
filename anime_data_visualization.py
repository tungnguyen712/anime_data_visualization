"""
    Author: TUNG NGUYEN

    Date: 12/05/2024

    The goal of this project is to visualize datas from anime.csv
    to answer the following three questions:
    1. Is there a correlation between the number of members and the score an anime receives?
    2. Do longer total durations correlate with higher popularity and better scores?
    3. How does the combination of genres and themes influence an anime's likelihood of being favorite?

"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import ast
from numpy.polynomial.polynomial import Polynomial


def line_graph(filename):
    """
    Draw a scatterplot and a polynomial regression curve that show the correlation
    between the number of members and the corresponding score of anime series

    :param filename: (str) name of the file to be analyzed
    :return: None
    """
    df = pd.read_csv(filename)
    # remove rows when score has missing values
    df = df.dropna(subset=["score"])

    # filter out extreme outliers
    upper_limit = df["members"].quantile(0.95)
    lower_limit = df["members"].quantile(0.05)
    df = df[(df["members"] >= lower_limit) & (df["members"] <= upper_limit)]
    df = df.sample(n = 500)

    # process dataset's elements of the same columns into arrays
    members = np.array(df["members"])
    score = np.array(df["score"])

    # process x and y values to plot a polynomial regression
    p = Polynomial.fit(members,score,deg=3)
    x_smooth = np.linspace(members.min(),members.max(),500)
    y_smooth = p(x_smooth)

    # plot scatterplot and polynomial regression curve
    plt.xlabel("Members")
    plt.ylabel("Score")
    plt.title("Scatterplot and Polynomial Regression of Members and Score")
    plt.scatter(members, score,alpha=0.5,s=10,color="blue")
    plt.plot(x_smooth,y_smooth,color="red")
    plt.savefig('visualization1.png')
    plt.show()


def bar_chart(filename):
    """
    Plot a bar graph that show the correlation of the total durations
    with popularity and scores

    :param filename: (str) name of the file to be analyzed
    :return: None
    """
    df = pd.read_csv(filename)
    # delete NaN values in total_duration
    df = df.dropna(subset = "total_duration")

    # change dataset's elements of the same row to list
    durations = df["total_duration"].tolist()
    popularity = df["members"].tolist()
    score = df["score"].tolist()

    # change all durations units to minutes
    minute_lst = []
    # iterate over each time cluster
    for elem in durations:
        # check if time cluster is a str
        if isinstance(elem,str):
            lst = elem.split()
            smaller_lst = lst[2].split(':')
            minute = int(lst[0])*24*60 + int(smaller_lst[0])*60 + int(smaller_lst[1]) + int(smaller_lst[2])//60
            minute_lst.append(minute)

    # calculate quantile for duration
    upper_limit1 = np.quantile(minute_lst,0.99)
    lower_limit1 = np.quantile(minute_lst,0.01)

    # filter all list simultaneously accordingly to what is filtered in minute_lst
    filtered_data = []
    for m,p,s in zip(minute_lst,popularity,score):
        if lower_limit1 <= m <= upper_limit1:
            tup = (m,p,s)
            filtered_data.append(tup)

    # check if there is data after being filtered
    if filtered_data:
        # unpack list of tuples back to separate lists
        minute_lst, popularity, score = zip(*filtered_data)
        minute_lst = list(minute_lst)
        popularity = list(popularity)
        score = list(score)

    data = pd.DataFrame({'minute_lst':minute_lst,'popularity':popularity,'score':score})

    # group data by labels
    bins = [0,500,1000,1500,2000]
    labels = ['0-500','500-1000','1000-1500','1500-2000']
    data['duration'] = pd.cut(data['minute_lst'],bins=bins,labels=labels)
    grouped_data = data.groupby('duration',observed=True)[['popularity','score']].mean().reset_index()
    print(grouped_data)

    # set chart's attributes
    x = np.arange(len(grouped_data))
    width = 0.2
    fig,ax1 = plt.subplots()

    # create bar chart for popularity y-axis
    ax1.bar(x-width/2,grouped_data['popularity'],width,label='Popularity',color="orange")
    ax1.set_xlabel("Total Duration")
    ax1.set_ylabel("Popularity")
    ax1.tick_params(axis='y')
    ax1.set_xticks(x, grouped_data['duration'])
    ax1.set_title("Bar Chart of Total Durations with Popularity and Score")

    # create bar chart for score y-axis
    ax2 = ax1.twinx()
    ax2.bar(x+width/2,grouped_data['score'] ,width,label="Score",color='green')
    ax2.set_ylabel("Score")
    ax2.tick_params(axis='y')
    ax2.set_ylim(0,10)

    # combine legends
    bar1, label1 = ax1.get_legend_handles_labels()
    bar2, label2 = ax2.get_legend_handles_labels()
    ax1.legend(bar1 + bar2, label1 + label2)

    fig.tight_layout()
    plt.savefig('visualization2.png')
    plt.show()


def heat_map(filename):
    """
    Plot a heat map to show how the combination of genres and themes influence
    an anime's likelihood of being favorite?

    :param filename: name of the file to be analyzed
    :return: None
    """
    df = pd.read_csv(filename)
    genres = df["genres"].tolist()
    themes = df["themes"].tolist()
    favorites = df["favorites"].tolist()

    # create 2 dicts that keep track of the number of appearance
    genres_dict = {}
    themes_dict = {}
    for genre in genres:
        if not genre in genres_dict:
            genres_dict[genre] = 0
        else:
            genres_dict[genre] += 1
    for theme in themes:
        if not theme in themes_dict:
            themes_dict[theme] = 0
        else:
            themes_dict[theme] += 1

    # get a list of five most appeared keys of genres and themes from known values of dict
    # create and sort list of values from genres and themes dict
    genres_lst = list(genres_dict.values())
    themes_lst = list(themes_dict.values())
    genres_lst.sort()
    themes_lst.sort()
    dict1 = {}
    dict2 = {}

    # reverse keys and values
    for keys,values in genres_dict.items():
        dict1[values] = keys
    for keys,values in themes_dict.items():
        dict2[values] = keys

    # only take top 10 genres and themes
    genres = [dict1[x] for x in genres_lst if dict1[x] != '[]'][-15:]
    themes = [dict2[x] for x in themes_lst if dict2[x] != '[]'][-15:]

    # list of string to nested list
    genres = [ast.literal_eval(genre) for genre in genres]
    themes = [ast.literal_eval(theme) for theme in themes]

    # take each genre separately
    genres = [genre for lst in genres for genre in lst]
    themes = [theme for lst in themes for theme in lst]

    # created nested list of three variables with genres as keys,
    # themes as keys of values, and favorites as values of values
    datapoints = {}
    # iterate over genres, themes, and favorites simultaneously
    for genre,theme,favorite in zip(genres,themes,favorites):
        if not genre in datapoints:
            datapoints[genre] = {}
        if not theme in datapoints:
            datapoints[genre][theme] = 0
        datapoints[genre][theme] += favorite

    # plot heatmap and set attributes
    df = pd.DataFrame(datapoints).fillna(0)
    sns.heatmap(df,cmap='PiYG',center=0,cbar_kws={'label':'Favorites'})
    plt.title("Heatmap of Favorites by Genres and Themes")
    plt.xlabel("Genres")
    plt.ylabel("Themes")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('visualization3.png')
    plt.show()


def main():
    filename = "anime.csv"
    # answer to first question
    line_graph(filename)
    # answer to second question
    bar_chart(filename)
    # answer to third question
    heat_map(filename)

if __name__ == "__main__":
    main()