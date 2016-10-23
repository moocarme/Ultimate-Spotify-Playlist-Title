# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 20:39:47 2016

@author: matt-666
"""

# Import libraries
import spotipy
import spotipy.util as util
from pickle import dump
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ===========================================

class spotify_analysis(object):
    
    def __init__(self, sp):
        self.sp = sp
        
    def get_playlist_info(self):
        '''
        Get info on playlist ids, owners, metadat and categories
        '''
        limit = 50 # use max limit
        
        categories = self.sp.categories(limit = limit) # get categories
        
        # Get list of category ids
        ids = [] # initialize
        for cat_id in (categories['categories']).items()[0][1]:
            ids.append(cat_id['id'])
        
        # Initialise lists and dict
        self.playlist_ids, self.playlist_owners = [], []
        self.playlist_metadata, self.playlist_categories = {}, []
        # Get playlist IDs, owners and metadata 
        for category_id in ids:
            all_playlists = self.sp.category_playlists(category_id = category_id, 
                                                  limit = limit)
            # go through all playlist items
            for playlist in all_playlists['playlists']['items']:
                self.playlist_ids.append(playlist['id'])             # get id
                self.playlist_owners.append(playlist['owner']['id']) # get owner
                self.playlist_metadata[playlist['name']] = playlist  # get metadata
                self.playlist_categories.append(category_id)
                
        # Initialise lists
        self.playlist_descs, self.playlist_names, self.playlist_followers = [], [], []
        
        # Iterate through playlists and get all relevent info
        for playlist_id, playlist_owner in zip(self.playlist_ids, self.playlist_owners):
            the_playlist = self.sp.user_playlist(user = playlist_owner, 
                                            playlist_id = playlist_id)
            self.playlist_names.append(the_playlist['name'])        # get name
            self.playlist_descs.append(the_playlist['description']) # get description
            self.playlist_followers.append(the_playlist['followers']['total']) # get total followers
        
        # find mean number of followers per owner
        df_dict = {'Name' : self.playlist_names, 
                   'Owner' : self.playlist_owners,
                   'followers' : self.playlist_followers,
                   'category': self.playlist_categories}
                   
        self.playlist_dataframe = pd.DataFrame(df_dict)
        return self.playlist_dataframe
        
    def create_wordcloud(self):
        '''
        Make wordcloud from playlist titles
        '''
        all_names = ' '.join(self.playlist_names)
        wordcloud = WordCloud(width = 700, height = 300, 
                              max_font_size=90, relative_scaling=.7).generate(all_names)
        plt.figure()
        plt.imshow(wordcloud)
        plt.axis('off')

        self.playlist_descs = ['' if desc is None else desc for desc in self.playlist_descs]
        all_descs = ' '.join(self.playlist_descs)
        wordcloud_descs = WordCloud().generate(all_descs)
        plt.figure()
        plt.imshow(wordcloud_descs)
        plt.axis('off')
        
    def plot_followers_by_owner(self):
        '''
        Plots a bar graph of the mean followers by playlist owner
        '''
        mean_followers_per_user = self.playlist_dataframe.groupby(['Owner']) \
                                                         .mean() \
                                                         .sort('followers', 
                                                               ascending = False)
        plt.figure()
        plt.bar(range(len(mean_followers_per_user)), mean_followers_per_user['followers'])

    def plot_followers_by_category(self):
        '''
        Plots a bar graph of the mean followers by category
        '''
        mean_followers_per_category = self.playlist_dataframe.groupby(['category']) \
                                                             .mean() \
                                                             .sort('followers', 
                                                                   ascending = False)
        plt.figure()
        plt.bar(range(len(mean_followers_per_category)), 
                np.round(mean_followers_per_category['followers']))

# =============================================

# API details and log in ======================
scope = 'user-library-read'
token = util.prompt_for_user_token('Matt Moocarme', scope, 
                                   client_id = '-client-id-', 
                                   client_secret = '-client-secret-',
                                   redirect_uri = 'https://www.google.com')

sp = spotipy.Spotify(auth = token)

analysis = spotify_analysis(sp)
playlist_dataframe = analysis.get_playlist_info()
analysis.create_wordcloud()
analysis.plot_followers_by_owner()
analysis.plot_followers_by_category()

# dump in pickle file ====================
with open('playlistData.p', 'wb') as pickleFile:
    dump((playlist_dataframe), pickleFile)
