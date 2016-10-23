# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 21:15:07 2016

@author: matt-666
"""

# Import libraries
import pickle
import nltk
import pandas as pd
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
import numpy as np 
import matplotlib.pyplot as plt
import scipy.stats as sp


# =================================================


class playlist_analysis(object):
    
    def __init__(self, playlist_names, playlist_descs, playlist_followers,
                 playlist_ids, playlist_owners, playlist_metadata):
        
        self.playlist_names = playlist_names
        self.playlist_descs = playlist_descs
        self.playlist_followers = playlist_followers
        self.playlist_ids = playlist_ids
        self.playlist_owners = playlist_owners
        self.playlist_metadata = playlist_metadata
            
        # remove null values
        self.playlist_followers = [0 if number is None else number for number in self.playlist_followers]

    def create_dataframe(self):
        '''
        nltk.help.upenn_tagset() # list of tags
        Create dataframe
        '''
        self.totdf = pd.DataFrame(columns = ('CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR',
                                         'JJS', 'LS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 
                                         'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RP', 
                                         'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 
                                         'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', 
                                         'Num_Words', 'Num_Chars'))

        # Create dict from analysing playlist titles
        self.word_ref = {}
        for playlist_name in self.playlist_names:
            numchars = len(playlist_name) # get number of chars in playlist name
            numwords = len(playlist_name.split()) # get number of words in playlist name
            text = nltk.word_tokenize(playlist_name.lower()) # tokenise 
            tags = nltk.pos_tag(text)
            
            # get just the word type tags, i.e., VB for verb
            type_list = []
            for (word, word_type) in tags:
                type_list.append(word_type)
                try:
                    tmp_list = self.word_ref[word_type]  # get list from dictionary 
                    tmp_list.append(word)           # append to list
                    self.word_ref[word_type] = tmp_list  # put back in dictionary
                except AttributeError:
                    self.word_ref[word_type] = []
                except KeyError:
                    self.word_ref[word_type] = []
                    
            # count each instance and insert to dict
            word_type_count = {}
            for word_type in list(self.totdf.columns.values[:-2]):
                word_type_count[word_type] = (type_list.count(word_type))
        
            word_type_count['Num_Words'] = numwords
            word_type_count['Num_Chars'] = numchars
            
            # append to dict
            self.totdf = self.totdf.append([word_type_count])

    def word_count(self):
        '''
        Count instance of each word
        '''
        self.word_type_count_type, self.word_type_count_list = [], []
        for key, value in self.word_ref.iteritems():
            self.word_type_count_type.append(key)
            self.word_type_count_list.append(len(value))
            
        # Dependence on number of characters with numer of followers
        sp.ttest_ind(self.totdf['Num_Chars'], self.playlist_followers)
        # p-value less than 0.05

    def plot_char_dependence(self):
        '''
        Look at dependence of number of chars and followers
        '''
        plt.figure()
        plt.scatter(self.totdf['Num_Chars'], self.playlist_followers)
        slope, intercept, r_val, p_val, slope_std_error = sp.linregress(self.totdf['Num_Chars'], playlist_followers)
        y_predict = intercept + self.totdf['Num_Chars']*slope
        plt.plot(self.totdf['Num_Chars'], y_predict)
        plt.xlim(0, 80); plt.xlabel('Number of Characters', fontsize = 20)
        plt.ylim(-0.1e7, 1.1e7); plt.ylabel('Number of Followers', fontsize = 20)

    def regression_setup(self, test_size = 0.2, seed = 666):
        '''
        Split into test-train
        '''
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.totdf, 
                                                                                self.playlist_followers,
                                                                                test_size=test_size, 
                                                                                random_state=seed)
    
    def linregress(self):
        '''
        Train with Linear regression
        '''
        LinRegress = linear_model.LinearRegression()
        LinRegress.fit(self.X_train, self.y_train) 
        print('Residual sum of squares is %.5e' % np.mean((LinRegress.predict(self.X_test) 
                                                          - self.y_test) ** 2))

    def ridge(self, alphas = np.logspace(1, 3, 60)):
        '''
        Train with ridge regression 
        '''
        train_errors_rr, test_errors_rr = [], []
        for alpha in alphas:
            clf = linear_model.Ridge(alpha = alpha)
            clf.fit(self.X_train, self.y_train) 
            train_errors_rr.append(clf.score(self.X_train, self.y_train))
            test_errors_rr.append(clf.score(self.X_test, self.y_test))
        
        alpha_optim_rr = alphas[np.argmax(test_errors_rr)]
        clf.set_params(alpha = alpha_optim_rr)
        clf.fit(self.X_train, self.y_train)
        print("Optimal regularization parameter : %s" % alpha_optim_rr)
        print('Residual sum of squares is %.5e' % np.mean((clf.predict(self.X_test) - self.y_test) ** 2))
        # 1.57414e+11
        
        plt.rc('ytick', labelsize = 20)
        plt.rc('xtick', labelsize = 20)
        plt.figure()
        plt.semilogx(alphas, train_errors_rr, label = 'Train Score', linewidth = 3)
        plt.semilogx(alphas, test_errors_rr, label = 'Test score', linewidth = 3)
        plt.semilogx(alpha_optim_rr, max(test_errors_rr),'o', label = 'Optimised')
        plt.xlabel('alpha', fontsize = 20); plt.ylabel('Score', fontsize = 20)
        plt.legend(loc = 2, fontsize = 20); 
        plt.title('Ridge Regression Scores', fontsize = 20)

    def lasso(self, alphas = np.logspace(4, 7, 60)):
        '''
        Train with Lasso regression 
        '''
        train_errors_l, test_errors_l = [], []
        for alpha in alphas:
            las = linear_model.Lasso(alpha = alpha)
            las.fit(self.X_train, self.y_train) 
            train_errors_l.append(las.score(self.X_train, self.y_train))
            test_errors_l.append(las.score(self.X_test, self.y_test))
        
        alpha_optim_l = alphas[np.argmax(test_errors_l)]
        las.set_params(alpha = alpha_optim_l)
        las.fit(self.X_train, self.y_train)
        print("Optimal regularization parameter : %s" % alpha_optim_l)
        print('Residual sum of squares is %.5e' % np.mean((las.predict(self.X_test) - self.y_test) ** 2))
        # 1.55680e+11
        
        plt.figure()
        plt.semilogx(alphas, train_errors_l, label = 'Train Score', linewidth = 3)
        plt.semilogx(alphas, test_errors_l, label = 'Test score', linewidth = 3)
        plt.semilogx(alpha_optim_l, max(test_errors_l),'o', label = 'Optimised')
        plt.xlabel('alpha', fontsize = 20); plt.ylabel('Score', fontsize = 20)
        plt.title('Lasso Scores', fontsize = 20)
        plt.legend(fontsize = 20)

        
    def enet(self, alphas = np.logspace(-1, 3, 60), ratios = np.linspace(0, 1, 20)):
        '''
        Train with elastic net
        '''
        train_errors, test_errors = [],[]
        
        # iterate through parameters
        for ratio in ratios:
            enet = linear_model.ElasticNet(l1_ratio=ratio)
            alpha_train_errors, alpha_test_errors = [], []
            for alpha in alphas:
                enet.set_params(alpha=alpha)
                enet.fit(self.X_train, self.y_train)
                alpha_train_errors.append(enet.score(self.X_train, self.y_train))
                alpha_test_errors.append(enet.score(self.X_test, self.y_test))
            train_errors.append(alpha_train_errors)
            test_errors.append(alpha_test_errors)
            
        i_alpha_ratio_optim = np.unravel_index(np.array(test_errors).argmax(), 
                                            np.array(test_errors).shape) # max because retuerns R^2 value 
        ratio_optim = ratios[i_alpha_ratio_optim[0]]
        alpha_optim = alphas[i_alpha_ratio_optim[1]]
        print("Optimal ratio parameter : %s" % ratio_optim)
        print("Optimal regularization parameter : %s" % alpha_optim)

        # Estimate the coef_ on full data with optimal regularization parameter
        enet.set_params(alpha=alpha_optim, l1_ratio = ratio_optim)
        enet.fit(self.X_train, self.y_train)
        print('Residual sum of squares is %.5e' % np.mean((enet.predict(self.X_test) - self.y_test) ** 2))
        #  1.57383e+11
        
        plt.figure(112); plt.clf()
        for i in range(int(len(ratios)/2)):
            plt.semilogx(alphas, np.array(test_errors)[2*i,:], 
                         label = 'Ratio:' +str(round(ratios[2*i], 4)),
                         color = plt.cm.RdYlBu(ratios[2*i]),
                         linewidth = 3)
        plt.legend(loc = 2, fontsize = 16)
        plt.xlabel('alpha', fontsize = 20); plt.ylabel('Score', fontsize = 20)
        plt.title('Elastic Net Test Scores', fontsize = 20)


        # Explain coefficients 
        colnames = list(self.totdf.columns.values)
        sorted_colnames = [x for (y,x) in sorted(zip(enet.coef_,colnames))]
        sorted_coefs = sorted(enet.coef_)
        
        plt.figure(666); plt.clf()
        plt.bar(range(len(sorted_coefs)), sorted_coefs)
        plt.xticks(range(len(sorted_coefs)),sorted_colnames)
        
# ====================================

# get data from pickle file ========================
with open(r"playlistData.p", "rb") as input_file:
    e = pickle.load(input_file)

playlist_names, playlist_descs, playlist_followers = e[0], e[1], e[2]
playlist_ids, playlist_owners, playlist_metadata  = e[3], e[4], e[5]

# initialise model
analysis = playlist_analysis(playlist_names, playlist_descs, playlist_followers, 
                             playlist_ids, playlist_owners, playlist_metadata)

analysis.create_dataframe() # create dataframe of features
analysis.word_count()       # count instances of words
analysis.plot_char_dependence() # plot dependece on number of characters
analysis.regression_setup(test_size = 0.2, seed = 666) # setup for regression
analysis.linregress()       # perform linear regression
analysis.ridge(alphas = np.logspace(1, 3, 60)) # ridge regression
analysis.lasso(alphas = np.logspace(4, 7, 60)) # LASSO regression
analysis.enet(alphas = np.logspace(-1, 3, 60), ratios = np.linspace(0, 1, 20)) # elastic net