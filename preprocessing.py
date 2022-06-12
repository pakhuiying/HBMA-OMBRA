import numpy as np
import pandas as pd
from os.path import join
from os import listdir
import glob
import re
from  itertools import combinations
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib.colors as pltcolors
import seaborn as sns
from collections import Counter
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

class LoadData:
    """
    creates a band ratio df object
    """
    def __init__(self, fp):
        """
        fp (str): filepath to where TSS_df_avg.csv is stored
        """
        self.fp = fp
        TSS_df_avg = pd.read_csv(self.fp)
        TSS_pivot = TSS_df_avg.pivot_table(
            values = 'Reflectance',
            index=['Type','Concentration','Wave','Sensor_Angle','Azimuth','Altitude'],
            columns='Wavelength'
        )
        self.df = TSS_pivot.reset_index()
        self.wavelengths = self.df.columns[6:].tolist()

    def create_band_ratio_df(self,store_directory):
        """
        store_directory (str): directory where the csv file will be saved
        Creates a dataframe where all the bands are transformed to every possible band-ratio combination
        file is saved to the directory
        """
        df_attributes = self.df.iloc[:,0:6]
        df_covariates = self.df.iloc[:,6:]
        df_upper = pd.concat([df_covariates[a].div(df_covariates[b]).rename(f'{a}/{b}') for a, b in combinations(df_covariates.columns, 2)], 1)        
        return pd.concat([df_attributes,df_upper],axis=1).to_csv(join(store_directory,'TSS_band_ratio.csv'),index=False)

    def plot_spectral_curve(self):
        """
        outputs a plot of reflectance vs wavelength, with concentration as different colours
        """
        select_cols = ['Concentration'] + self.wavelengths
        grouped = self.df[select_cols].groupby('Concentration').mean()
        transpose_group = grouped.transpose()
        transpose_group.plot(figsize=(12,6),
        logy=True,
        colormap='Spectral_r',#_r represents reverse
        xlabel='Wavelength (nm)',
        ylabel="Reflectance (%)"
        )
        plt.legend(title="Concentration (mg/l)",bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()
        return

    def band_ratio(self,df,test_size=0.3):
        """input takes a df and a test_size (ranges from 0 to 1)
        last 61 columns are the wavelengths
        contains a column named 'Concentration'
        outputs a df of upper triangle matrix where columns and indexes are wavelengths. 
        """
        # wavelength_cols = df.reset_index().set_index('Concentration').iloc[:,-61:]
        wavelength_cols = df.iloc[:,-61:]
        X = wavelength_cols
        y = df['Concentration']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)

        #compute all combinations of wavelength ratios:
        col_combinations = [(a,b) for a, b in combinations(X_train.columns, 2)]
        upper_tri_train = pd.concat([np.log(X_train[a].div(X_train[b]).rename(f'{a}/{b}')) for a, b in col_combinations], 1)
        upper_tri_test = pd.concat([np.log(X_test[a].div(X_test[b]).rename(f'{a}/{b}')) for a, b in col_combinations], 1)
        
        n = len(self.wavelengths)
        R2_matrix = np.zeros((n,n))
        #compute upper triangle R2
        count = 0
        for i in range(0,n): #rows
            for j in range(i+1,n): #columns
                reg = LinearRegression().fit(upper_tri_train.iloc[:,count:count+1],y_train) #regression model
                y_pred =reg.predict(upper_tri_test.iloc[:,count:count+1])
                r2 = r2_score(y_test,y_pred)
                R2_matrix[i,j] = r2
                count += 1

        r2_df = pd.DataFrame(R2_matrix,index=self.wavelengths,columns=self.wavelengths)
        
        return r2_df#,coeff_matrix,intercept_matrix


    def plot_OBRA(self,test_size = 0.3):
        """
        outputs the best band ratio in each environmental variable
        """
        df = self.df
        df['Altitude'] = round(self.df['Altitude'],2)
        wavelengths = df.iloc[:,-61:].columns
        var = df.columns[2:6]
        dict_var = {k:None for k in var}
        np.random.seed(1)
        print(dict_var)
        group_df_list = [] #stores a list of list of df
        for i in var:#iterate through environmental variables
            unique_var = sorted(list(set(round(df[i],2))))
            dict_var[i] = unique_var
            var_df_list = [("{}: {}".format(i,v),x) for v, x in self.df.groupby(self.df[i])] #split df by group,already sorted by group value
            group_df_list.append(var_df_list) #print([v for v,x in var_df_list])
            
        n_rows = len(group_df_list)
        n_cols = max([len(l) for l in group_df_list])
        fig, axes = plt.subplots(n_rows,n_cols, figsize = (12,12),sharex=True, sharey=True)
        cbar_ax = fig.add_axes([.91, .3, .03, .4]) #parameters are a list of rect coordinates. [x,y,xsize,ysize], values are relative to canvas size
        for row,g in enumerate(group_df_list):
            for col,(df_name,df) in enumerate(g):
                r2_df = self.band_ratio(df,test_size)
                R2_matrix = r2_df.to_numpy()
                b0,b1 = np.unravel_index(np.argmax(R2_matrix, axis=None), R2_matrix.shape)
                optimal_band_ratio = str(wavelengths[b0]) + '/' + str(wavelengths[b1])
                highest_r2 = np.max(R2_matrix)
                mask = np.zeros_like(R2_matrix)
                mask[np.tril_indices_from(mask)] = True #create a mask of lower triangle
                sns.heatmap(r2_df,
                vmin=0,vmax=1, #ensures that colorbar is synchronised
                xticklabels=5, yticklabels=5, #labels separated every 5 interval
                ax=axes[row,col], #subplot axis
                mask=mask,
                square=True,
                cbar = col == 0, #cbar = True if i==0 returns True
                cbar_kws={'label':r'$R^2$'},
                cbar_ax=None if col else cbar_ax)
                r2_label = r'$R^2$'
                title_label = '{}\nBand ratio: {}\n{}: {:.4}'.format(df_name,optimal_band_ratio,r2_label,highest_r2)
                axes[row,col].set_title(title_label)
                axes[row,col].scatter(b1,b0,marker='*',s=100,color = 'yellow',label='Band ratio with the highest R2')

        fig.tight_layout(rect=[0, 0, .9, 1],h_pad=0.3) #left bottom right top in normalised (0,1) figure coordinates
        
        #remove subplots that are not plotted
        for row,g in enumerate(group_df_list):
            for col in range(len(g),n_cols):
                fig.delaxes(axes[row][col])
        
        plt.show()


        return

def plot_OBRA(tss_br_df,wavelengths,test_size=0.3):
    """
    plot OBRA for each data collected under a scenario
    """
    def tssbr(df,test_size):
        X = df.iloc[:,6:]
        y = df['Concentration']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)

        n = 61#len(self.wavelengths)
        R2_matrix = np.zeros((n,n))
        #compute upper triangle R2
        count = 0
        for i in range(0,n): #rows
            for j in range(i+1,n): #columns
                reg = LinearRegression().fit(X_train.iloc[:,count:count+1],y_train) #regression model
                y_pred =reg.predict(X_test.iloc[:,count:count+1])
                r2 = r2_score(y_test,y_pred)
                R2_matrix[i,j] = r2
                count += 1

        r2_df = pd.DataFrame(R2_matrix,index=wavelengths,columns=wavelengths)
        return r2_df

    tss_br_df['Altitude'] = round(tss_br_df['Altitude'],2)
    var = tss_br_df.columns[2:6]
    dict_var = {k:None for k in var}
    np.random.seed(1)
    group_df_list = [] #stores a list of list of df
    for i in var:#iterate through environmental variables
        unique_var = sorted(list(set(round(tss_br_df[i],2))))
        dict_var[i] = unique_var
        var_df_list = [("{}: {}".format(i,v),x) for v, x in tss_br_df.groupby(tss_br_df[i])] #split df by group,already sorted by group value
        group_df_list.append(var_df_list) #print([v for v,x in var_df_list])
        
    n_rows = len(group_df_list)
    n_cols = max([len(l) for l in group_df_list])
    fig, axes = plt.subplots(n_rows,n_cols, figsize = (12,12),sharex=True, sharey=True)
    cbar_ax = fig.add_axes([.91, .3, .03, .4]) #parameters are a list of rect coordinates. [x,y,xsize,ysize], values are relative to canvas size
    for row,g in enumerate(group_df_list):
        for col,(df_name,df) in enumerate(g):
            r2_df = tssbr(df,test_size)
            R2_matrix = r2_df.to_numpy()
            b0,b1 = np.unravel_index(np.argmax(R2_matrix, axis=None), R2_matrix.shape)
            optimal_band_ratio = str(wavelengths[b0]) + '/' + str(wavelengths[b1])
            highest_r2 = np.max(R2_matrix)
            mask = np.zeros_like(R2_matrix)
            mask[np.tril_indices_from(mask)] = True #create a mask of lower triangle
            sns.heatmap(r2_df,
            vmin=0,vmax=1, #ensures that colorbar is synchronised
            xticklabels=5, yticklabels=5, #labels separated every 5 interval
            ax=axes[row,col], #subplot axis
            mask=mask,
            square=True,
            cbar = col == 0, #cbar = True if i==0 returns True
            cbar_kws={'label':r'$R^2$'},
            cbar_ax=None if col else cbar_ax)
            r2_label = r'$R^2$'
            title_label = '{}\nBand ratio: {}\n{}: {:.4}'.format(df_name,optimal_band_ratio,r2_label,highest_r2)
            axes[row,col].set_title(title_label)
            axes[row,col].scatter(b1,b0,marker='*',s=100,color = 'yellow',label='Band ratio with the highest R2')

    fig.tight_layout(rect=[0, 0, .9, 1],h_pad=0.3) #left bottom right top in normalised (0,1) figure coordinates

    #remove subplots that are not plotted
    for row,g in enumerate(group_df_list):
        for col in range(len(g),n_cols):
            fig.delaxes(axes[row][col])

    plt.show()
    return

class BandRatio:
    """
    takes a df and a test_size (ranges from 0 to 1) as input
    last 61 columns are the wavelengths
    contains a column named 'Concentration'
    outputs 3 matrices - r2,coeff,intercept, in the form of a df of upper triangle matrix where columns and indexes are wavelengths. 
    """
    def __init__(self,df,test_size):
        wavelength_cols = df.iloc[:,-61:]
        X = wavelength_cols
        y = df['Concentration']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)

        #compute all combinations of wavelength ratios:
        inverse_col_name = ["{}/{}".format(b,a) for a, b in combinations(wavelength_cols.columns, 2)]
        col_combinations = [(a,b) for a, b in combinations(X_train.columns, 2)]
        upper_tri_train = pd.concat([np.log(X_train[a].div(X_train[b]).rename(f'{a}/{b}')) for a, b in col_combinations], 1)
        upper_tri_test = pd.concat([np.log(X_test[a].div(X_test[b]).rename(f'{a}/{b}')) for a, b in col_combinations], 1)
        
        n = len(wavelength_cols.columns)
        R2_matrix = np.zeros((n,n))
        coeff_matrix = np.zeros((n,n))
        intercept_matrix = np.zeros((n,n))
        #compute upper triangle R2
        count = 0
        for i in range(0,n): #rows
            for j in range(i+1,n): #columns
                reg = LinearRegression().fit(upper_tri_train.iloc[:,count:count+1],y_train) #regression model
                reg_coeff = reg.coef_ #since only 1 variable is a predictor, only one coefficient
                y_pred =reg.predict(upper_tri_test.iloc[:,count:count+1])
                r2 = r2_score(y_test,y_pred)
                R2_matrix[i,j] = r2
                coeff_matrix[i,j] = reg_coeff[0] #it only has one coefficient anyway
                intercept_matrix[i,j] = reg.intercept_
                count += 1

        wavelengths = wavelength_cols.columns.tolist()
        r2_df = pd.DataFrame(R2_matrix,index=wavelengths,columns=wavelengths)
        coeff_df = pd.DataFrame(coeff_matrix,index=wavelengths,columns=wavelengths)
        intercept_df = pd.DataFrame(intercept_matrix,index=wavelengths,columns=wavelengths)

        self.r2 = r2_df
        self.coeff = coeff_df
        self.intercept = intercept_df


class ENBRAS:
    def __init__(self,df,frac_to_sample=1,n_sample=10):
        """
        frac_to_sample (float): (fraction of df to sample, determines size of sampled dataframe),
        n_sample (int): (number of times sampling is conducted), 
        """
        self.df = df
        self.frac_to_sample = frac_to_sample
        self.n_sample = n_sample

    def best_model_parameters(self,lbr_object):
        """
        input is an LogBandRatio object
        outputs a pandas series e.g. pd.Series({'b0':b0,'b1':b1,'r2':r2,'coeff':coeff,'intercept':intercept})
        """
        r2_df = lbr_object.r2
        coeff_df = lbr_object.coeff.values
        intercept_df = lbr_object.intercept.values
        wavelength = r2_df.columns.tolist()
        r2_array = r2_df.values
        r2_max_index = np.unravel_index(r2_array.argmax(),r2_array.shape)
        b0 = wavelength[r2_max_index[0]]
        b1 = wavelength[r2_max_index[1]]
        r2 = r2_array[r2_max_index]
        coeff = coeff_df[r2_max_index]
        intercept = intercept_df[r2_max_index]

        return pd.Series({'b0':b0,'b1':b1,'r2':r2,'coeff':coeff,'intercept':intercept})

    def bagging(self,test_size,store_directory,label=""):
        """
        This function first samples rows randomly from df with replacement and then does the splitting of train and test set
        so train and test set keeps changing
        because we are trying to see how well the model performs with different distribution of test distribution
        the selected candidate band ratios will be the band ratios that are relatively invariant to different environmental conditions
        number of rows it samples depends on test_size. it samples (1-test_size)
        inputs are:
        df (pd dataframe): where last 61 columns are wavelengths, and contains a column called 'Concentration'),
        test_size (float): (determines the partition of sampled dataframe to test and train set, ranges from 0 to 1),
        label (str): To save a unique filename in csv
        store_directory (str): directory to output the file
        Doesnt return anything but df is saved as a csv
        """
        np.random.seed(1) #to ensure reproducibility

        # candidate_df_list = []
        best_candidate = []
        
        for i in range(self.n_sample):
            df_ran = self.df.sample(frac=self.frac_to_sample,replace=True) #sampling with replacement
            #sampling with replacement makes sense because in reality it is possible that we will have very similar TSS measurements and spectra
            lbr_object = BandRatio(df_ran,test_size)
            best_candidate.append(self.best_model_parameters(lbr_object))
        
        pd.concat(best_candidate,1).transpose().to_csv(join(store_directory,'candidate_BBRs_test{}{}.csv'.format(test_size,label)),index=False)
        return

class Clustering:
    """
    performs modified Batchelor & Wilkin's algorithm
    """
    def __init__(self,prefix,store_directory):
        """
        prefix (str): names
        store_directory (str): directory where the candidate BBRs csv files are stored
        """
        self.prefix = prefix
        self.store_directory = store_directory

    def import_csv_list(self):
        """
        this function imports csv files that contains the prefix
        input is a prefix of the csv files you want to import
        >>>import_csv_list("best_candidate_lbr")
        """
        df_list = []
        test_list = [0.5,0.4,0.1,0.3,0.2] 
        #the order in which the files are imported is important because the initialisation of the clustering depends on this. 
        # Sometimes glob changes the order at which the files are imported, so results may differ at times
        for t in test_list:
            f = join(self.store_directory,"candidate_BBRs_test{}.csv".format(str(t)))
            print(f)
            df_list.append(pd.read_csv(f))
        # df_list = []
        # for f in listdir(self.store_directory)
        # for file in glob.glob(join(self.store_directory,"{}*.csv".format(self.prefix))):
        #     df_list.append(pd.read_csv(file))
        return df_list

    def plot_probability(self):
        df_list = self.import_csv_list()
        best_candidate_concat = pd.concat(df_list,axis=0)
        # best_candidate_concat_drop_dup = pd.concat(df_list,axis=0).drop_duplicates(['b0','b1'])
        df_summary = best_candidate_concat.groupby(['b0','b1']).agg({'r2':['mean','std','count']}).reset_index().sort_values(['b0','b1'])
        df_summary['probability'] = df_summary['r2']['count']/df_summary['r2']['count'].sum() #count falls under multi-index
        df_summary['band_ratio'] = df_summary['b0'].astype(str) + '/' + df_summary['b1'].astype(str)
        df_summary['weight_std'] = 1-df_summary['r2']['std']/df_summary['r2']['std'].sum()
        df_summary['count_weighted_std'] = df_summary['r2']['count']*df_summary['weight_std']
        df_summary['probability_weighted_std'] = df_summary['count_weighted_std']/df_summary['count_weighted_std'].sum()
        df_summary['weight_std_infl'] = df_summary['r2']['std'].sum()/df_summary['r2']['std']
        df_summary['count_weighted_std_infl'] = df_summary['r2']['count']*df_summary['weight_std_infl']
        df_summary['probability_weighted_std_infl'] = df_summary['count_weighted_std_infl']/df_summary['count_weighted_std_infl'].sum()
        df_summary_sort = df_summary.sort_values('probability',ascending=False)

        text_size = 18
        band_ratio_labels = ["{:.2f}/{:.2f}".format(float(b0),float(b1)) for b0,b1 in zip(df_summary['b0'],df_summary['b1'])]
        fig, axes = plt.subplots(figsize=(5,15))
        ax = df_summary[['probability']].plot.barh(xlim=(0,1),logx=True,ax=axes,color="#A9A9A9")
        ax.set_yticklabels(band_ratio_labels)
        ax.set_xlabel('Probability',fontsize=text_size)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_ylabel('Candidate best band ratios',fontsize=text_size)
        legend_labels = ['Probability']
        ax2 = ax.twiny()
        ax2.errorbar(df_summary['r2']['mean'],df_summary['band_ratio'],xerr=df_summary['r2']['std'],c="black",ls=':',marker='.',capsize=3.0, label=r'$R^2$')
        ax2.set_xlabel(r'$R^2$',fontsize=text_size)
        ax2.set_xlim(0,1)
        ax.legend(labels=legend_labels, loc=(1.05,0.9), fontsize=14)
        ax2.legend(loc=(1.05,0.8),fontsize=14)
        plt.show()
        return df_summary_sort

    def baw(self,X,p):
        """
        This function computes the batchelor & Wilkon's algorithm for clustering
        do not require user-define number of cluster, but requires user-define p (proportion of maximum distance)
        input is list of coordinates e.g. np.array([[0,0],[0,1],[1,0],[1,1]])
        make sure input has been sorted by b0 & b1, then it's easier to select a coord from a corner
        p = ranges from 0 to 1, if max_of_min > p*previous maximum distance, assign new sample to new cluster
        """
        
        #initialise z1 as the first item
        z1 = X[0] #after sorting by b0 & b1, first element should be one corner
        #compute z2
        # z2 = X[-1] ##after sorting by b0 & b1, first element should be the opp corner to save computation time
        max_d = 0
        for x in X:
            d = np.linalg.norm(x-z1)
            if d>max_d:
                max_d = d #update maximum distance
                z2 = x #update point as the furthest point
        z1z2 = np.linalg.norm(z2-z1)  #distance between z1 and z2

        cluster_centers=[z1,z2] #initiate cluster centers as z1,z2

        max_dist = z1z2
        
        while max_dist > p*z1z2:#prev_max_dist: #ensures that we enter the loop
            max_of_min = (X[0],0) #to keep track of the maximum of min distances with every iteration
            #get the maximum of minimum distance and the corresponding point
            for i,x in enumerate(X):
                min_dist = (0,max_dist) #initialise min dist as the current max dist
                for c in cluster_centers:
                    if np.linalg.norm(x-c) < min_dist[1]: #compute distance between point and cluster center
                        min_dist = (x,np.linalg.norm(x-c)) #update with coord and dist
                if min_dist[1] > max_of_min[1]: 
                    #if min_dist <max_of_min, max_of_min wont get updated, it will continue to be prev value
                    #as such, max_of_min == prev_max_dist. if so, break the loop
                    max_of_min = (min_dist[0],min_dist[1]) #keep track of the coord & max of min values

            if max_of_min[1] > p*z1z2:
                cluster_centers.append(max_of_min[0])
                max_dist = max_of_min[1] #update current max dist

            else:
                break

        # assign points to clusters
        cluster_dict = {key:[] for key in list(range(len(cluster_centers)))} #dict comprehension
        for x in X:
            dist_to_center = []
            for i,c in enumerate(cluster_centers): 
                dist = np.linalg.norm(x-c)
                dist_to_center.append(dist)
            cluster_index = np.argmin(dist_to_center)
            cluster_dict[cluster_index].append(x)
                    
        # print("final cluster_centers:{}\nCluster dict:{}".format(cluster_centers,cluster_dict))
        
        return cluster_centers,cluster_dict


    def determine_optimal_p(self,X,start_p,stop_p,step_size):
        """
        determines what is the optimal p for baw function
        outputs a list of graph that shows the clustering, and a graph of how the MSE changes with different p values
        p values ranges from [0.1 to 1], increment by step_size
        inputs are array of coord and start_p, stop_p, step_size
        >>> determine_optimal_p([[x1,x2],[x3,x4]],0.5)
        """
        p_list = np.arange(start_p,stop_p,step_size)
        initial_error_list = []
        recomp_error_list = []

        prev_baw_attr = self.recompute_cluster(X,p_list[0])
        prev_initial_mse = prev_baw_attr['initial mse']
        prev_recomp_mse = prev_baw_attr['recomp mse']

        for i,p in enumerate(p_list):
            # classified_points = baw(X,0.5)
            baw_attr = self.recompute_cluster(X,p)
            initial_error_list.append(baw_attr['initial mse'])
            recomp_error_list.append(baw_attr['recomp mse'])

            if i == 0:
                self.graph_baw(baw_attr,p) #plot first graph
            else:
                if abs(baw_attr['initial mse'] - prev_initial_mse) > 1e-3 or abs(baw_attr['recomp mse'] - prev_recomp_mse) > 1e-3:
                    #update prev mse values
                    prev_initial_mse = baw_attr['initial mse']
                    prev_recomp_mse = baw_attr['recomp mse']
                    self.graph_baw(baw_attr,p)
                elif abs(baw_attr['initial mse'] - prev_initial_mse) < 1e-3 and abs(baw_attr['recomp mse'] - prev_recomp_mse) < 1e-3:
                    #if graph the same as before, dont plot it
                    continue
            
            #graph using different p
        min_index = np.argmin(recomp_error_list)
        min_mse = np.min(recomp_error_list)

        plt.figure()
        plt.plot(p_list,initial_error_list,label='Initial MSE')
        plt.plot(p_list,recomp_error_list,label='Recomputed MSE')
        plt.scatter(p_list[min_index],min_mse,marker='x',label='Minimum MSE',c="black")
        plt.xlabel('p')
        plt.ylabel('Mean Squared Error')
        # plt.text(p_list[min_index],min_mse,"p: {}, MSE:{:.4}".format(p_list[min_index],min_mse),
        # bbox=dict(alpha=0.5,facecolor='white'))
        plt.annotate("p = {:.3}, MSE ={:.4}".format(p_list[min_index],min_mse),(p_list[min_index],min_mse),
        xytext=(10, 10), textcoords='offset points',bbox=dict(alpha=0.7,facecolor='white'))
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()

        return p_list[min_index]
    
    def compute_error_clusters(self,cluster_centers,cluster_dict):
        """
        computes mean squared error (MSE) for each cluster and sums them up.
        outputs the total MSE
        inputs are a list of coordinates of cluster centers and a dictionary of clusters
        >>>compute_error_clusters([coord1,coord2,coord3],{0:[x1,x2,x3],1:[x4,x5,x6]})
        """
        # cluster_centers,cluster_dict = baw(X,p)
        cluster_mean_squared_error = []
        for c,(k,v) in zip(cluster_centers,cluster_dict.items()):
            sum_v = 0
            n = len(v)
            for i in v:
                sum_v += np.linalg.norm(i-c)
            avg_cluster_center = sum_v/n
            cluster_mean_squared_error.append(avg_cluster_center)
        total_mean_squared_error = np.sum(cluster_mean_squared_error) #sum over all clusters

        return total_mean_squared_error
    
    def recompute_cluster(self,X,p):
        """
        this function takes the output of baw function and recomputes the mean coord of clusters, and re-classify the points
        inputs are an array of coordinates of shape (n,2), and 
        p, where p = proportion/fraction of the maximum distance
        """
        cluster_centers,cluster_dict = self.baw(X,p)
        recomputed_cluster_center = []
        for k,v in cluster_dict.items():
            sum_v = 0
            n = len(v)
            for i in v:
                sum_v += i
            avg_cluster_center = sum_v/n
            recomputed_cluster_center.append(avg_cluster_center)
        
        initial_mse = self.compute_error_clusters(cluster_centers,cluster_dict)
        # print("initial MSE:{:.4}".format(initial_mse))
        # print("recomputed cluster center: {}".format(recomputed_cluster_center))

        #recompute cluster center
        recomputed_cluster_dict = {key:[] for key in list(range(len(recomputed_cluster_center)))} #dict comprehension
        for x in X:
            dist_to_center = []
            for i,c in enumerate(recomputed_cluster_center): 
                dist = np.linalg.norm(x-c)
                dist_to_center.append(dist)
            cluster_index = np.argmin(dist_to_center)
            recomputed_cluster_dict[cluster_index].append(x)

        # print("recomputed cluster dict: {}".format(recomputed_cluster_dict))
        recomp_mse = self.compute_error_clusters(recomputed_cluster_center,recomputed_cluster_dict)
        # print("recomputed MSE:{:.4}".format(recomp_mse))


        return {'initial dict': cluster_dict,'initial mse': initial_mse,'initial centers':cluster_centers,
        'recomp dict':recomputed_cluster_dict,'recomp mse':recomp_mse, 'recomp centers': recomputed_cluster_center}

    def graph_baw(self,baw_attr,p):
        """
        outputs a 2 x 1 graph, where L is the inital cluster and R is the recomputed cluster
        inputs are the cluster dictionary (output from recompute_cluster), and p
        >>> graph_baw({'initial dict': cluster_dict,'initial mse': initial_mse,'initial centers':cluster_centers,
        'recomp dict':recomputed_cluster_dict,'recomp mse':recomp_mse, 'recomp centers': recomputed_cluster_center}, 0.5)
        """
        fig, axes = plt.subplots(1,2,figsize=(10,4))
        diff_graphs = [baw_attr['initial dict'],baw_attr['recomp dict']]
        cluster_centers = [baw_attr['initial centers'], baw_attr['recomp centers']]
        n_cluster = len(baw_attr['initial centers'])
        cluster_centers = [np.vstack(i) for i in cluster_centers]
        err_labels = [baw_attr['initial mse'],baw_attr['recomp mse']]
        titles = ['Initial cluster','Recomputed cluster']
        p_label = "\np = {:.2}".format(p)

        titles = [t + p_label + ", MSE = {:.3}".format(err) for t,err in zip(titles,err_labels)]
        for j,d in enumerate(diff_graphs):
            # print(j,d)
            coords = []
            cluster_labels = []
            for k,v in d.items():
                for i in v:
                    coords.append(i)
                    cluster_labels.append(k)

            coords = np.array(coords)
            # axes[j].scatter(coords[:,0],coords[:,1],c=cluster_labels,cmap=plt.cm.get_cmap('viridis'))
            axes[j].scatter(coords[:,1],coords[:,0],c=cluster_labels,cmap=plt.cm.get_cmap('viridis'))
            axes[j].set_ylabel('b0 wavelength')
            axes[j].set_xlabel('b1 wavelength')
            cluster_center = cluster_centers[j]
            axes[j].scatter(cluster_center[:,1], cluster_center[:,0],marker='x',label='{} cluster centers'.format(n_cluster),c='black',alpha=0.5)
            # axes[j].scatter(cluster_center[:,0], cluster_center[:,1],marker='x',label='{} cluster centers'.format(n_cluster),c='black',alpha=0.5)
            axes[j].set_title(titles[j])
            axes[j].legend()
        
        fig.show()
        return #baw_attr['recomp dict']

    def plot_cluster(self,output="output"):
        """
        this function takes in best_candidates as an input
        it aims to cluster the band ratios together (based on distance) to create different clusters
        in each cluster.
        outputs df with classes mapped to each b0,b1 - clustered_candidates
        outputs a graph with colorbar corresponding to diff r2, and markers for different clusters
        """
        df_list = self.import_csv_list()
        probable_candidates = pd.concat(df_list,axis=0)
        # probable_candidates = pd.concat(df_list,axis=0).drop_duplicates(['b0','b1'])
        X = probable_candidates[['b0','b1']].values
        p = self.determine_optimal_p(X,0.1,1,0.1)
        classified_X = self.recompute_cluster(X,p)
        classified_X = classified_X['recomp dict']

        test_dict1 = {k:np.vstack(v) for k,v in classified_X.items()}
        dict_to_df = pd.concat([pd.DataFrame(v) for k, v in test_dict1.items()], axis = 1, keys = list(test_dict1.keys()))
        dict_to_df.rename(columns={0:'b0',1:'b1'},level=1,inplace=True)
        bands_class = dict_to_df.stack(0).reset_index().rename(columns={'level_1':'class'}).drop(columns=['level_0']).drop_duplicates()
        df_merge = pd.merge(probable_candidates,bands_class,how='left',on=['b0','b1'])
        if (pd.isnull(df_merge['class']).sum() > 0):
            print("NA values in class column, please recheck")
            
        df_merge['class'] = df_merge['class'].astype('int32') #change float to int

        # #create markers list
        markers = ['o','P','X','*','p','D','s']
        # groups = df_merge.groupby('class')
        cm = plt.cm.get_cmap('RdYlBu_r')
        # fig, ax = plt.subplots()
        plt.figure()
        n_groups = len(set(df_merge['class']))
        min_r2,max_r2 = df_merge['r2'].min(), df_merge['r2'].max()
        mkr_dict = {i:markers[i] for i in range(n_groups)}
        for kind in mkr_dict:
            d = df_merge[df_merge['class'] == kind]
            plt.scatter(d.b1,d.b0,c=d.r2,marker=mkr_dict[kind],cmap=cm,alpha=0.5,label='cluster {}'.format(kind))
            plt.clim(min_r2,max_r2) #important so that cmap doesnt only map the colours for the subset of df, to ensure color bars for all subset are consistent
        plt.colorbar().set_label(r'$R^2$',rotation=270,labelpad=15)
        plt.ylabel('b0 wavelength')
        plt.xlabel('b1 wavelength')
        # plt.xlim(400,1000)
        # plt.ylim(400,1000)
        plt.legend()
        ax = plt.gca()
        leg = ax.get_legend()
        for i in range(n_groups):
            leg.legendHandles[i].set_color('black')
        plt.show()

        df_merge.to_csv(join(output,'classified_df.csv'),index=False)
        return df_merge

    def clustered_probability(self,clustered_candidates):
        """
        input is clustered_candidates, must have b0,b1,r2,class columns
        outputs a graph that shows the deviation inter and intra class
        """
        band_class = clustered_candidates.groupby(['class','b0','b1']).agg({'r2':['min','max', 'mean','std','count']})#.reset_index()
        class_df = clustered_candidates.groupby(['class']).agg({'r2':['min','max', 'mean','std','count']})
        n_class = len(class_df.index)
        class_column = clustered_candidates['class'].tolist()
        class_dict = dict((Counter(class_column))) #number of items in a class
        xtick_locations = list(range(n_class))
        fig, axes = plt.subplots(figsize=(6,4))
        axes.bar(list(range(n_class)),class_df['r2']['mean'],yerr=class_df['r2']['std'],color="white",edgecolor="black",linewidth=1, capsize=3.0)
        axes.set(xticks=xtick_locations)
        axes.yaxis.set_major_formatter(ScalarFormatter())
        axes.set_ylabel(r'$R^2$')
        axes.set_yscale('log')
        axes.set_xticklabels(["Cluster {}".format(i) for i in xtick_locations])
        plt.show()

       
        simple_df = band_class['r2'].reset_index()

        fig,axes = plt.subplots(n_class,1,figsize=(10,3*n_class)) #stretch flexibly with n_class

        for class_i in range(n_class):
            class0 = simple_df[simple_df['class']==class_i]
       
            b0_count = Counter(class0['b0']) #count number of b1 grouped by b0


            count_b1 = [v for k,v in b0_count.items()]
            n_sections = len(count_b1) #number of grouped barplots
            n_bars = sum(count_b1)
            n_bars_w_space = n_bars + n_sections - 1
            x_axis_w_space = list(range(n_bars_w_space))
            class0_mean = class0['mean'].tolist()
            class0_std = class0['std'].tolist()
            class0_count = class0['count'].tolist()
            b1_labels = class0['b1'].tolist()
            b0_labels = class0['b0'].tolist()

            #keep track of how many space inserted
            sum_of_counter = 0
            # mean_w_spaces = []
            for nth_row in count_b1[:-1]: #exclude the last one 
                sum_of_counter += nth_row 
                class0_mean.insert(sum_of_counter,0)
                class0_std.insert(sum_of_counter,0)
                class0_count.insert(sum_of_counter,0)
                b1_labels.insert(sum_of_counter,None)
                b0_labels.insert(sum_of_counter,None)
                sum_of_counter += 1
                #counter += 1
            #-------custom color map----------
            #map values to color
            min_col = min(class0_count)
            max_col = max(class0_count)
            norm = pltcolors.Normalize(vmin=min_col, vmax=max_col, clip=True)
            mapper = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.Spectral_r)
            color_list = [mapper.to_rgba(i) for i in class0_count]
            #-------custom color map----------

            #------major b0 labels------------
            prev_b0 = b0_labels[0]
            major_b0_labels = ["{:.2f}/ ".format(prev_b0)]
            for b0 in b0_labels[1:]:
                if b0 == prev_b0 or b0 == None:
                    major_b0_labels.append("")
                else:
                    major_b0_labels.append("{:.2f}/ ".format(b0))
                    prev_b0 = b0 #update prev_b0
            #------major b0 labels------------

            # fig,axes = plt.subplots(figsize=(20,4))
            axes[class_i].bar(x_axis_w_space,class0_mean,yerr=class0_std,capsize=3.0,width=1,edgecolor="black",linewidth=0.2,color = color_list) #set width=1 to remove gaps between bars
            axes[class_i].set_yscale('log')
            axes[class_i].set_title('Cluster {}, n = {}'.format(class_i,class_dict[class_i]))
            axes[class_i].set_ylabel(r'$R^2$')
            axes[class_i].yaxis.set_major_formatter(ScalarFormatter())
            # axes[class_i].yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
            axes[class_i].set_xticks([x_tick for x_tick,x_label in zip(x_axis_w_space,b1_labels) if x_label != None])
            axes[class_i].set_xticklabels(["{}{}".format(b0_label,b1_label) for b0_label,b1_label in zip(major_b0_labels,b1_labels) if b1_label != None],rotation=90)
            fig.colorbar(mapper,ax=axes[class_i]).set_label('Frequency of band ratio',rotation=270,labelpad=15)
        
        plt.tight_layout()
        fig.show()


        return 


