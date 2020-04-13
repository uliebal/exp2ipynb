"""
Function collection for data manipulation.
Basic function file for analysis of expression strength based on promoter 
sequence. In this file simple functions and visualization is
handled.


Author: Ulf Liebal
Contact: ulf.liebal@rwth-aachen.de
Date: 11/2019
"""
# Data preparation
def Data_Src_Load(Name_Dict):
    '''
    Data loading of Excel sheet to pandas data frame.
    
    Input:
            Name_Dict:      dictionary; contains source data adress and additional information of multiple library expression measurements and multiple sequence libraries.
    
    Output:
            SeqDat:           dataframe
    '''
    import os
#     import tkinter as tk
    import numpy as np
    import pandas as pd
#     from tkinter import filedialog
    from ExpressionExpert_Functions import list_integer, list_onehot


    DataPath = Name_Dict['Data_File']

    SeqDat = pd.read_csv(DataPath, delimiter=',|;', engine='python')
    if bool(Name_Dict['Library_Sequence_single']) == True:
        Seq_Col = Name_Dict['Sequence_column']
        SeqDat['Sequence_label-encrypted'] = list_integer(SeqDat[Seq_Col])
        SeqDat['Sequence_letter-encrypted'] = SeqDat[Seq_Col]
        SeqDat['Sequence'] = list_onehot(SeqDat['Sequence_label-encrypted'])
        # GC content calculation
        NuclSum = [np.sum(SeqDat['Sequence'][i], axis=0) for i in range(len(SeqDat))]
        GCcont = np.sum(np.delete(np.vstack(NuclSum),[0,3],1), axis=1)/np.sum(np.vstack(NuclSum)[0])
        SeqDat['GC-content'] = GCcont
    else:
        print('Multiple indepented sequence library analysis not yet supported.')
        
    return SeqDat

def make_DataDir(Name_Dict):
    '''
    Set-up of directory for data storage.
    '''
    import os
    
    Data_File = Name_Dict['Data_File']
    # extract the filename for naming of newly generated files
    File_Base = Data_File.split('.')[0]
    # the generated files will be stored in a subfolder with custom name
    Data_Folder = 'data-{}'.format(File_Base)
    try:
        os.mkdir(Data_Folder)
        print('Data directory ', Data_Folder, 'created.')
    except FileExistsError:
        print('Already existent data directory ', Data_Folder, '.')

def ExpressionScaler(SeqDat, Name_Dict):
    '''
    Scaling of expression values to zero mean and unit variance.
    
    Input:
        SeqDat:     dataframe, contains data
        Name_Dict:  dictionary, information on number of independent expression measurements
        
     Output:
         SeqDat:    dataframe, contains data with scaled measurements
         Expr_Scaler: dictionary, contains the scaler functions
    '''
    from sklearn.preprocessing import StandardScaler
    import copy
    
    Y_Col_Name = eval(Name_Dict['Y_Col_Name'])
    Measurement_Number = int(Name_Dict['Library_Expression'])
    
    Expr_Scaler = dict()
    Expr_Scaler_n = StandardScaler() 
    for idx in range(Measurement_Number):
        Column_Name = '{}_scaled'.format(Y_Col_Name[idx])
        myData = SeqDat[Y_Col_Name[idx]].values.reshape(-1, 1)
        SeqDat[Column_Name] = Expr_Scaler_n.fit_transform(myData)
        Scaler_Name = '{}_Scaler'.format(Y_Col_Name[idx])
        Expr_Scaler[Scaler_Name] = copy.deepcopy(Expr_Scaler_n)
     
    return SeqDat, Expr_Scaler
    

def Insert_row_(row_number, df, row_value): 
    '''
    The function allow the insertion of rows in a dataframe. The index counting 
    is reversed, the last element is set to -1 and each element to the top is 
    decrease by 1.
    
    Input:
        row_number: numpy integer/vector, index from top to insert line
        df:         pandas dataframe
        row_value:  numpy vector/matrix, contains numbers being inserted
    '''
    import pandas as pd
    
    for idx, element in zip(row_number, row_value):
        # Slice the upper half of the dataframe 
        df1 = df[0:idx] 

        # Store the result of lower half of the dataframe 
        df2 = df[idx:] 

        # Inser the row in the upper half dataframe 
        df1.loc[idx]=element 

        # Concat the two dataframes 
        df = pd.concat([df1, df2]) 

        # Reassign the index labels 
        df.index = [*range(df.shape[0])] 

    # Return the updated dataframe 
    df.index = [*range(-df.shape[0],0,1)] 
    return df

def split_train_test(SeqDat, test_ratio=.1):
    '''
    Data split into training and test sets, with given ratio (default 10% test)
    
    Input:
        SeqDat:     dataframe, contains data
        test_ratio: float [0,1), ratio of test data to total data
    
    Output:
        Arg#1:      dataframe, training data
        Arg#2:      dataframe, test data        

    '''
    import random 
    import numpy as np
    
    My_Observed,_ = SeqDat.shape
    Test_Idx = np.sort(np.array(random.sample(range(My_Observed), np.int(test_ratio* My_Observed))))
    Train_Idx = np.setdiff1d(range(My_Observed), Test_Idx)
    return SeqDat.iloc[Train_Idx], SeqDat.iloc[Test_Idx]

###############################################################################
# The following functions serve for the one-hot encoding
# It is derived from:
#    https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
def list_integer(SeqList):
    '''define input values'''
    alphabet = 'ACGT'
    char_to_int = dict((c,i) for i,c in enumerate(alphabet))
    IntegerList = list()
    for mySeq in SeqList:    
        # integer encode input data
        integer_encoded = [char_to_int[char] for char in mySeq.upper()]
        IntegerList.append(integer_encoded)
    return IntegerList
        
def list_onehot(IntegerList):
    OneHotList = list()
    for integer_encoded in IntegerList:    
        onehot_encoded = list()
        for value in integer_encoded:
            letter = [0 for _ in range(4)]
            letter[value] = 1
            onehot_encoded.append(letter)
        OneHotList.append(onehot_encoded)
    return OneHotList

def list_sequence(IntegerList):
    '''define input values'''
    import numpy as np
    alphabet = 'ACGT'
    int_to_char = dict((i,c) for i,c in enumerate(alphabet))
    SequenceList = list()
    for IntRow in IntegerList:    
        # integer encode input data
        sequence_encoded = [int_to_char[int(myint)] for myint in np.nditer(IntRow)]
        SequenceList.append(sequence_encoded)
    return SequenceList
    
###############################################################################
# General function to identify conserved sequences
def Conserved_Sequence_Exclusion(SeqLab, n=0):
    '''Returns the sequence positions that have a lower or equal diversity than given as threshold (n).
    Input:
          SeqLab: np array, columns represent sequence position, rows represent samples
          n:      float,  threshold for variability report in entropy values, default=0
    Output:
          Position_Conserved: Array, positions of low variability within the sequence length'''
#    import pandas as pd
    import numpy as np
    
    PSEntropy = Entropy_on_Position(SeqLab)
    Position_Conserved = np.arange(len(PSEntropy))[PSEntropy <= n]
    return Position_Conserved, PSEntropy

def Sequence_Conserved_Adjusted(SeqDat, Name_Dict, n=0):
    '''Returns the dataframe with adjusted sequence length by removal of positions with low variability
    Input:
          SeqDat_current: DataFrame, From original Data, the labeled encrypted sequence column needs to be called 'Sequence_label-encrypted'
          Name_Dict:  dictionary, information on number of independent expression measurements
          n:     Integer, sequence variability limit in percent, default=0'''
    import numpy as np
    
    Position_Conserved, PSEntropy = Conserved_Sequence_Exclusion(np.array(SeqDat['Sequence_label-encrypted'].tolist()), n)
    SeqDat = SeqDat.assign(ColName=SeqDat['Sequence_label-encrypted'])
    SeqDat.rename(columns={'ColName':'Sequence_label-encrypted_full'}, inplace=True);
    SeqDat['Sequence_label-encrypted'] = list(np.delete(np.array(list(SeqDat['Sequence_label-encrypted'])),Position_Conserved, axis=1))
    SeqDat['OneHot'] = list_onehot(SeqDat['Sequence_label-encrypted'])
    
    return SeqDat, Position_Conserved, PSEntropy



###############################################################################

def Sequence_Dist_DiffSum(SeqObj):
    '''Returns the genetic sequence distance all sequences in the data list.
    Input:
           SeqDF: list, From original Data, the sequence in conventional letter format
    Output:
           PromDist_SymMatrix: list object, genetic distances as determined from the sum of difference in bases divided by total base number, i.e. max difference is 1, identical sequence =0
    '''
    import numpy as np

    Num_Samp = len(SeqObj)
    PromDist = list()
    for idx1 in range(Num_Samp):
        for idx2 in range(idx1, Num_Samp):
            PromDist.append(np.sum([int(seq1!=seq2) for seq1,seq2 in zip(SeqObj[idx1],SeqObj[idx2])], dtype='float')/len(SeqObj[idx1]))
    
    Entry_Size = np.insert(np.cumsum(np.arange(Num_Samp,0,-1)),0,0)
    PromDist_SymMatrix = np.zeros((Num_Samp,Num_Samp))
    for index in range(0,Num_Samp):
        PromDist_SymMatrix[index,index:] = PromDist[Entry_Size[index]:Entry_Size[index+1]]
        PromDist_SymMatrix[index:,index] = PromDist[Entry_Size[index]:Entry_Size[index+1]]

    return PromDist_SymMatrix

def Sequence_Ref_DiffSum(SeqObj):
    '''Returns the genetic sequence distance relative to the first sequence in the data-frame to all following sequences in the data list.
    Input:
           SeqDF: list, From original Data, the sequence in conventional letter format
    Output:
           PromDist: array, genetic distances as determined from the sum of difference in bases divided by total base number, i.e. max difference is 1, identical sequence =0
    '''
    import numpy as np

    RefSeq = SeqObj[0]
    Num_Samp = len(SeqObj)
    PromDist = list()
    for idx1 in range(1, Num_Samp):
        PromDist.append(np.sum([int(seq1!=seq2) for seq1,seq2 in zip(RefSeq, SeqObj[idx1])], dtype='float')/len(SeqObj[idx1]))
    
    return np.array(PromDist)

def Find_Near_Seq(SeqTest, SeqRef):
    '''
    Identifies the closes Reference sequence to a test sequence
    
    Input:      SeqTest:    string sequence
                SeqRef:     string array
    
    Output:     SeqTarget:  string sequence from reference with closest distance to test
    '''
    import numpy as np
    
    # irrelevant sequence positions can be marked by 'N', here we find them and only use true bases for finding the target sequence
    PosTrue = [i for i,x in enumerate(SeqTest) if x != 'N']
    Num_Samp = len(SeqRef)
    PromDist = list()
    for idx1 in range(Num_Samp):
        # only selecting relevant bases
        seqtest = [SeqTest[i] for i in PosTrue]
        seqref = [SeqRef[idx1][i] for i in PosTrue]
        PromDist.append(np.sum([int(seqtest!=seqref) for seqtest,seqref in zip(seqtest, seqref)], dtype='float')/len(PosTrue))
    
    PromDist = np.array(PromDist)
    SeqClose_Idx = np.argsort(PromDist)[0]
    
    SeqTarget = np.unique(SeqRef[SeqClose_Idx])
    
    return SeqTarget, SeqClose_Idx

def Extract_MultLibSeq(SeqDat, Target, Seq_Numb, Y_Col_Name):
    '''
    finding the closest measured samples in experimental library to the target expression 
    '''
    import numpy as np
    
    Target_lst = []
    SeqObj = []
    Expr_Prom_Num = Target.shape[1]
    
    for Expr_idx in range(Expr_Prom_Num):
        # finding positions of unique sequences
        u, index = np.unique(SeqDat['Sequence_letter-encrypted'].str.upper(), return_inverse=True)
        Base_Expr = SeqDat[Y_Col_Name].values
        # calculation of the distance of each promoter strength to target strength
        TR_Dist = np.abs(1-Base_Expr/Target[:,Expr_idx])
        # sorting the closest expression strength to the beginning
        Target_Idx = np.argsort(np.sum(TR_Dist, axis=1))
        # replicates should all have similar expression values, but we want to know different sequences that are close to the target sequence
        # We insert the ordered expression distance into the categories from the unique sequences
        # the output gives the position of the closest expression in the categorized sequences
        _, i2 = np.unique(index[Target_Idx], return_index=True)
        Seq_lst = u[np.argsort(i2)[:Seq_Numb]]

        # The extraction of the position of the closest expression in the original data is complex because of replicates.
        # In the following we select the closest indices for unique sequences
        Expr_ord = index[Target_Idx]
        Expr_uni = []
        [Expr_uni.append(x) for x in Expr_ord if x not in Expr_uni]  
        Idx_lst = np.array([list(Expr_ord).index(Indx) for Indx in Expr_uni])
        Target_unique = Target_Idx[Idx_lst]

        Target_lst.append(Target_unique[:Seq_Numb])
        SeqObj.append(Seq_lst)
    
    return SeqObj, Target_lst

# Entropy calculation
def Entropy_on_Position(PSArray):
    '''
    Analysis of position specific entropy. 
    
    Input: 
        PSArray: np array, columns represent sequence position, rows represent samples
        
    Output:
        PSEntropy: np vector, entropy of each sequence position
    '''
    import numpy as np
    from scipy.stats import entropy
    
    PSEntropy = list()
    for col in PSArray.T:
        value, counts = np.unique(col, return_counts=True)
        PSEntropy.append(entropy(counts, base=2))
        
    return np.array(PSEntropy)

def Est_Grad_Save(SeqOH, Validation_cutoff=.1, Num=100, Y_Col_Name='promoter activity'):
    '''
    This function performs gradient search for optimal parameters with shuffle shift and stores it.
    
    Input:
        DataArray
    '''
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
#    from sklearn.metrics import r2_score
    import numpy as np

    Sequence_Samples, Sequence_Positions, Sequence_Bases = np.array(SeqOH['OneHot'].values.tolist()).shape
    X = np.array(SeqOH['OneHot'].values.tolist()).reshape(Sequence_Samples,Sequence_Positions*Sequence_Bases)
    Y = SeqOH[Y_Col_Name].values
    groups = SeqOH['Sequence_letter-encrypted']
    Number_Estimators = np.arange(20,35,1)
    Max_Features = np.arange(9,15,1)
    param_grid = [{'bootstrap':[False], 'n_estimators': Number_Estimators, 'max_features': Max_Features}]
    # Group shuffle split removes groups with identical sequences from the development set
    # This is more realistic for parameter estimation
    cv = GroupShuffleSplit(n_splits=Num, test_size=Validation_cutoff, random_state=42)

    forest_grid = RandomForestRegressor()
    grid_forest = GridSearchCV(forest_grid, param_grid, cv=cv, n_jobs=-1)
    grid_forest.fit(X, Y, groups)
    
    Feature_Importance = np.array(grid_forest.best_estimator_.feature_importances_).reshape(-1,4)

    return grid_forest, Feature_Importance

def Est_Grad_Feat(SeqOH, Validation_cutoff=.1, Num=100, Y_Col_Name='promoter activity', AddFeat=None):
    '''
    This function performs gradient search for optimal parameters with shuffle shift and stores it.
    
    Input:
        DataArray
    '''
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
#    from sklearn.metrics import r2_score
    import numpy as np

    Sequence_Samples, Sequence_Positions, Sequence_Bases = np.array(SeqOH['OneHot'].values.tolist()).shape
    X = np.array(SeqOH['OneHot'].values.tolist()).reshape(Sequence_Samples,Sequence_Positions*Sequence_Bases)
    # adding rows to x for additional features
    if AddFeat != None:
        X = np.append(X,np.array(SeqOH[AddFeat]), axis=1)
    Y = SeqOH[Y_Col_Name].values
    groups = SeqOH['Sequence_letter-encrypted']
    Number_Estimators = np.arange(20,35,1)
    Max_Features = np.arange(9,15,1)
    param_grid = [{'bootstrap':[False], 'n_estimators': Number_Estimators, 'max_features': Max_Features}]
    # Group shuffle split removes groups with identical sequences from the development set
    # This is more realistic for parameter estimation
    cv = GroupShuffleSplit(n_splits=Num, test_size=Validation_cutoff, random_state=42)

    forest_grid = RandomForestRegressor()
    grid_forest = GridSearchCV(forest_grid, param_grid, cv=cv, n_jobs=-1)
    grid_forest.fit(X, Y, groups)
    
#     # remember to extract the features before rearranging the feature importance to a sequence position matrix form
#     if AddFeat != None:
#         Feat_num = len(AddFeat)
#         Feature_Importance_Nucl = np.array(grid_forest.best_estimator_.feature_importances_[0:-Feat_num]).reshape(-1,4)
#         Feature_Importance_Eng = grid_forest.best_estimator_.feature_importances_[-Feat_num]
#     else:
#         Feature_Importance_Nucl = np.array(grid_forest.best_estimator_.feature_importances_).reshape(-1,4)
#         Feature_Importance_Eng = []
        
    return grid_forest #, Feature_Importance_Nucl, Feature_Importance_Eng

def SequenceRandomizer_Parallel(RefSeq, Base_SequencePosition, n=1000):
    '''
    This function generates random sequence combinations. It takes the reference sequence and changes nucleotides at positions that have been experimentally tested. Only as much nucleotides are changed to remain within a given sequence distance.
    '''
    import numpy as np
    import multiprocessing
    from joblib import Parallel, delayed
    from ExpressionExpert_Functions import SequenceRandomizer_Single
    
    num_cores = multiprocessing.cpu_count()
    use_core = min([num_cores, n])
  
    Result = Parallel(n_jobs=use_core)(delayed(SequenceRandomizer_Single)(RefSeq, Base_SequencePosition) for idx in range(n))

    return Result # Sequence_multiple

def SequenceRandomizer_Single(RefSeq, Base_SequencePosition):
    
    import numpy as np
    import random
    from ExpressionExpert_Functions import list_integer
    
    Alphabet = ['A','C','G','T']
    
    #   Maximum number of nucleotides that can be changed simultaneously based on the sequence distance cut-off
    Nucleotide_Replace_Numb = len(Base_SequencePosition)
    # Generating the positions with nucleotide randomization
    MySynChange = np.array(random.sample(list(Base_SequencePosition.index), Nucleotide_Replace_Numb))
    # the following dataframe has the information of experimentally tested nucleotides as boolean table
    mytmp = Base_SequencePosition.loc[MySynChange]
    # The boolean table of tested nucleotides is converted into an array containing the explicit nucleotide letters
    myArr = np.tile(Alphabet, (Nucleotide_Replace_Numb,1))
    # following, non-tested nucleotides are deleted
    Pos_Del, Nucl_Del = np.where(mytmp.values == 0)
    if not Pos_Del.any(): # != '':
        print('deleting non-tested nucleotides')        
        myArr[tuple([Pos_Del,Nucl_Del])] = 'X'
    # Generating a reference sequence to work with
    TstSeq = list(RefSeq)
    # Changing indices from nucleotide oriented to array oriented
    ArSynChange = MySynChange + len(RefSeq)
    # setting the reference nucleotides at the positions to 'X', if they should not be chosen for new random sequences. This maximizes the distance of all random sequences, dont use it by default
    # RefSeq_IntLab = np.array(list_integer(RefSeq))
    # myArr[[range(Nucleotide_Replace_Numb)], [np.reshape(RefSeq_IntLab[ArSynChange], -1)]] = 'X'

    # converting the nucleotide array to a list, so we can delete non-tested nucleotides
    Position_list = myArr.tolist()
    Seq_Base = list()
    for Position in Position_list:
        Seq_Base.append(list(set(Position).difference(set('X'))))
    # print(Seq_Base)
    # randomly choosing a possible nucleotide over the total number of exchange positions
    Replace_Bases = [PosIdx[np.random.randint(len(PosIdx))] for PosIdx in Seq_Base]
    #   Replacing the bases in the reference sequence
    for MutIdx, MutNucl in zip(ArSynChange, Replace_Bases):
        TstSeq[MutIdx] = MutNucl    
    Sequence_Single = ''.join(TstSeq)
    
    return Sequence_Single

###############################################################################
# visualization
def ExpressionStrength_HeatMap(SeqList, Y_Col_Name = 'promoter activity'):
    '''
    Calculating the base and position specific average expression strength.
    Input:
        SeqList_df: dataframe, contains OneHot encoded sequence labelled 'OneHot' and 
                            expression strength parameters labelled as in variable 'Y_Col_Name'
    Output:
        Expression_HeatMap_df: dataframe, contains the average expression strength for each base on each position
    '''
    import numpy as np
    import pandas as pd
    
    # extracting the One-Hot encoding for all sequences
    my_rows = SeqList['OneHot'].shape
    Seq_OneHot_ar = np.array(list(SeqList['OneHot'])).reshape(my_rows[0],-1)
    # getting the number of positions considered
    myBasePos = Seq_OneHot_ar.shape[1]
    Exp_mult_ar = np.tile(np.array(SeqList[Y_Col_Name]),[myBasePos,1]).T
    Expr_OneHot_mean_ar = np.mean(np.multiply(Seq_OneHot_ar, Exp_mult_ar), axis=0)
    Expr_OneHot_mean_ar[Expr_OneHot_mean_ar==0.] = np.nan
    Expression_HeatMap_df = pd.DataFrame(Expr_OneHot_mean_ar.reshape(-1,4), columns=['A','C','G','T'])
    return Expression_HeatMap_df

def ExpressionVariation_HeatMap(SeqList, Y_Col_Name = 'promoter activity'):
    '''
    Calculating the base and position specific variation of expression strength.
    Input:
        SeqList_df: dataframe, contains OneHot encoded sequence labelled 'OneHot' and 
                            expression strength parameters labelled as in variable 'Y_Col_Name'
    Output:
        Variation_HeatMap_df: dataframe, contains the variation of expression strength for each base on each position
    '''
    import numpy as np
    import pandas as pd
    
    # extracting the One-Hot encoding for all sequences
    my_rows = SeqList['OneHot'].shape
    Seq_OneHot_ar = np.array(list(SeqList['OneHot'])).reshape(my_rows[0],-1)
    # getting the number of positions considered
    myBasePos = Seq_OneHot_ar.shape[1]
    Exp_mult_ar = np.tile(np.array(SeqList[Y_Col_Name]),[myBasePos,1]).T
    Expr_OneHot_var_ar = np.var(np.multiply(Seq_OneHot_ar, Exp_mult_ar), axis=0)
    Expr_OneHot_var_ar[Expr_OneHot_var_ar==0.] = np.nan
    Expression_HeatMap_df = pd.DataFrame(Expr_OneHot_var_ar.reshape(-1,4), columns=['A','C','G','T'])
    return Expression_HeatMap_df

def df_HeatMaps(Data_df, Z_Label, Plot_Save=False, Plot_File='dummy', cbar_lab=None):
    '''
    Function for heat map generation
    Input:
            Data_df: dataframe, rows represent sequence position, columns represent bases with their names as labels
            Z_Label: string, label for the colorbar
            Plot_Save: boolean, decision whether to save plot
            Plot_File: string, name for the figure file
            cbar_lab: sting vector, names for the color bar
    '''
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, ax = plt.subplots()
    im = ax.pcolor(Data_df.T, cmap='rainbow')
        
    # identifying dataframe dimensions
    my_rows, my_cols = Data_df.shape
    
    #label names
    my_Seq_range = np.arange(0,my_rows,5)
    row_labels = np.arange(-my_rows,0,5)
    col_labels = Data_df.T.index
    #move ticks and labels to the center
    ax.set_xticks(ticks=my_Seq_range+0.5, minor=False)
    ax.set_yticks(np.arange(Data_df.T.shape[0])+0.5, minor=False)
    #insert labels
    ax.set_xticklabels(row_labels, minor=False)
    ax.set_yticklabels(col_labels, minor=False)
    #rotate label if too long
#     plt.xticks(rotation=90)

    plt.xlabel('Position')
    plt.ylabel('Base')
    # plt.title('asdf')
    if cbar_lab is not None:
        tick_num = len(cbar_lab)
        myDat = Data_df.values.reshape(-1)
        myticks = np.histogram(myDat[~np.isnan(myDat)], bins=tick_num-1)[1]
        cbar = fig.colorbar(im, ticks=myticks, label=Z_Label)
        cbar.ax.set_yticklabels(cbar_lab)
    else:
        fig.colorbar(im, label=Z_Label)
        
    if Plot_Save:
        Fig_Type = Plot_File.split('.')[1]
        plt.savefig(Plot_File, format=Fig_Type)
    plt.show()

def my_CrossValScore(X, Y, groups, cv, ML_fun, metric):
    '''
    Function to generate statistics according to cross validation scoring
    '''
    import numpy as np
    
    mytrain = list()
    mytest = list()
    tst_num = cv.n_splits
    for idx in range(tst_num):
        cv.n_splits = 1
        mycv = list(cv.split(X, Y, groups))
        onetrain = mycv[0][0]
        onetest = mycv[0][1]
        ML_fun.fit(X[onetrain], Y[onetrain])
        mytrain.append({'train':metric(ML_fun, X[onetest], Y[onetest])})
#         myscores.append(metric(ML_fun, X[onetest], Y[onetest]))
        
    myscores = dict({'TrainR2':mytrain})
    return myscores
    
def my_SVR(SeqOH, Y_Col_Name):
    '''
    My SVR code
    '''
    from sklearn import svm
    import numpy as np

    Sequence_Samples, Sequence_Positions, Sequence_Bases = np.array(SeqOH['OneHot'].values.tolist()).shape
    X = np.array(SeqOH['OneHot'].values.tolist()).reshape(Sequence_Samples,Sequence_Positions*Sequence_Bases)
    Y = SeqOH[Y_Col_Name].values

    SVR = svm.SVR(C=128, kernel='rbf', gamma=24.25)
    SVR.fit(X, Y)
        
    return SVR