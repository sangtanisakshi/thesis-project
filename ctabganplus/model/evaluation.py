import numpy as np
import pandas as pd 
from sklearn import metrics
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, BayesianRidge
from sklearn import svm,tree
from sklearn.ensemble import RandomForestClassifier
import glob
from dython.nominal import associations
from scipy.stats import wasserstein_distance
from scipy.spatial import distance
import warnings

warnings.filterwarnings("ignore")

def supervised_model_training(x_train, y_train, x_test, 
                              y_test, model_name,problem_type):
  
  if model_name == 'lr':
    model  = LogisticRegression(random_state=42,max_iter=500) 
  elif model_name == 'svm':
    model  = svm.SVC(random_state=42,probability=True)
  elif model_name == 'dt':
    model  = tree.DecisionTreeClassifier(random_state=42)
  elif model_name == 'rf':      
    model = RandomForestClassifier(random_state=42)
  elif model_name == "mlp":
    model = MLPClassifier(random_state=42,max_iter=100)
  elif model_name == "l_reg":
    model = LinearRegression()
  elif model_name == "ridge":
    model = Ridge(random_state=42)
  elif model_name == "lasso":
    model = Lasso(random_state=42)
  elif model_name == "B_ridge":
    model = BayesianRidge()
  
  model.fit(x_train, y_train)
  pred = model.predict(x_test)

  if problem_type == "Classification":
    if len(np.unique(y_train))>2:
      predict = model.predict_proba(x_test)        
      acc = metrics.balanced_accuracy_scoreaccuracy_score(y_test,pred)*100
      auc = metrics.roc_auc_score(y_test, predict,average="macro",multi_class="ovo")
      f1_score = metrics.precision_recall_fscore_support(y_test, pred,average="macro")[2]
      classification_report = metrics.classification_report(y_test,pred, target_names=['benign', 'bruteForce', 'dos', 'pingScan', 'portScan'])
      return [acc, auc, f1_score], classification_report

    else:
      predict = model.predict_proba(x_test)[:,1]    
      acc = metrics.accuracy_score(y_test,pred)*100
      auc = metrics.roc_auc_score(y_test, predict)
      f1_score = metrics.precision_recall_fscore_support(y_test,pred)[2].mean()
      return [acc, auc, f1_score] 
  
  else:
    mse = metrics.mean_absolute_percentage_error(y_test,pred)
    evs = metrics.explained_variance_score(y_test, pred)
    r2_score = metrics.r2_score(y_test,pred)
    return [mse, evs, r2_score]

def cr_processing(a, cr_df, model):

  a = a.split("\n")
  a = [x.split() for x in a]
  a[0] = ['attack_type', 'precision', 'recall', 'f1-score', 'support']
  a = [x for x in a if x]
  a = {x[0]:x[1:] for x in a}

  #if the length of the values is less than 4, add empty strings to the left to make it 4
  for k,v in a.items():
      if len(v) < 4:
          a[k] = ["0.0"]*(4-len(v)) + v
          
  #if length of the values is more than 4, remove the first value
  for k,v in a.items():
      if len(v) > 4:
          a[k] = v[1:]
          
  a = pd.DataFrame(a, columns=a.keys()).transpose()
  #convert index to a column
  a.reset_index(inplace=True)
  a.rename(columns={"index":"attack_type"}, inplace=True)
  a.columns = a.iloc[0].str.strip()
  #remove first row of a
  a = a[1:]
  a["Model"] = model

  #concat a with b
  cr_df = pd.concat([cr_df,a])
  cr_df.reset_index(drop=True, inplace=True)
  
  return cr_df

def get_utility_metrics(real_data,fake_paths,scaler="MinMax",type={"Classification":["lr","dt","rf","mlp"]},test_ratio=.20):

    data_real = real_data.to_numpy()
    data_dim = data_real.shape[1]

    data_real_y = data_real[:,-1]
    data_real_X = data_real[:,:data_dim-1]

    problem = list(type.keys())[0]
    
    models = list(type.values())[0]
    
    if problem == "Classification":
      X_train_real, X_test_real, y_train_real, y_test_real = model_selection.train_test_split(data_real_X ,data_real_y, test_size=test_ratio, stratify=data_real_y,random_state=42) 
    else:
      X_train_real, X_test_real, y_train_real, y_test_real = model_selection.train_test_split(data_real_X ,data_real_y, test_size=test_ratio,random_state=42) 
    

    if scaler=="MinMax":
        scaler_real = MinMaxScaler()
    else:
        scaler_real = StandardScaler()
        
    scaler_real.fit(X_train_real)
    X_train_real_scaled = scaler_real.transform(X_train_real)
    X_test_real_scaled = scaler_real.transform(X_test_real)

    all_real_results = []
    real_cr = pd.DataFrame()
    for model in models:
      real_results, real_classification_report = supervised_model_training(X_train_real_scaled,y_train_real,X_test_real_scaled,y_test_real,model,problem)
      print("Model: ", model,"trained on real data")
      all_real_results.append(real_results)
      real_cr = cr_processing(real_classification_report, real_cr, model)
      
    all_fake_results_avg = []
    
    data_fake  = fake_paths.to_numpy()
    data_fake_y = data_fake[:,-1]
    data_fake_X = data_fake[:,:data_dim-1]

    if problem=="Classification":
      X_train_fake, _ , y_train_fake, _ = model_selection.train_test_split(data_fake_X ,data_fake_y, test_size=test_ratio, stratify=data_fake_y,random_state=42) 
    else:
      X_train_fake, _ , y_train_fake, _ = model_selection.train_test_split(data_fake_X ,data_fake_y, test_size=test_ratio,random_state=42)  

    if scaler=="MinMax":
      scaler_fake = MinMaxScaler()
    else:
      scaler_fake = StandardScaler()
    
    scaler_fake.fit(data_fake_X)
    
    X_train_fake_scaled = scaler_fake.transform(X_train_fake)
    
    all_fake_results = []
    fake_cr = pd.DataFrame()
    for model in models:
      fake_results, fake_classification_report = supervised_model_training(X_train_fake_scaled,y_train_fake,X_test_real_scaled,y_test_real,model,problem)
      print("Model: ", model, "trained on fake data")
      all_fake_results.append(fake_results)
      fake_cr = cr_processing(fake_classification_report, fake_cr, model)
    
    all_fake_results_avg.append(all_fake_results)
    
    diff_results = np.array(all_real_results)- np.array(all_fake_results_avg).mean(axis=0)

    real_cr["precision"] = (real_cr["precision"]).astype("float64")
    real_cr["recall"] = real_cr["recall"].astype("float64")
    real_cr["f1-score"] = real_cr["f1-score"].astype("float64")
    real_cr["support"] = real_cr["support"].astype("int64")

    fake_cr["precision"] = fake_cr["precision"].astype("float64")
    fake_cr["recall"] = fake_cr["recall"].astype("float64")
    fake_cr["f1-score"] = fake_cr["f1-score"].astype("float64")
    fake_cr["support"] = fake_cr["support"].astype("int64")

    real_cr["type"] = "real"
    fake_cr["type"] = "fake"
    
    #get the difference between precision, recall and f1-score in real and fake data
    diff_cr = real_cr.copy()
    diff_cr["precision"] = real_cr["precision"] - fake_cr["precision"]
    diff_cr["recall"] = real_cr["recall"] - fake_cr["recall"]
    diff_cr["f1-score"] = real_cr["f1-score"] - fake_cr["f1-score"]
    diff_cr["support"] = real_cr["support"] - fake_cr["support"]
    diff_cr["type"] = "diff"

    cr = pd.concat([real_cr,fake_cr,diff_cr])
    
        # get real, fake and diff results and put the acc, auc and f1 scores in a dataframe
    diff_df = pd.DataFrame(diff_results,columns=["Acc","AUC","F1_Score"])
    diff_df.index = list(models)
    diff_df.index.name = "Model"
    diff_df["Model"] = diff_df.index
    diff_df["Type"] = "Difference"

    real_df = pd.DataFrame(real_results,columns=["Acc","AUC","F1_Score"])
    real_df.index = list(models)
    real_df.index.name = "Model"
    real_df["Model"] = real_df.index
    real_df["Type"] = "Real"

    fake_df = pd.DataFrame(fake_results,columns=["Acc","AUC","F1_Score"])
    fake_df.index = list(models)
    fake_df.index.name = "Model"
    fake_df["Model"] = fake_df.index
    fake_df["Type"] = "Fake"

    #concatenate the dataframes
    result_df = pd.concat([real_df,fake_df,diff_df])
    result_df = result_df.reset_index(drop=True)
    
    return result_df, cr

def stat_sim(real_data,fake_path,cat_cols=None):
    
    Stat_dict={}
    
    real = real_data
    fake = fake_path

    really = real.copy()
    fakey = fake.copy()

    real_corr = associations(real, nominal_columns=cat_cols, compute_only=True)['corr']

    fake_corr = associations(fake, nominal_columns=cat_cols, compute_only=True)['corr']

    corr_dist = np.linalg.norm(real_corr - fake_corr)
    
    cat_stat = []
    num_stat = []
    
    for column in real.columns:
        
        if column in cat_cols:

            real_pdf=(really[column].value_counts()/really[column].value_counts().sum())
            fake_pdf=(fakey[column].value_counts()/fakey[column].value_counts().sum())
            categories = (fakey[column].value_counts()/fakey[column].value_counts().sum()).keys().tolist()
            sorted_categories = sorted(categories)
            
            real_pdf_values = [] 
            fake_pdf_values = []

            for i in sorted_categories:
                real_pdf_values.append(real_pdf[i])
                fake_pdf_values.append(fake_pdf[i])
            
            if len(real_pdf)!=len(fake_pdf):
                zero_cats = set(really[column].value_counts().keys())-set(fakey[column].value_counts().keys())
                for z in zero_cats:
                    real_pdf_values.append(real_pdf[z])
                    fake_pdf_values.append(0)
            Stat_dict[column]=(distance.jensenshannon(real_pdf_values,fake_pdf_values, 2.0))
            cat_stat.append(Stat_dict[column])    
            print("column: ", column, "JSD: ", Stat_dict[column])  
        else:
            scaler = MinMaxScaler()
            scaler.fit(real[column].values.reshape(-1,1))
            l1 = scaler.transform(real[column].values.reshape(-1,1)).flatten()
            l2 = scaler.transform(fake[column].values.reshape(-1,1)).flatten()
            Stat_dict[column]= (wasserstein_distance(l1,l2))
            print("column: ", column, "WD: ", Stat_dict[column])
            num_stat.append(Stat_dict[column])

    return [np.mean(num_stat),np.mean(cat_stat),corr_dist]

def privacy_metrics(real_data,fake_path,data_percent=15):
    
    real = real_data
    fake = fake_path.drop_duplicates(keep=False)

    real_refined = real.sample(n=int(len(real)*(.01*data_percent)), random_state=42).to_numpy()
    fake_refined = fake.sample(n=int(len(fake)*(.01*data_percent)), random_state=42).to_numpy()

    scalerR = StandardScaler()
    scalerR.fit(real_refined)
    scalerF = StandardScaler()
    scalerF.fit(fake_refined)
    df_real_scaled = scalerR.transform(real_refined)
    df_fake_scaled = scalerF.transform(fake_refined)
    
    dist_rf = metrics.pairwise_distances(df_real_scaled, Y=df_fake_scaled, metric='minkowski', n_jobs=-1)
    dist_rr = metrics.pairwise_distances(df_real_scaled, Y=None, metric='minkowski', n_jobs=-1)
    rd_dist_rr = dist_rr[~np.eye(dist_rr.shape[0],dtype=bool)].reshape(dist_rr.shape[0],-1)
    dist_ff = metrics.pairwise_distances(df_fake_scaled, Y=None, metric='minkowski', n_jobs=-1)
    rd_dist_ff = dist_ff[~np.eye(dist_ff.shape[0],dtype=bool)].reshape(dist_ff.shape[0],-1) 
    smallest_two_indexes_rf = [dist_rf[i].argsort()[:2] for i in range(len(dist_rf))]
    smallest_two_rf = [dist_rf[i][smallest_two_indexes_rf[i]] for i in range(len(dist_rf))]       
    smallest_two_indexes_rr = [rd_dist_rr[i].argsort()[:2] for i in range(len(rd_dist_rr))]
    smallest_two_rr = [rd_dist_rr[i][smallest_two_indexes_rr[i]] for i in range(len(rd_dist_rr))]
    smallest_two_indexes_ff = [rd_dist_ff[i].argsort()[:2] for i in range(len(rd_dist_ff))]
    smallest_two_ff = [rd_dist_ff[i][smallest_two_indexes_ff[i]] for i in range(len(rd_dist_ff))]
    nn_ratio_rr = np.array([i[0]/i[1] for i in smallest_two_rr])
    nn_ratio_ff = np.array([i[0]/i[1] for i in smallest_two_ff])
    nn_ratio_rf = np.array([i[0]/i[1] for i in smallest_two_rf])
    nn_fifth_perc_rr = np.percentile(nn_ratio_rr,5)
    nn_fifth_perc_ff = np.percentile(nn_ratio_ff,5)
    nn_fifth_perc_rf = np.percentile(nn_ratio_rf,5)

    min_dist_rf = np.array([i[0] for i in smallest_two_rf])
    fifth_perc_rf = np.percentile(min_dist_rf,5)
    min_dist_rr = np.array([i[0] for i in smallest_two_rr])
    fifth_perc_rr = np.percentile(min_dist_rr,5)
    min_dist_ff = np.array([i[0] for i in smallest_two_ff])
    fifth_perc_ff = np.percentile(min_dist_ff,5)
    
    return np.array([fifth_perc_rf,fifth_perc_rr,fifth_perc_ff,nn_fifth_perc_rf,nn_fifth_perc_rr,nn_fifth_perc_ff]).reshape(1,6)    