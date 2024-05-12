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

from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

def supervised_model_training(x_train, y_train, x_test, 
                              y_test, model_name,problem_type, sample_weights=None):
  
  if model_name == 'lr':
    model  = LogisticRegression(random_state=23,max_iter=1500, class_weight="balanced")
  elif model_name == 'svm':
    model  = svm.SVC(random_state=23,probability=True)
  elif model_name == 'dt':
    model = tree.DecisionTreeClassifier(random_state=23, max_depth=None, min_samples_split=2, min_samples_leaf=1, class_weight="balanced")
  elif model_name == 'rf':      
    model = RandomForestClassifier(n_estimators=300, random_state=23, class_weight="balanced")
  elif model_name == "mlp":
    model = MLPClassifier(random_state=23,max_iter=300, batch_size=2000)
  elif model_name == "xgb":
    model = XGBClassifier(random_state=23, objective="multi:softmax", num_class=5, eval_metric="mlogloss", n_estimators=300)
  elif model_name == "l_reg":
    model = LinearRegression()
  elif model_name == "ridge":
    model = Ridge(random_state=42)
  elif model_name == "lasso":
    model = Lasso(random_state=42)
  elif model_name == "B_ridge":
    model = BayesianRidge()
  
  if model_name in ["lr","dt","xgb"]:
    model.fit(x_train, y_train, sample_weight=sample_weights)
  else:
    model.fit(x_train, y_train)
  pred = model.predict(x_test)

  if problem_type == "Classification":
    if len(np.unique(y_train))>2:
      predict = model.predict_proba(x_test)        
      acc = metrics.balanced_accuracy_score(y_test,pred)*100
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

def get_utility_metrics(real_data,test_data,fake_paths,scaler="MinMax",type={"Classification":["lr","dt","rf","mlp"]}):

    data_real = real_data.to_numpy()
    data_dim = data_real.shape[1]

    data_real_y = data_real[:,-1]
    data_real_X = data_real[:,:data_dim-1]
    
    data_test = test_data.to_numpy()
    data_dim = test_data.shape[1]

    data_test_y = data_test[:,-1]
    data_test_X = data_test[:,:data_dim-1]

    problem = list(type.keys())[0]
    
    models = list(type.values())[0]
    
    X_train_real = data_real_X
    X_test_real = data_test_X
    y_train_real = data_real_y
    y_test_real = data_test_y
    
    if scaler=="MinMax":
        scaler_real = MinMaxScaler()
    else:
        scaler_real = StandardScaler()
        
    scaler_real.fit(X_train_real)
    X_train_real_scaled = scaler_real.transform(X_train_real)
    X_test_real_scaled = scaler_real.transform(X_test_real)

    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train_real)
    all_real_results = []
    real_cr = pd.DataFrame()
    for model in models:
      real_results, real_classification_report = supervised_model_training(X_train_real_scaled,y_train_real,X_test_real_scaled,y_test_real,model,problem, sample_weights=sample_weights)
      print("Model: ", model,"trained on real data")
      all_real_results.append(real_results)
      real_cr = cr_processing(real_classification_report, real_cr, model)
      
    all_fake_results_avg = []
    
    data_fake  = fake_paths.to_numpy()
    data_fake_y = data_fake[:,-1]
    data_fake_X = data_fake[:,:data_dim-1]

    X_train_fake = data_fake_X
    y_train_fake = data_fake_y  

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

    real_df = pd.DataFrame(all_real_results,columns=["Acc","AUC","F1_Score"])
    real_df.index = list(models)
    real_df.index.name = "Model"
    real_df["Model"] = real_df.index
    real_df["Type"] = "Real"

    fake_df = pd.DataFrame(all_fake_results,columns=["Acc","AUC","F1_Score"])
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
            print(column, real_pdf)
            fake_pdf=(fakey[column].value_counts()/fakey[column].value_counts().sum())
            print(column, fake_pdf)
            categories = (fakey[column].value_counts()/fakey[column].value_counts().sum()).keys().tolist()
            sorted_categories = sorted(categories)
            print("sorted categories: ", sorted_categories)
            real_pdf_values = [] 
            fake_pdf_values = []

            for i in sorted_categories:
                print("i: ", i)
                real_pdf_values.append(real_pdf[i])
                fake_pdf_values.append(fake_pdf[i])
                print("real_pdf_values: ", real_pdf_values)
                print("fake_pdf_values: ", fake_pdf_values)
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