from sklearn.preprocessing import LabelEncoder
import pandas as pd
import seaborn as sns
import plotly.io as pio
import matplotlib.pyplot as plt
from sdmetrics.reports.single_table import DiagnosticReport
from sdmetrics.reports.single_table import QualityReport
pio.renderers.default = 'iframe'
from table_evaluator import TableEvaluator

def eval_model(real_data, syn_data, metadata, op_dir):
    
    scores = {}
    print("\n Diagnostic Report: Data Validity and Structure")
    diagnostic = DiagnosticReport()
    diagnostic.generate(real_data, syn_data, metadata, verbose=True)
    
    if (diagnostic.get_score() < 100.000):
        
        data_validity = diagnostic.get_details(property_name='Data Validity')
        scores["data_validity"] = data_validity.Score.mean()
        scores["data structure"] = diagnostic.get_details(property_name='Data Structure').Score.mean()
        print("Please check if the columns match if ds score is less than 100.")
        create_bar_plot(data_validity, x='Column', y='Score', hue='Metric', op_dir=op_dir, name="data_validity")
        print("Data validity and structure score is less than 100.00, plotted graph for reference.")
        
    else:
        print("Data validity and structure score is 100.00, no need to plot.")
    
    print("\n Quality Report: Column Shapes and Column Pair Trends")
    quality_report = QualityReport()
    quality_report.generate(real_data, syn_data, metadata)
    
    column_shapes = quality_report.get_details(property_name='Column Shapes')
    scores["column_shapes"] = column_shapes.Score.mean()
    create_bar_plot(column_shapes, x='Column', y='Score', hue='Metric', op_dir=op_dir, name="column_shapes")
    
    column_pair_trends = quality_report.get_details(property_name='Column Pair Trends')
    scores["column_pair_trends"] = column_pair_trends.Score.mean()
    print("\n Column Shapes score: ", str(column_shapes),
          "\n Column Pair Trends score: ", str(), "...Plotted graphs for reference.")

    real = real_data.copy()
    syn = syn_data.copy()
    #remove columns with only one unique value
    for col in real_data.columns:
        if len(real_data[col].unique()) == 1:
            print(f"Removing column {col} as it has only one unique value")
            real.drop(columns=[col], inplace=True)
            syn.drop(columns=[col], inplace=True)
            
    print("Statistical Analysis: Data Distribution, Cumsum, Statistical Similarity and Correlation")
    table_eval = TableEvaluator(real, syn, verbose=True)  # Added a comma between real_data and syn_data
    table_eval.visual_evaluation(op_dir)
    
    print("Statistical Analysis complete. Check the output folder for the plots.")
    return scores

def create_bar_plot(data, x, y, hue, op_dir, name):
    plt.clf()
    plt.cla()
    fig = plt.figure(figsize=(35, 6))
    ax = fig.add_subplot(111)
    sns.barplot(x=x, y=y, hue=hue, data=data, ax=ax)
    ax.bar_label(ax.containers[0])
    ax.bar_label(ax.containers[1])
    plt.savefig(str(op_dir + name + ".png"))

