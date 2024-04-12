from sklearn.preprocessing import LabelEncoder
import pandas as pd
import seaborn as sns
import plotly.io as pio
import matplotlib.pyplot as plt
from sdmetrics.reports.single_table import DiagnosticReport
from sdmetrics.reports.single_table import QualityReport
pio.renderers.default = 'iframe'
from table_evaluator import TableEvaluator

def eval_model(model, real_data, syn_data, metadata, op_dir):
    
    if model == 'ctabgan':
        columns = ["attack_type", "label", "proto", "day_of_week"]
        for c in columns:
            exec(f'le_{c} = LabelEncoder()')
            real_data[c] = globals()[f'le_{c}'].fit_transform(real_data[c])
            real_data[c] = real_data[c].astype("int64")
            
    real_data.drop(columns=["tcp_urg"], inplace=True)
    syn_data.drop(columns=["tcp_urg"], inplace=True)
    scores = {}
    
    print("Statistical Analysis: Data Distribution, Cumsum, Statistical Similarity and Correlation")
    table_eval = TableEvaluator(real_data, syn_data, verbose=True)  # Added a comma between real_data and syn_data
    table_eval.visual_evaluation(op_dir)
    
    print("Statistical Analysis complete. Check the output folder for the plots.")
    
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
    
    cpt_plot = quality_report.get_visualization(property_name="Column Pair Trends")
    pio.write_image(cpt_plot, str(op_dir + "column_pair_trends.png"), format='png', engine='kaleido')
    scores["\n column_pair_trends"] = quality_report.get_details(property_name='Column Pair Trends').Score.mean()
    print("\n Column Shapes score: ", str(column_shapes),
          "\n Column Pair Trends score: ", str(), "...Plotted graphs for reference.")

    return scores

def create_bar_plot(data, x, y, hue, op_dir, name):
# Create bar plot
    plt.clf()
    plt.cla()
    sns.set_theme(rc={"figure.figsize":(28, 6), 'axes.labelsize': 10, 'legend.fontsize': 10})
    fig = sns.barplot(data=data, x=x, y=y, hue=hue)
    fig.bar_label(fig.containers[0])
    plt.savefig(str(op_dir + name + ".png"))
