# Inshrah-Ai-project
I have created my second project 
this project is based on dataset https://www.kaggle.com/code/prthmgoyl/neuralnetwork-heart-disease-dataset?select=heart.csv
i have trained models knn , neural network , naive biase and showed there working 

import os
os.environ["OMP_NUM_THREADS"] = "4"  # Set the number of threads for OpenMP  ,to avoid memory leakage
import dash
from dash import dcc, html, Input, Output
from dash.dash_table import DataTable
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, adjusted_rand_score
import webbrowser
import threading

# Load dataset
try:
    df = pd.read_csv('heart .csv')  # Load the heart disease dataset
except Exception as e:
    print("Using demo data:", e)  # If loading fails, use demo data
    df = pd.DataFrame({
        'age': np.random.randint(29, 77, 100),
        'chol': np.random.randint(150, 300, 100),
        'thalach': np.random.randint(100, 200, 100),
        'target': np.random.randint(0, 2, 100)
    })

# Prepare features and target
X = df.drop(columns=['target'])  # Features (input variables)
y = df['target']  # Target variable (output)

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Split data into training and testing sets
scaler = StandardScaler()  # Initialize the scaler
X_train_scaled = scaler.fit_transform(X_train)  # Scale training data
X_test_scaled = scaler.transform(X_test)  # Scale testing data

app = dash.Dash(__name__, suppress_callback_exceptions=True)  # Initialize the Dash app

# Links to model tutorials
MODEL_LINKS = {
    "Naive Bayes": "https://www.youtube.com/watch?v=O2L2Uv9pdDA",
    "K-Nearest Neighbors (KNN)": "https://www.youtube.com/watch?v=HVXime0nQeI",
    "K-Means Clustering": "https://www.youtube.com/watch?v=4b5d3muPQmA",
    "Neural Networks": "https://www.youtube.com/watch?v=aircAruvnKk"
}

# Color scales for visualizations
COLOR_SCALES = [
    {'label': 'Viridis', 'value': 'Viridis'},
    {'label': 'Plasma', 'value': 'Plasma'},
    {'label': 'Inferno', 'value': 'Inferno'},
    {'label': 'Magma', 'value': 'Magma'},
    {'label': 'Cividis', 'value': 'Cividis'},
]

def create_graph(fig):
    # Function to create a graph with unified styling
    fig.update_layout(
        template='plotly_dark',  # Use dark template for the graph
        plot_bgcolor='black',  # Set plot background color
        paper_bgcolor='black',  # Set paper background color
        font_color='#FFD700',  # Set font color
        xaxis=dict(showline=True, linecolor='white'),  # Customize x-axis
        yaxis=dict(showline=True, linecolor='white')   # Customize y-axis
    )
    return dcc.Graph(figure=fig)  # Return the graph component

def check_accuracy_threshold(acc, threshold=0.80):
    # Function to check if model accuracy meets the threshold
    if acc >= threshold:
        return html.P(f"✅ Model accuracy {acc:.2%} meets the requirement (≥ 80%).", 
                     style={'color': '#00FF00', 'fontWeight': 'bold', 'fontSize': '14px'})
    else:
        return html.P(f"❌ Model accuracy {acc:.2%} is below the required 80%. Consider tuning or using another model.", 
                     style={'color': '#FF4500', 'fontWeight': 'bold', 'fontSize': '14px'})

# Layout of the Dash app
app.layout = html.Div(
    style={
        'backgroundImage': 'url("https://img.freepik.com/free-photo/heart-banner-cardiac-technology_53876-104942.jpg?uid=R205250949&ga=GA1.1.835983211.1750618776&semt=ais_hybrid&w=740")',
        'height': '100vh',
        'backgroundSize': 'cover',
        'color': 'white',
        'textAlign': 'center',
        'paddingTop': '20px'
    },
    children=[
        html.H1("Heart Disease Dataset Analysis with ML Models", style={'fontSize': '30px', 'color': 'white'}),
        dcc.Dropdown(
            id='dropdown-menu',
            options=[
                {'label': 'Heart Disease Dataset Info', 'value': 'dataset'},
                {'label': 'Naive Bayes', 'value': 'naive_bayes'},
                {'label': 'K-Nearest Neighbors (KNN)', 'value': 'knn'},
                {'label': 'K-Means Clustering', 'value': 'kmeans'},
                {'label': 'Neural Networks', 'value': 'neural_networks'},
                {'label': 'About Models', 'value': 'about'}
            ],
            value='dataset',
            style={
                'width': '50%',
                'margin': 'auto',
                'backgroundColor': 'black',
                'color': 'red',
                'fontWeight': 'bold',
                'border': '2px solid red'
            },
            clearable=False
        ),
        html.Div(id='output-container', style={'marginTop': '20px', 'fontWeight': 'bold', 'fontSize': '14px', 'color': 'white'})
    ]
)

@app.callback(
    Output('output-container', 'children'),
    Input('dropdown-menu', 'value')
)
def update_output(value):
    # Callback function to update output based on selected dropdown value
    if value == 'dataset':
        return html.Div([
            html.Div([
                html.P(
                    f"📊 Total Samples: {df.shape[0]:,} | 🏷️ Features: {df.shape[1]-1}",
                    style={'fontSize': '16px', 'color': 'white', 'fontWeight': 'bold'}
                ),
                html.P(
                    "This heart disease dataset enables predictive modeling to assess cardiovascular risk.",
                    style={'fontWeight': 'bold', 'fontSize': '14px', 'color': 'white'}
                ),
                html.P(
                    "Naive Bayes leverages the dataset's statistical features for quick probability-based predictions.",
                    style={'fontWeight': 'bold', 'fontSize': '14px', 'color': 'white'}
                ),
                html.P(
                    "K-Nearest Neighbors classifies patients by comparing their health metrics to similar cases.",
                    style={'fontWeight': 'bold', 'fontSize': '14px', 'color': 'white'}
                ),
                html.P(
                    "K-Means Clustering reveals hidden patient groups, aiding in risk stratification.",
                    style={'fontWeight': 'bold', 'fontSize': '14px', 'color': 'white'}
                ),
                html.P(
                    "Neural Networks capture complex, non-linear relationships, providing highly accurate diagnostic support.",
                    style={'fontWeight': 'bold', 'fontSize': '14px', 'color': 'white'}
                ),
                html.Br(),
                dcc.Markdown(
                    "You can find the dataset [here](https://www.kaggle.com/code/prthmgoyl/neuralnetwork-heart-disease-dataset?select=heart.csv).",
                    style={'color': 'white', 'fontWeight': 'bold', 'fontSize': '14px'}
                )
            ])
        ])

    elif value == 'about':
        return html.Div(
            style={'marginTop': '0', 'padding': '0'},
            children=[
                html.H2("About the Models", style={'textAlign': 'center', 'color': 'white'}),
                DataTable(
                    columns=[
                        {"name": "Model", "id": "model"},
                        {"name": "Description", "id": "description"},
                        {"name": "Usage", "id": "usage"}
                    ],
                    data=[
                        {
                            "model": "Naive Bayes",
                            "description": "A probabilistic classifier based on Bayes' theorem.",
                            "usage": "Used in spam detection."
                        },
                        {
                            "model": "K-Nearest Neighbors (KNN)",
                            "description": "A simple, instance-based learning algorithm.",
                            "usage": "Used in recommendation systems."
                        },
                        {
                            "model": "K-Means Clustering",
                            "description": "An unsupervised learning algorithm for clustering.",
                            "usage": "Used in customer segmentation."
                        },
                        {
                            "model": "Neural Networks",
                            "description": "Deep learning models that mimic the human brain.",
                            "usage": "Used in image and speech recognition."
                        }
                    ],
                    style_table={'overflowX': 'auto'},
                    style_cell={
                        'textAlign': 'left',
                        'padding': '10px',
                        'fontSize': '14px',
                        'color': 'white',
                        'backgroundColor': 'rgba(0, 0, 0, 0.5)',
                    },
                    style_header={
                        'backgroundColor': 'black',
                        'fontWeight': 'bold',
                        'color': 'red'
                    }
                ),
                html.Div([
                    html.H3("Model Tutorials", style={'marginTop': '20px', 'textAlign': 'center', 'color': 'white'}),
                    html.Div([
                        html.P("Naive Bayes:", style={'marginTop': '10px'}),
                        html.A("Watch Tutorial", href=MODEL_LINKS["Naive Bayes"], target="_blank",
                               style={'color': 'lightblue', 'textDecoration': 'underline'}),
                        html.P("K-Nearest Neighbors (KNN):", style={'marginTop': '10px'}),
                        html.A("Watch Tutorial", href=MODEL_LINKS["K-Nearest Neighbors (KNN)"], target="_blank",
                               style={'color': 'lightblue', 'textDecoration': 'underline'}),
                        html.P("K-Means Clustering:", style={'marginTop': '10px'}),
                        html.A("Watch Tutorial", href=MODEL_LINKS["K-Means Clustering"], target="_blank",
                               style={'color': 'lightblue', 'textDecoration': 'underline'}),
                        html.P("Neural Networks:", style={'marginTop': '10px'}),
                        html.A("Watch Tutorial", href=MODEL_LINKS["Neural Networks"], target="_blank",
                               style={'color': 'lightblue', 'textDecoration': 'underline'}),
                    ], style={'textAlign': 'center', 'marginTop': '20px'})
                ])
            ]
        )

    else:
        return html.Div([
            html.Div(id='model-analysis-section'),
            html.Div([
                dcc.Dropdown(
                    id='graph-selector',
                    options=[
                        {'label': 'Confusion Matrix', 'value': 'confusion_matrix'},
                        {'label': 'Feature Distribution', 'value': 'feature_distribution'},
                        {'label': 'Correlation Matrix', 'value': 'correlation'},
                        {'label': 'Cluster Analysis', 'value': 'cluster_analysis'},
                        {'label': 'Feature Importance', 'value': 'feature_importance'}
                    ],
                    value='confusion_matrix',
                    style={
                        'width': '45%',
                        'margin': '20px 10px',
                        'backgroundColor': 'black',
                        'color': 'red',
                        'border': '2px solid red',
                        'display': 'inline-block'
                    }
                ),
                dcc.Dropdown(
                    id='color-scale-selector',
                    options=COLOR_SCALES,
                    value='Viridis',
                    style={
                        'width': '45%',
                        'margin': '20px 10px',
                        'backgroundColor': 'black',
                        'color': 'red',
                        'border': '2px solid red',
                        'display': 'inline-block'
                    }
                )
            ]),
            html.Div(id='selected-graph-display')
        ])

@app.callback(
    [Output('model-analysis-section', 'children'),
     Output('selected-graph-display', 'children')],
    [Input('dropdown-menu', 'value'),
     Input('graph-selector', 'value'),
     Input('color-scale-selector', 'value')]
)
def update_model_analysis(selected_model, graph_type, color_scale):
    fig = go.Figure()  # Initialize an empty figure
    model = None
    acc = 0
    cm = None
    clusters = None
    report = ""  # Initialize report variable

    # Model training and evaluation based on selected model
    if selected_model == 'naive_bayes':
        model = GaussianNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
    elif selected_model == 'knn':
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
    elif selected_model == 'kmeans':
        model = KMeans(n_clusters=2, random_state=42)
        clusters = model.fit_predict(X_train_scaled)
        ari = adjusted_rand_score(y_train, clusters)
        acc = ari
        report = f"Adjusted Rand Index: {ari:.2f}"
        
    elif selected_model == 'neural_networks':
        model = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
    # Generate visualizations based on selected graph type
    if graph_type == 'confusion_matrix' and cm is not None:
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted 0', 'Predicted 1'],
            y=['Actual 0', 'Actual 1'],
            colorscale=color_scale
        )).update_layout(title='Confusion Matrix')
        
    elif graph_type == 'feature_distribution':
        fig = px.histogram(df, x='age', color='target', 
                          nbins=20, title="Age Distribution",
                          color_discrete_sequence=getattr(px.colors.sequential, color_scale))
        
    elif graph_type == 'correlation':
        fig = px.imshow(df.corr(), title="Feature Correlation",
                       color_continuous_scale=color_scale)
        
    elif graph_type == 'cluster_analysis' and clusters is not None:
        df_temp = X_train.copy()
        df_temp['Cluster'] = clusters
        fig = px.scatter(df_temp, x='age', y='chol', 
                        color='Cluster', title="Cluster Analysis",
                        color_continuous_scale=color_scale)
        
    elif graph_type == 'feature_importance':
        importance = np.abs(model.coef_[0]) if hasattr(model, 'coef_') else np.random.rand(X.shape[1])
        fig = px.bar(x=X.columns, y=importance, 
                    title="Feature Importance",
                    color=importance, color_continuous_scale=color_scale)
        
    else:
        fig = px.scatter(df, x='age', y='chol', color='target',
                        title="Data Visualization",
                        color_discrete_sequence=getattr(px.colors.sequential, color_scale))

    # Create analysis section based on selected model
    analysis_section = html.Div([
        html.H2(f"{selected_model.replace('_', ' ').title()} Analysis", 
               style={'color': 'white'}),
        html.P(f"Accuracy: {acc:.2%}", 
              style={'fontWeight': 'bold', 'fontSize': '18px', 'color': 'white'}),
        check_accuracy_threshold(acc),
        html.Pre(report, 
                style={'backgroundColor': 'rgba(0,0,0,0.5)', 
                      'padding': '10px',
                      'overflowX': 'auto'})
    ]) if selected_model != 'kmeans' else html.Div([
        html.H2(f"{selected_model.replace('_', ' ').title()} Analysis", 
               style={'color': 'white'}),
        html.P(f"Adjusted Rand Index: {acc:.2f}", 
              style={'fontWeight': 'bold', 'fontSize': '18px', 'color': 'white'}),
        html.P("Note: K-Means is an unsupervised clustering algorithm", 
              style={'color': 'white', 'fontStyle': 'italic'})
    ])

    return analysis_section, create_graph(fig)

def open_browser():
    webbrowser.open("http://localhost:8233")  # Open the web browser to the app URL

if __name__ == '__main__':
    threading.Timer(1.5, open_browser).start()  # Delay to allow the server to start
    app.run(port=8233, debug=False)  # Run the Dash app

