import streamlit as st
st.set_page_config(
    page_title="Prediction of leaching rates of BSRFs fertilizers in soil",
    layout="wide"
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import shap

@st.cache_resource
def load_and_train():
    data = pd.read_csv('dataSRF.csv', header=0)
    X = data.drop('LR', axis=1)
    y = data['LR']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numerical_features = ['T', 't', 'ph-M', 'E-M', 'BET', 'ph-S', 'CEC', 'TOC', 'E-S', 'V']
    categorical_features = ['RMC', 'FPM', 'E']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ], verbose_feature_names_out=False)

    best_params = {
        'n_estimators': 437,
        'learning_rate': 0.22648248189516848,
        'num_leaves': 188,
        'max_depth': 10,
        'min_child_samples': 16,
        'subsample': 0.5779972601681014,
        'colsample_bytree': 0.5290418060840998,
        'reg_alpha': 0.4589458612326466,
        'reg_lambda': 0.001026006512489678,
        'random_state': 42
    }
    
    final_model = Pipeline([
        ('preprocessor', preprocessor),
        ('model', lgb.LGBMRegressor(**best_params))
    ])
    final_model.fit(X_train, y_train)

    y_pred = final_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    preprocessor = final_model.named_steps['preprocessor']
    X_train_preprocessed = pd.DataFrame(
        preprocessor.transform(X_train),
        columns=preprocessor.get_feature_names_out()
    )
    explainer = shap.TreeExplainer(final_model.named_steps['model'], X_train_preprocessed)
    X_test_preprocessed = pd.DataFrame(
        preprocessor.transform(X_test),
        columns=preprocessor.get_feature_names_out()
    )
    shap_values = explainer(X_test_preprocessed)

    return {
        'model': final_model,
        'X_train': X_train,
        'X_test': X_test,
        'rmse': rmse,
        'r2': r2,
        'explainer': explainer,
        'shap_values': shap_values,
        'feature_names': preprocessor.get_feature_names_out(),
        'categories': {
            'RMC': X_train['RMC'].unique(),
            'FPM': X_train['FPM'].unique(),
            'E': X_train['E'].unique()
        }
    }

data = load_and_train()

with st.sidebar:
    st.header("Input Features")
    inputs = {}
    
    num_features = [
        ('T', '°C'), ('t', 'h'), ('ph-M', ''), ('E-M', 'g/kg'),
        ('BET', 'm²/g'), ('ph-S', ''), ('CEC', 'cmol/kg'),
        ('TOC', 'g/kg'), ('E-S', 'g/kg'), ('V', 'mL')
    ]
    
    for col, unit in num_features:
        min_val = float(data['X_train'][col].min())
        max_val = float(data['X_train'][col].max())
        default_val = float(data['X_train'][col].median())
        label_unit = f" ({unit})" if unit else ""
        label = f"{col}{label_unit} [Range: {min_val:.1f}-{max_val:.1f}]"
        
        inputs[col] = st.number_input(
            label=label,
            min_value=min_val,
            max_value=max_val,
            value=default_val,
            step=0.1,
            format="%.1f"
        )
    
    inputs['RMC'] = st.selectbox('RMC', options=data['categories']['RMC'])
    inputs['FPM'] = st.selectbox('FPM', options=data['categories']['FPM'])
    inputs['E'] = st.selectbox('E', options=data['categories']['E'])

if st.sidebar.button('Predict'):
    input_df = pd.DataFrame([inputs])
    prediction = data['model'].predict(input_df)[0]
    st.success(f"Predicted Release Rate (LR): {prediction:.2f}%")

st.header("Model Performance")
col1, col2 = st.columns(2)
with col1:
    st.metric("Root Mean Squared Error (RMSE)", 
             f"{data['rmse']:.4f}",
             help="Measure the difference between the predicted value of the model and the actual value")
with col2:
    st.metric("R-squared (R²)", 
             f"{data['r2']:.4f}",
             help="Explanatory ratio of explanatory variable to target variable")

st.header("Model Interpretation")
st.markdown("""
### SHAP (SHapley Additive exPlanations)
""")

st.subheader("Global Feature Importance")
st.markdown("""
Show the contribution of features to the prediction results.
""")
fig1, ax1 = plt.subplots(figsize=(10, 6))
shap.plots.bar(data['shap_values'], max_display=15, show=False)
st.pyplot(fig1)

st.subheader("Individual Prediction Explanation")
st.markdown("""
Decision diagrams show how each feature pushes the predicted value from the baseline to the final result.
""")
sample_id = st.selectbox("Select sample to explain", 
                        options=range(len(data['X_test'])),
                        format_func=lambda x: f"Sample {x+1}")
fig2, ax2 = plt.subplots(figsize=(10, 6))
shap.decision_plot(
    data['explainer'].expected_value,
    data['shap_values'].values[sample_id],
    features=data['shap_values'].data[sample_id],
    feature_names=data['feature_names'],
    show=False
)
st.pyplot(fig2)