import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import ( mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix)
import pickle

#title 

st.title('Bondora LoanApp')
st.subheader('Simple loan default prediction app')

data = pd.read_csv('Bondora_preprocessed.csv', low_memory=False)
data.head()

# Display basic data info
st.write("**Data Shape:**", data.shape)
st.write("**Data Columns:**", data.columns)
st.write("**Head:**", data.head())
st.write("**Description:**", data.describe())

# Step 3: Select Feature and Target Columns
st.subheader("Select Feature and Target Columns")
target_column = st.selectbox("Select the Target Column", data.columns)
feature_columns = st.multiselect(
        "Select Feature Columns", data.columns.drop(target_column)
    )

if target_column and feature_columns:
        X = data[feature_columns]
        y = data[target_column]

        # Step 4: Identify Problem Type (Regression or Classification)
if pd.api.types.is_numeric_dtype(data[target_column]):
            problem_type = "Regression"
            st.write("The problem is identified as **Regression**.")
else:
            problem_type = "Classification"
            st.write("The problem is identified as **Classification**.")

        # Step 6: Preprocess Data
st.subheader("Preprocessing the Data")

# Handle missing values using Iterative Imputer
imputer = IterativeImputer()
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

        # Scale numerical features
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        # Encode categorical features if present
encoders = {}
for col in X.select_dtypes(include=['object']).columns:
            encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
            X_encoded = encoder.fit_transform(X[[col]])
            X = X.drop(col, axis=1)
            X = pd.concat([X, pd.DataFrame(X_encoded)], axis=1)
            encoders[col] = encoder

        # Step 7: Model Selection
st.subheader("Model Selection")

if problem_type == "Regression":
            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Support Vector Machine": SVR()
            }
else:
            models = {
                "Logistic Regression": LogisticRegression(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Support Vector Machine": SVC()
            }

model_name = st.selectbox("Select Model", list(models.keys()))
model = models[model_name]

# Step 8: Train-Test Split
st.subheader("Train-Test Split")
test_size = st.slider("Select Test Size", 0.1, 0.5, 0.2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

 # Step 10: Train the Model
st.subheader("Training the Model")
model.fit(X_train, y_train)
st.write(f"**{model_name}** model trained successfully!")

# Step 11: Evaluate the Model
st.subheader("Evaluating the Model")
y_pred = model.predict(X_test)
if problem_type == "Regression":
            st.write("**Mean Squared Error:**", mean_squared_error(y_test, y_pred))
            st.write("**Mean Absolute Error:**", mean_absolute_error(y_test, y_pred))
            st.write("**R2 Score:**", r2_score(y_test, y_pred))
else:
            st.write("**Accuracy:**", accuracy_score(y_test, y_pred))
            st.write("**Precision:**", precision_score(y_test, y_pred, average='weighted'))
            st.write("**Recall:**", recall_score(y_test, y_pred, average='weighted'))
            st.write("**F1 Score:**", f1_score(y_test, y_pred, average='weighted'))

            # Confusion Matrix
            st.subheader("**Confusion Matrix:**")
            fig, ax = plt.subplots()
            cm = confusion_matrix(y_test, y_pred)
            ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center')
            st.pyplot(fig)

# Step 13: Save the Model
st.subheader("Save the Model")
if st.button("Download Model"):
            model_filename = f"{model_name}.pkl"
            with open(model_filename, "wb") as f:
                pickle.dump(model, f)
            st.download_button("Download Model", model_filename)

# # Step 15: Make Predictions on New Data
# st.subheader("Make Predictions")
# if st.button("Upload New Data for Prediction"):
#             new_data = st.file_uploader("Upload New Data", type=["csv", "xlsx"])
#             if new_data:
#                 if new_data.name.endswith(".csv"):
#                     new_data = pd.read_csv(new_data)
#                 else:
#                     new_data = pd.read_excel(new_data)

#                 # Preprocess new data similarly
#                 new_data = pd.DataFrame(imputer.transform(new_data), columns=new_data.columns)
#                 new_data = pd.DataFrame(scaler.transform(new_data), columns=new_data.columns)

#                 for col, encoder in encoders.items():
#                     encoded = encoder.transform(new_data[[col]])
#                     new_data = new_data.drop(col, axis=1)
#                     new_data = pd.concat([new_data, pd.DataFrame(encoded)], axis=1)

#                 predictions = model.predict(new_data)
#                 st.write("**Predictions:**", predictions)


st.header("Make Predictions on User Inputs")
# Collect user inputs for prediction
st.subheader("Provide Input Values for Prediction")
numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
user_inputs = {}
for col in X.columns:
    if col in numerical_cols:
        user_inputs[col] = st.number_input(f"Enter value for {col}", value=float(X[col].mean()))
    else:
        user_inputs[col] = st.selectbox(f"Select value for {col}", options=data[col].unique())

# Make prediction
user_df = pd.DataFrame([user_inputs])
for col, encoder in encoders.items():
    encoded = encoder.transform(user_df[[col]])
    user_df = user_df.drop(col, axis=1)
    user_df = pd.concat([user_df, pd.DataFrame(encoded)], axis=1)

prediction = model.predict(user_df)
st.write(f"Prediction: {prediction}")

# --- Visualizations ---
st.header('Data Visualizations')

# Correlation Heatmap for Limited Number of Numerical Columns
st.subheader("Correlation Heatmap for Numerical Columns")
selected_numerical_cols = st.multiselect("Select Numerical Columns for Correlation Heatmap", numerical_cols, default=numerical_cols[:6])
if selected_numerical_cols:
    corr = data[selected_numerical_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Heatmap for Selected Numerical Columns')
    st.pyplot(fig)
else:
    st.write("Please select at least one numerical column.")


# Pairplot for Limited Number of Numerical Columns
st.subheader("Pairplot of Numerical Columns")
selected_numerical_cols = st.multiselect("Select Numerical Columns for Pairplot", numerical_cols, default=numerical_cols[:4])
if selected_numerical_cols:
    fig = sns.pairplot(data, vars=selected_numerical_cols, kind='scatter', diag_kind='kde', palette='husl')
    st.pyplot(fig)
else:
    st.write("Please select at least one numerical column.")


categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
# which plots can be made for categorical columns
st.subheader("Plots for Categorical Columns")
if categorical_cols:
    plot_type = st.selectbox("Select Plot Type", ["Count Plot", "Pie Chart"])
    selected_col = st.selectbox("Select a Column", categorical_cols)
    if plot_type == "Count Plot":
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x=selected_col, data=data, palette='viridis', ax=ax)
        ax.set_title(f"Count Plot of {selected_col}")
        st.pyplot(fig)
    else:
        fig, ax = plt.subplots()
        data[selected_col].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, cmap='viridis', ax=ax)
        ax.set_title(f"Pie Chart of {selected_col}")
        st.pyplot(fig)
else:
    st.write("No categorical columns found in the dataset.")
    
# Which plots can be made numarical columns
st.subheader("Plots for Numerical Columns")
if numerical_cols:
    plot_type = st.selectbox("Select Plot Type", ["Histogram", "Box Plot", "Violin Plot", "Bar Plot", "Swarm Plot", "Strip Plot"])
    selected_col = st.selectbox("Select a Column", numerical_cols)
    if plot_type == "Histogram":
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data[selected_col], kde=True, color='skyblue', edgecolor='black', ax=ax)
        ax.set_title(f"Histogram of {selected_col}")
        st.pyplot(fig)
    else:
        fig, ax = plt.subplots()
        sns.boxplot(x=selected_col, data=data, palette='viridis', ax=ax)
        ax.set_title(f"Box Plot of {selected_col}")
        st.pyplot(fig)
else:
    st.write("No numerical columns found in the dataset.")
