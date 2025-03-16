import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
import io
import base64
from sklearn.preprocessing import PolynomialFeatures
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Linear Regression: Econometrics vs ML", layout="wide")

# App title and description
st.title("Linear Regression: Econometrics vs Machine Learning")
st.markdown("""
This application demonstrates linear regression from two perspectives:
- **Econometrics**: Focus on model interpretation, statistical inference, and assumptions testing
- **Machine Learning**: Focus on prediction accuracy, train-test splits, and learning performance

Upload your own data or use the provided example dataset to explore both approaches.
""")

# Sidebar for navigation and options
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose approach:", ["Econometrics", "Machine Learning"])


# Function to load example data
@st.cache_data
def load_example_data():
    # Creating a synthetic dataset that resembles economic data
    np.random.seed(42)
    n = 200

    # Independent variables
    income = 50000 + 15000 * np.random.randn(n)
    education = 12 + 4 * np.random.randn(n)
    education = np.round(np.clip(education, 0, 25))
    experience = 10 + 7 * np.random.randn(n)
    experience = np.round(np.clip(experience, 0, 45))

    # Adding some non-linearity and interaction
    salary = 20000 + 0.5 * income + 2000 * education + 1500 * experience + 0.0001 * income * education + 5000 * np.random.randn(
        n)

    # Create a DataFrame
    df = pd.DataFrame({
        'Income': income,
        'Education_Years': education,
        'Work_Experience': experience,
        'Salary': salary
    })

    return df


# Data upload section
st.sidebar.header("Data Options")
data_option = st.sidebar.radio("Select data source:", ["Upload your own data", "Use example data"])

if data_option == "Upload your own data":
    uploaded_file = st.sidebar.file_uploader("Upload XLSX file", type="xlsx")
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error loading file: {e}")
            df = None
    else:
        df = None
else:
    df = load_example_data()

# Display the data if it's loaded
if df is not None:
    with st.expander("Dataset Preview"):
        st.dataframe(df.head())
        st.write(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

        # Data summary
        st.subheader("Descriptive Statistics")
        st.dataframe(df.describe())

        # Display columns data types
        st.subheader("Data Types")
        dtypes_df = pd.DataFrame(df.dtypes, columns=["Data Type"])
        st.dataframe(dtypes_df)

    # Variable selection
    st.sidebar.header("Variable Selection")

    if page == "Econometrics":
        # For econometrics: one dependent and multiple independent variables
        dependent_var = st.sidebar.selectbox("Select Dependent Variable (Y):", df.columns)
        independent_vars = st.sidebar.multiselect("Select Independent Variables (X):",
                                                  [col for col in df.columns if col != dependent_var])

        if dependent_var and independent_vars:
            st.header("Econometric Analysis")

            # Prepare data
            X = df[independent_vars]
            y = df[dependent_var]

            # Add constant for statsmodels
            X_with_const = sm.add_constant(X)

            # Fit the model
            model = sm.OLS(y, X_with_const).fit()

            # Display results
            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("Model Summary")
                st.text(model.summary().as_text())

                # Download link for regression results
                summary_html = model.summary().as_html()
                b64 = base64.b64encode(summary_html.encode()).decode()
                href = f'<a href="data:text/html;base64,{b64}" download="regression_results.html">Download regression results as HTML</a>'
                st.markdown(href, unsafe_allow_html=True)

            with col2:
                st.subheader("Interpretation")
                st.markdown("""
                **Key Statistics:**
                - **R-squared**: Proportion of variance explained by the model
                - **Adjusted R-squared**: R-squared adjusted for the number of predictors
                - **F-statistic**: Tests if at least one coefficient is non-zero
                - **Prob (F-statistic)**: p-value for the F-statistic, should be < 0.05
                - **Coefficient**: Effect of one unit change in X on Y
                - **P>|t|**: p-value for each coefficient, should be < 0.05
                - **Confidence Interval**: Range of plausible values for each coefficient
                """)

            # Visualization
            st.subheader("Visualizations")
            viz_col1, viz_col2 = st.columns(2)

            # Residuals plot
            with viz_col1:
                st.write("Residuals vs Fitted Values")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(model.fittedvalues, model.resid)
                ax.axhline(y=0, color='r', linestyle='-')
                ax.set_xlabel("Fitted Values")
                ax.set_ylabel("Residuals")
                ax.set_title("Residuals vs Fitted")
                st.pyplot(fig)

            # QQ plot
            with viz_col2:
                st.write("QQ Plot (Residual Normality)")
                fig, ax = plt.subplots(figsize=(10, 6))
                sm.qqplot(model.resid, ax=ax, line='45')
                ax.set_title("Normal Q-Q Plot")
                st.pyplot(fig)

            # Diagnostic tests
            st.subheader("Diagnostic Tests")

            # Heteroskedasticity test (Breusch-Pagan)
            bp_test = het_breuschpagan(model.resid, model.model.exog)
            bp_labels = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value']
            bp_df = pd.DataFrame({'Test': bp_labels, 'Value': bp_test})

            # Autocorrelation test (Durbin-Watson)
            dw_stat = durbin_watson(model.resid)

            test_col1, test_col2 = st.columns(2)

            with test_col1:
                st.write("Breusch-Pagan Test (Heteroskedasticity)")
                st.dataframe(bp_df)
                if bp_test[1] < 0.05:
                    st.warning("⚠️ Heteroskedasticity detected (p < 0.05). The variance of residuals is not constant.")
                else:
                    st.success("✅ No significant heteroskedasticity detected.")

            with test_col2:
                st.write("Durbin-Watson Test (Autocorrelation)")
                st.metric("Durbin-Watson Statistic", f"{dw_stat:.4f}")
                if dw_stat < 1.5:
                    st.warning("⚠️ Positive autocorrelation likely (DW < 1.5)")
                elif dw_stat > 2.5:
                    st.warning("⚠️ Negative autocorrelation likely (DW > 2.5)")
                else:
                    st.success("✅ No significant autocorrelation detected (DW ≈ 2)")

            # Feature importance in econometrics
            st.subheader("Feature Importance")

            # Calculate standardized coefficients
            X_std = (X - X.mean()) / X.std()
            X_std = sm.add_constant(X_std)
            model_std = sm.OLS(y, X_std).fit()

            # Extract coefficients (excluding constant)
            std_coefs = model_std.params[1:].abs()
            std_coefs = std_coefs.sort_values(ascending=False)

            fig, ax = plt.subplots(figsize=(10, 6))
            std_coefs.plot(kind='bar', ax=ax)
            ax.set_title('Standardized Coefficients (Absolute Value)')
            ax.set_ylabel('Standardized Coefficient')
            ax.set_xlabel('Variable')
            st.pyplot(fig)

            st.markdown("""
            **Standardized coefficients** show the relative importance of each predictor.
            They represent the change in Y (in standard deviations) for a one standard deviation change in X.
            """)

    elif page == "Machine Learning":
        # For ML: one target and multiple features
        target_var = st.sidebar.selectbox("Select Target Variable (y):", df.columns)
        feature_vars = st.sidebar.multiselect("Select Feature Variables (X):",
                                              [col for col in df.columns if col != target_var])

        # ML Options
        st.sidebar.header("ML Options")
        test_size = st.sidebar.slider("Test Set Size (%)", 10, 50, 20) / 100
        random_state = st.sidebar.number_input("Random State", value=42)

        # Models to demonstrate
        model_complexity = st.sidebar.selectbox(
            "Model Complexity Example:",
            ["Basic Linear Regression", "Polynomial Regression (Underfitting vs Overfitting)"]
        )

        if target_var and feature_vars:
            st.header("Machine Learning Analysis")

            # Prepare data
            X = df[feature_vars].values
            y = df[target_var].values

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

            st.write(f"Training set size: {X_train.shape[0]} samples")
            st.write(f"Test set size: {X_test.shape[0]} samples")

            # Training the model
            if model_complexity == "Basic Linear Regression":
                # Simple linear regression
                model = LinearRegression()
                model.fit(X_train, y_train)

                # Make predictions
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                # Calculate metrics
                train_mse = mean_squared_error(y_train, y_train_pred)
                test_mse = mean_squared_error(y_test, y_test_pred)
                train_r2 = r2_score(y_train, y_train_pred)
                test_r2 = r2_score(y_test, y_test_pred)

                # Display metrics
                metrics_col1, metrics_col2 = st.columns(2)

                with metrics_col1:
                    st.subheader("Training Metrics")
                    st.metric("Mean Squared Error (MSE)", f"{train_mse:.2f}")
                    st.metric("Root Mean Squared Error (RMSE)", f"{np.sqrt(train_mse):.2f}")
                    st.metric("Mean Absolute Error (MAE)", f"{mean_absolute_error(y_train, y_train_pred):.2f}")
                    st.metric("R² Score", f"{train_r2:.4f}")

                with metrics_col2:
                    st.subheader("Test Metrics")
                    st.metric("Mean Squared Error (MSE)", f"{test_mse:.2f}")
                    st.metric("Root Mean Squared Error (RMSE)", f"{np.sqrt(test_mse):.2f}")
                    st.metric("Mean Absolute Error (MAE)", f"{mean_absolute_error(y_test, y_test_pred):.2f}")
                    st.metric("R² Score", f"{test_r2:.4f}")

                # Model coefficients
                st.subheader("Model Coefficients")
                coef_df = pd.DataFrame({
                    'Feature': feature_vars,
                    'Coefficient': model.coef_
                })
                st.dataframe(coef_df)
                st.write(f"Intercept: {model.intercept_:.4f}")

                # Learning curve
                st.subheader("Learning Curve")

                # Generate learning curve data
                train_sizes, train_scores, test_scores = learning_curve(
                    LinearRegression(), X, y, cv=5,
                    train_sizes=np.linspace(0.1, 1.0, 10),
                    scoring="neg_mean_squared_error"
                )

                # Calculate mean and std for training and test sets
                train_scores_mean = -train_scores.mean(axis=1)
                train_scores_std = train_scores.std(axis=1)
                test_scores_mean = -test_scores.mean(axis=1)
                test_scores_std = test_scores.std(axis=1)

                # Plot learning curve
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.grid()
                ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                                train_scores_mean + train_scores_std, alpha=0.1, color="r")
                ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                                test_scores_mean + test_scores_std, alpha=0.1, color="g")
                ax.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training Error")
                ax.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation Error")
                ax.set_xlabel("Training examples")
                ax.set_ylabel("Mean Squared Error")
                ax.set_title("Learning Curve for Linear Regression")
                ax.legend(loc="best")
                st.pyplot(fig)

                st.markdown("""
                **Learning Curve Interpretation:**
                - **Training Error (red)**: Error on the training set
                - **Cross-validation Error (green)**: Error on the validation set

                When both curves:
                - Are close together with low error: Good model fit
                - Have a large gap: May indicate overfitting
                - Are close but with high error: May indicate underfitting
                """)

                # Predicted vs Actual values
                st.subheader("Predicted vs Actual Values")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(y_test, y_test_pred, alpha=0.5)

                # Plot ideal prediction line
                min_val = min(min(y_test), min(y_test_pred))
                max_val = max(max(y_test), max(y_test_pred))
                ax.plot([min_val, max_val], [min_val, max_val], 'r--')

                ax.set_xlabel("Actual Values")
                ax.set_ylabel("Predicted Values")
                ax.set_title("Predicted vs Actual Values (Test Set)")
                st.pyplot(fig)

                # Feature importance in ML context
                st.subheader("Feature Importance")

                # Standardize features to get comparable importance
                from sklearn.preprocessing import StandardScaler

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # Train model on standardized features
                model_scaled = LinearRegression()
                model_scaled.fit(X_scaled, y)

                # Get absolute coefficients as importance
                importance = np.abs(model_scaled.coef_)

                # Create feature importance DataFrame
                importance_df = pd.DataFrame({
                    'Feature': feature_vars,
                    'Importance': importance
                }).sort_values('Importance', ascending=False)

                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
                ax.set_title('Feature Importance')
                st.pyplot(fig)

            else:  # Polynomial Regression for underfitting/overfitting
                st.subheader("Polynomial Regression: Underfitting vs Overfitting")

                # Choose one feature for visualization
                if len(feature_vars) > 1:
                    st.info("Using only the first selected feature for polynomial visualization")

                X_single = df[feature_vars[0]].values.reshape(-1, 1)
                y_single = df[target_var].values

                # Train-test split
                X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
                    X_single, y_single, test_size=test_size, random_state=random_state
                )

                # Create and plot models with different degrees
                degrees = [1, 3, 10]  # Underfitting, good fit, overfitting

                fig, axes = plt.subplots(1, 3, figsize=(18, 6))

                train_errors = []
                test_errors = []

                for i, degree in enumerate(degrees):
                    # Create polynomial features
                    poly = PolynomialFeatures(degree=degree)
                    X_train_poly = poly.fit_transform(X_train_s)
                    X_test_poly = poly.transform(X_test_s)

                    # Fit model
                    model = LinearRegression()
                    model.fit(X_train_poly, y_train_s)

                    # Make predictions
                    y_train_pred = model.predict(X_train_poly)
                    y_test_pred = model.predict(X_test_poly)

                    # Calculate errors
                    train_mse = mean_squared_error(y_train_s, y_train_pred)
                    test_mse = mean_squared_error(y_test_s, y_test_pred)

                    train_errors.append(train_mse)
                    test_errors.append(test_mse)

                    # Create a mesh for smooth curve plotting
                    X_mesh = np.linspace(
                        min(X_train_s.min(), X_test_s.min()),
                        max(X_train_s.max(), X_test_s.max()),
                        100
                    ).reshape(-1, 1)
                    X_mesh_poly = poly.transform(X_mesh)
                    y_mesh_pred = model.predict(X_mesh_poly)

                    # Plot
                    axes[i].scatter(X_train_s, y_train_s, color='blue', alpha=0.5, label='Training data')
                    axes[i].scatter(X_test_s, y_test_s, color='green', alpha=0.5, label='Test data')
                    axes[i].plot(X_mesh, y_mesh_pred, color='red', linewidth=2, label='Model')

                    if i == 0:
                        title = f"Degree {degree} (Underfitting)"
                    elif i == 1:
                        title = f"Degree {degree} (Good Fit)"
                    else:
                        title = f"Degree {degree} (Overfitting)"

                    axes[i].set_title(title)
                    axes[i].set_xlabel(feature_vars[0])
                    axes[i].set_ylabel(target_var)
                    axes[i].legend()

                plt.tight_layout()
                st.pyplot(fig)

                # Plot training vs test error
                st.subheader("Training vs Test Error")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(degrees, train_errors, 'o-', label='Training Error')
                ax.plot(degrees, test_errors, 'o-', label='Test Error')
                ax.set_xlabel('Polynomial Degree')
                ax.set_ylabel('Mean Squared Error')
                ax.set_title('Error vs Polynomial Degree')
                ax.legend()
                st.pyplot(fig)

                st.markdown("""
                **Polynomial Regression Interpretation:**

                1. **Underfitting (Degree 1):**
                   - Model is too simple to capture the underlying pattern
                   - High training and test error
                   - High bias, low variance

                2. **Good Fit (Degree 3):**
                   - Model captures the underlying pattern well
                   - Low training and test error
                   - Good balance between bias and variance

                3. **Overfitting (Degree 10):**
                   - Model is too complex and captures noise in the data
                   - Very low training error but high test error
                   - Low bias, high variance
                   - Poor generalization to new data
                """)

                # Learning curves for different polynomial degrees
                st.subheader("Learning Curves for Different Polynomial Degrees")

                fig, axes = plt.subplots(1, 3, figsize=(18, 6))

                for i, degree in enumerate(degrees):
                    poly = PolynomialFeatures(degree=degree)
                    X_poly = poly.fit_transform(X_single)

                    model = LinearRegression()

                    train_sizes, train_scores, test_scores = learning_curve(
                        model, X_poly, y_single, cv=5,
                        train_sizes=np.linspace(0.1, 1.0, 10),
                        scoring="neg_mean_squared_error"
                    )

                    train_scores_mean = -train_scores.mean(axis=1)
                    train_scores_std = train_scores.std(axis=1)
                    test_scores_mean = -test_scores.mean(axis=1)
                    test_scores_std = test_scores.std(axis=1)

                    axes[i].grid()
                    axes[i].fill_between(train_sizes, train_scores_mean - train_scores_std,
                                         train_scores_mean + train_scores_std, alpha=0.1, color="r")
                    axes[i].fill_between(train_sizes, test_scores_mean - test_scores_std,
                                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
                    axes[i].plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training Error")
                    axes[i].plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation Error")

                    if i == 0:
                        title = f"Degree {degree} (Underfitting)"
                    elif i == 1:
                        title = f"Degree {degree} (Good Fit)"
                    else:
                        title = f"Degree {degree} (Overfitting)"

                    axes[i].set_title(title)
                    axes[i].set_xlabel("Training examples")
                    axes[i].set_ylabel("Mean Squared Error")
                    axes[i].legend(loc="best")

                plt.tight_layout()
                st.pyplot(fig)

                st.markdown("""
                **Learning Curves Interpretation:**

                1. **Underfitting (Degree 1):**
                   - Both curves converge to a high error
                   - Adding more training data won't significantly improve performance
                   - Need to increase model complexity

                2. **Good Fit (Degree 3):**
                   - Gap between training and validation error is small
                   - Both errors converge to a low value
                   - Adding more data might slightly improve performance

                3. **Overfitting (Degree 10):**
                   - Large gap between training and validation error
                   - Training error is much lower than validation error
                   - Need regularization or to reduce model complexity
                   - Adding more data might help reduce overfitting
                """)

    # If no variables are selected
    else:
        st.info("Please select variables in the sidebar to start the analysis.")
else:
    st.info("Please upload a dataset or use the example data to get started.")

# Comparison section
st.sidebar.header("Comparison")
with st.sidebar.expander("Econometrics vs ML Approach"):
    st.markdown("""
    **Econometrics:**
    - Focus on model interpretation and inference
    - Tests statistical assumptions (normality, heteroskedasticity)
    - Emphasis on parameter estimation and confidence intervals
    - Concerned with causal relationships

    **Machine Learning:**
    - Focus on prediction accuracy
    - Emphasis on model performance metrics
    - Cross-validation and train-test splits
    - Concern with overfitting/underfitting
    - Learning curves and model complexity
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("© 2025 Linear Regression App")