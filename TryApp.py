import sqlite3
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_curve, auc)
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import streamlit as st

# Initialize database and load CSV
def initialize_database():
    con = sqlite3.connect('HR.db')
    cur = con.cursor()

    # Create table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS EmployeesRecords (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            Age INTEGER,
            Accessibility INTEGER,
            EducationLevel TEXT,
            Gender TEXT,
            WorkedBefore INTEGER,
            MentalHealth TEXT,
            MainBranch TEXT,
            YearsOfCoding INTEGER,
            YearsOfCodingWhileWorking INTEGER,
            Country TEXT,
            PreviousSalary REAL,
            HaveWorkedWith TEXT,
            ComputerSkills INTEGER,
            Employed INTEGER
        );
    """)

    # Import CSV
    with open("stackoverflow_full .csv", 'r') as file:
        reader = csv.reader(file)
        headers = next(reader)  # Skip the header row
        rows_to_insert = []
        for count, row in enumerate(reader):
            if count >= 50000:  # Limit to 50,000 rows
                break
            rows_to_insert.append(row[1:])  # Skip the first column (assuming it's an ID column)
        cur.executemany(f"""
            INSERT INTO EmployeesRecords (
                Age, Accessibility, EducationLevel, Gender, WorkedBefore, MentalHealth,
                MainBranch, YearsOfCoding, YearsOfCodingWhileWorking, Country, PreviousSalary,
                HaveWorkedWith, ComputerSkills, Employed
            )
            VALUES ({','.join(['?'] * (len(headers) - 1))});
        """, rows_to_insert)

    con.commit()
    con.close()

# Retrieve Data from DB to DataFrame for EDA
def load_data_to_df():
    con = sqlite3.connect('HR.db')
    df = pd.read_sql_query("SELECT * FROM EmployeesRecords", con)
    con.close()
    return df

# EDA & Data Preprocessing
def perform_eda_and_preprocessing(df):
    st.subheader("EDA & Data Preprocessing")

    # Checking for missing values
    st.write("Missing Values:")
    st.write(df.isnull().sum())

    # Dropping duplicate rows
    df = df.drop_duplicates()

    # Encoding categorical variables
    label_encoder = LabelEncoder()
    categorical_columns = ['Age', 'Accessibility', 'EducationLevel', 'Gender', 'MentalHealth', 'MainBranch', 'Country']

    for col in categorical_columns:
        df[col] = label_encoder.fit_transform(df[col].astype(str))

    # Handle outliers (IQR method)
    def remove_outliers_iqr(data, column_name):
        Q1 = data[column_name].quantile(0.25)
        Q3 = data[column_name].quantile(0.75)
        IQR = Q3 - Q1
        return data[(data[column_name] >= Q1 - 1.5 * IQR) & (data[column_name] <= Q3 + 1.5 * IQR)]

    # Remove outliers
    for col in ['YearsOfCoding', 'PreviousSalary', 'ComputerSkills']:
        df = remove_outliers_iqr(df, col)

    # Feature and target selection
    X = df[['Age', 'Accessibility', 'ComputerSkills', 'PreviousSalary']]
    y = df['Employed']

    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.success("EDA and Preprocessing complete.")
    return X_train, X_test, y_train, y_test

# Train the selected model and print results
def train_model(model_name, X_train, X_test, y_train, y_test):
    st.subheader(f"Training {model_name}")

    if model_name == "Logistic Regression":
        model = LogisticRegression(random_state=0)
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
    elif model_name == "Random Forest":
        model = RandomForestClassifier(random_state=42)
    else:
        st.error("Invalid model name.")
        return None

    model.fit(X_train, y_train)
    y_pred = model .predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    st.write(f'**{model_name} Results:**')
    st.write(f'Accuracy: {accuracy:.4f}')
    st.write('Classification Report:')
    st.text(class_report)

    # Save the model
    joblib.dump(model, f"{model_name.replace(' ', '_')}.joblib")

    # Plot the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Not Employed', 'Employed'], yticklabels=['Not Employed', 'Employed'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{model_name} - Confusion Matrix')
    st.pyplot(plt)

    # Calculate the AUC
    y_scores = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    auc_score = auc(fpr, tpr)
    st.write(f'{model_name} - AUC: {auc_score:.4f}')

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} - ROC Curve')
    plt.legend(loc='lower right')
    st.pyplot(plt)

    return model

# Train the models based on user selection
def train_models_and_get_models(X_train, X_test, y_train, y_test):
    st.subheader("Select Models to Train")
    available_models = ["Logistic Regression", "Decision Tree", "Random Forest"]
    models_to_train = st.multiselect("Select models to train:", available_models)

    trained_models = {}
    for model_name in models_to_train:
        trained_models[model_name] = train_model(model_name, X_train, X_test, y_train, y_test)

    return trained_models

# AI-Powered HR Mode
def ai_powered_hr_mode():
    st.subheader("AI-Powered HR Mode")
    df = load_data_to_df()
    X_train, X_test, y_train, y_test = perform_eda_and_preprocessing(df)
    models = train_models_and_get_models(X_train, X_test, y_train, y_test)

    if st.button("Back to Home"):
        st.session_state.page = "home"

    if models:
        model_name = st.selectbox("Select a model for predictions", list(models.keys()))
        model = models[model_name]
        st.write(f"Using {model_name} for predictions")

        # Get input for prediction
        age = st.number_input("Age", min_value=0)
        accessibility = st.selectbox("Accessibility (1 or 0)", [0, 1])
        computer_skills = st.number_input("Number of Computer Skills", min_value=0)
        previous_salary = st.number_input("Previous Salary", min_value=0.0)

        # Predict
        if st.button("Predict"):
            prediction = model.predict([[age, accessibility, computer_skills, previous_salary]])[0]
            st.write(f"Prediction Result: {'Accepted' if prediction == 1 else 'Rejected'}")

# Retrieve number of head rows
def retrieve_head_rows():
    con = sqlite3.connect('HR.db')
    df = pd.read_sql_query("SELECT * FROM EmployeesRecords", con)
    con.close()

    num_rows = st.number_input("Enter the number of rows you want to retrieve:", min_value=1, max_value=len(df))
    if st.button("Retrieve"):
        st.write(f"\nFirst {num_rows} rows from the database:")
        st.write(df.head(num_rows))

# Add Employee function
def add_employee():
    con = sqlite3.connect('HR.db')
    cur = con.cursor()

    # Input from user
    age = st.number_input("Enter Age:", min_value=0)
    accessibility = st.selectbox("Enter Accessibility (1 or 0):", [0, 1])
    education_level = st.text_input("Enter Education Level:")
    gender = st.text_input("Enter Gender:")
    worked_before = st.selectbox("Worked Before (1 or 0):", [0, 1])
    mental_health = st.text_input("Enter Mental Health (Yes or No):")
    main_branch = st.text_input("Enter Main Branch (Dev or NotDev):")
    years_of_coding = st.number_input("Enter Years of Coding:", min_value=0)
    years_of_coding_while_working = st.number_input("Enter Years of Coding While Working:", min_value=0)
    country = st.text_input("Enter Country:")
    previous_salary = st.number_input("Enter Previous Salary:", min_value=0.0)
    have_worked_with = st.text_input("Enter Skills Have Worked With (comma separated):")
    computer_skills = st.number_input("Enter number of Computer Skills:", min_value=0)
    employed = st.selectbox("Employed (1 or 0):", [0, 1])

    # Insert the new employee into the database
    if st.button("Add Employee"):
        cur.execute("""
            INSERT INTO EmployeesRecords (
                Age, Accessibility, EducationLevel, Gender, WorkedBefore, MentalHealth,
                MainBranch, YearsOfCoding, YearsOfCodingWhileWorking, Country, PreviousSalary,
                HaveWorkedWith, ComputerSkills, Employed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (age, accessibility, education_level, gender, worked_before, mental_health, main_branch,
              years_of_coding, years_of_coding_while_working, country, previous_salary, have_worked_with,
              computer_skills, employed))

        con.commit()
        con.close()
        st.success("Employee added successfully!")

# Retrieve Employee function
def retrieve_employee():
    con = sqlite3.connect('HR.db')
    df = pd.read_sql_query("SELECT * FROM EmployeesRecords", con)
    con.close()

    emp_id = st.number_input("Enter Employee ID to retrieve:", min_value=1)
    if st.button("Retrieve Employee"):
        employee = df[df['ID'] == emp_id]
        if not employee.empty:
            st.write("Employee Details:")
            st.write(employee)
        else:
            st.error("Employee not found.")

# Delete Employee function
def delete_employee():
    con = sqlite3.connect('HR.db')
    cur = con.cursor()

    emp_id = st.number_input("Enter Employee ID to delete:", min_value=1)
    if st.button("Delete Employee"):
        cur.execute("DELETE FROM EmployeesRecords WHERE ID=?", (emp_id,))
        con.commit()
        con.close()
        st.success(f"Employee with ID {emp_id} deleted successfully!")

# Update Employee function
def update_employee():
    con = sqlite3.connect('HR.db')
    cur = con.cursor()

    emp_id = st.number_input("Enter Employee ID to update:", min_value=1)

    # Get new data for updating
    age = st.number_input("Enter new Age:", min_value=0)
    accessibility = st.selectbox("Enter new Accessibility (1 or 0):", [0, 1])
    education_level = st.text_input("Enter new Education Level:")
    gender = st.text_input("Enter new Gender:")
    worked_before = st.selectbox("Worked Before (1 or 0):", [0, 1])
    mental_health = st.text_input("Enter new Mental Health (Yes or No):")
    main_branch = st.text_input("Enter new Main Branch (Dev or NotDev):")
    years_of_coding = st.number_input("Enter new Years of Coding:", min_value=0)
    years_of_coding_while_working = st.number_input("Enter new Years of Coding While Working:", min_value=0)
    country = st.text_input("Enter new Country:")
    previous_salary = st.number_input("Enter new Previous Salary:", min_value=0.0)
    have_worked_with = st.text_input("Enter new Skills Have Worked With (comma separated):")
    computer_skills = st.number_input("Enter new number of Computer Skills:", min_value=0)
    employed = st.selectbox("Employed (1 or 0):", [0, 1])

    # Update the employee record in the database
    if st.button("Update Employee"):
        cur.execute("""
            UPDATE EmployeesRecords
            SET Age=?, Accessibility=?, EducationLevel=?, Gender=?, WorkedBefore=?, MentalHealth=?,
                MainBranch=?, YearsOfCoding=?, YearsOfCodingWhileWorking=?, Country=?, PreviousSalary=?,
                HaveWorkedWith=?, ComputerSkills=?, Employed=?
            WHERE ID=?
        """, (age, accessibility, education_level, gender, worked_before, mental_health, main_branch,
              years_of_coding, years_of_coding_while_working, country, previous_salary, have_worked_with,
              computer_skills, employed, emp_id))

        con.commit()
        con.close()
        st .success(f"Employee with ID {emp_id} updated successfully!")

# Manual HR Mode
def manual_hr_mode():
    st.subheader("Manual HR Mode")
    option = st.selectbox("Choose an action:", ["Add Employee", "Retrieve Employee", "Delete Employee", "Update Employee", "Retrieve Head Rows from DB", "Back to Home"])

    if option == "Add Employee":
        add_employee()
    elif option == "Retrieve Employee":
        retrieve_employee()
    elif option == "Delete Employee":
        delete_employee()
    elif option == "Update Employee":
        update_employee()
    elif option == "Retrieve Head Rows from DB":
        retrieve_head_rows()
    elif option == "Back to Home":
        st.session_state.page = "home"

# Main Menu
def main_menu():
    st.title("HR Management System")
    if "initialized" not in st.session_state:
        initialize_database()
        st.session_state.initialized = True

    if "page" not in st.session_state:
        st.session_state.page = "home"

    if st.session_state.page == "home":
        if st.session_state.initialized:
            st.success("Database initialized and CSV data imported successfully!")
            st.session_state.initialized = False  # Ensure it only shows once
        if st.button("Manual HR Mode"):
            st.session_state.page = "manual_hr_mode"
        if st.button("AI-Powered HR Mode"):
            st.session_state.page = "ai_powered_hr_mode"
        if st.button("Exit"):
            st.stop()

    elif st.session_state.page == "manual_hr_mode":
        manual_hr_mode()

    elif st.session_state.page == "ai_powered_hr_mode":
        ai_powered_hr_mode()

if __name__ == "__main__":
    main_menu()