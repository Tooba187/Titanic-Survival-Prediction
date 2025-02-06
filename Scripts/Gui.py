# gui.py
import tkinter as tk
from tkinter import messagebox
import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load(r'C:\Users\Administrator\Desktop\internship\codeAlpha\task 1\Scripts\titanic_model.pkl')
scaler = joblib.load(r'C:\Users\Administrator\Desktop\internship\codeAlpha\task 1\Scripts\titanic_scaler.pkl')

def predict_survival():
    try:
        # Get input values from the GUI
        pclass = int(entry_pclass.get())
        sex = int(entry_sex.get())
        age = float(entry_age.get())
        fare = float(entry_fare.get())
        embarked = int(entry_embarked.get())
        family_size = int(entry_family_size.get())
        is_alone = int(entry_is_alone.get())

        # Create a DataFrame from inputs
        input_data = pd.DataFrame({
            'Pclass': [pclass],
            'Sex': [sex],
            'Age': [age],
            'Fare': [fare],
            'Embarked': [embarked],
            'FamilySize': [family_size],
            'IsAlone': [is_alone]
        })

        # Standardize the input data
        input_data_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_data_scaled)

        # Show the result
        if prediction[0] == 1:
            messagebox.showinfo("Prediction", "The person is likely to survive.")
        else:
            messagebox.showinfo("Prediction", "The person is unlikely to survive.")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Create the main window
root = tk.Tk()
root.title("Titanic Survival Prediction")

# Create input fields
tk.Label(root, text="Pclass (1, 2, 3):").grid(row=0, column=0)
entry_pclass = tk.Entry(root)
entry_pclass.grid(row=0, column=1)

tk.Label(root, text="Sex (0 for female, 1 for male):").grid(row=1, column=0)
entry_sex = tk.Entry(root)
entry_sex.grid(row=1, column=1)

tk.Label(root, text="Age:").grid(row=2, column=0)
entry_age = tk.Entry(root)
entry_age.grid(row=2, column=1)

tk.Label(root, text="Fare:").grid(row=3, column=0)
entry_fare = tk.Entry(root)
entry_fare.grid(row=3, column=1)

tk.Label(root, text="Embarked (0 for C, 1 for Q, 2 for S):").grid(row=4, column=0)
entry_embarked = tk.Entry(root)
entry_embarked.grid(row=4, column=1)

tk.Label(root, text="Family Size:").grid(row=5, column=0)
entry_family_size = tk.Entry(root)
entry_family_size.grid(row=5, column=1)

tk.Label(root, text="Is Alone (0 for no, 1 for yes):").grid(row=6, column=0)
entry_is_alone = tk.Entry(root)
entry_is_alone.grid(row=6, column=1)

# Create predict button
predict_button = tk.Button(root, text="Predict Survival", command=predict_survival)
predict_button.grid(row=7, column=0, columnspan=2)

# Run the application
root.mainloop()