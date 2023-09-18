import time
from tkinter import font, messagebox
import tkinter as tk
from tkinter.font import Font
from problog.engine import DefaultEngine
from problog.program import PrologString
from problog import get_evaluatable
from problog.core import ProbLog
from problog.logic import Term
from problog.learning import lfi
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sys




global thresholds
# i set the thresholds to determine if a parameter is small/short or the opposite
thresholds = {
        'radius_mean': [('small_radius', 12)],
        'texture_mean': [('smooth_texture', 10)],
        'perimeter_mean': [('short_perimeter', 50)],
        'area_mean': [('small_area', 500)],
        'smoothness_mean': [('smooth_smoothness', 0.05 )],
        'compactness_mean': [('high_compactness', 0.05)],
        'concavity_mean': [('high_concavity', 0.05)],
        'symmetry_mean': [('low_symmetry', 0.1)],
        'fractal_dimension_mean': [('low_fractal_dimension', 0.05)]
    }   



def read_data(variables):
    evidences = []



    for i in range(0,data.shape[0]):
        evidence = []
        patient = data.loc[i].values
        # print(patient)
        j = 0


        for column in data.columns.values:
            data_value = 0;
            column_index = data.columns.get_loc(column)
            if(column == 'diagnosis'):
                if patient[0] == 'B':
                    my_tuple = (variables[-1], False)
                    evidence.append(my_tuple)
                else:
                    my_tuple = (variables[-1], False)
                    evidence.append(my_tuple)                
            else:
                data_value = float(patient[column_index])
                for threshold in thresholds[column]:
                    label, value = threshold
                    # print(label)
                    # print(value)
                    if data_value <= value:
                        evidence.append((variables[j], True))
                    else:
                        evidence.append((variables[j], False))
                    break
            j += 1
            
        evidences.append(evidence)
    return evidences

def call_problog(my_data):

    i  = 0
    new_data = []
    for key, value in thresholds.items():
        # print(key)
        data_value = float(my_data[i])
        i += 1
        for threshold in thresholds[key]:
            label, value = threshold
            if data_value <= value:
                new_data.append(label)
                break
            else:
                new_data.append("\\+{}".format(label))
        
    # print(new_data)
    my_formatted_data = "t(_) :: malignant:-{}, {}, {}, {}, {}, {}, {}, {}, {}.". format(new_data[0], new_data[1], new_data[2],
                          new_data[3], new_data[4], new_data[5], new_data[6], new_data[7], new_data[8])


    # Create a Problog program
    problog_code = """
    malignant.
    small_radius.
    smooth_texture.
    short_perimeter.
    small_area.
    smooth_smoothness.
    high_compactness.
    high_concavity.
    high_symmetry.
    high_fractal_dimension.

    {}
    """.format(my_formatted_data)

    # print(problog_code)
    # print("\n\n\n\n\n\n\n")

    malignant = Term('malignant')   
    radius = Term('small_radius')
    texture = Term('smooth_texture')
    perimeter = Term('short_perimeter')
    area = Term('small_area')
    smoothness = Term('smooth_smoothness')
    compactness = Term('high_compactness')
    concavity = Term('high_concavity')
    symmetry = Term('high_symmetry')
    fractal_dimension = Term('high_fractal_dimension')
    

    variables = [radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, fractal_dimension, malignant]

    evidences = read_data(variables)
    # print(evidences)

    score, weights, atoms, iteration, lfi_problem = lfi.run_lfi(PrologString(problog_code), evidences)


    result = lfi_problem.get_model()
    print("\n\n\n\n")
    print (result)


    lines = result.splitlines()
    for i in range(len(lines)):
        if i == len(lines)-1:
            parts = lines[i].split("::")
            if len(parts) > 1:
                probability = parts[0].strip()
            break
    print(f"Probabilty of having malignant cancer: {probability}")
    return probability



def calculate_number():
    # Get the input values from the user
    
    inputs = [entry.get() for entry in entries]

    # Check if any input is empty
    if any(not input_value for input_value in inputs):
        messagebox.showerror("Empty Input", "Please fill in all the input fields.")
        return
    

    radius_mean = float(inputs[0])
    texture_mean = float(inputs[1])
    perimeter_mean = float(inputs[2])
    area_mean = float(inputs[3])
    smoothness_mean = float(inputs[4])
    compactness_mean = float(inputs[5])
    concavity_mean = float(inputs[6])
    symmetry_mean = float(inputs[7])
    fractal_dimension_mean = float(inputs[8])
    # Convert the input values to appropriate types if needed
    user_input = [radius_mean, texture_mean, perimeter_mean, 
                  area_mean, smoothness_mean, compactness_mean, 
                  concavity_mean, symmetry_mean, fractal_dimension_mean]

    # Perform the calculation based on the input values calling the problog function
    calculated_number = call_problog(user_input)
    calculated_label  = tk.Label(window, text=calculated_number, font=("Arial", 12, "bold"))
    calculated_label.grid(row=len(labels) + 2, column=0, columnspan=2, padx=5, pady=5)

    




def evaluate_model(data):
    # Split data into train and test sets
    features = data.drop('diagnosis', axis=1)
    labels = data['diagnosis']
    labels = labels.map({'B': 0, 'M': 1})
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Convert the train and test dataframes to numpy arrays
    train_data = train_features.values
    test_data = test_features.values

    # Train the model and calculate probabilities for test set
    probabilities = []
    for i in range(test_data.shape[0]):
        probability = call_problog(test_data[i])
        probabilities.append(probability)


    # Convert probabilities to binary predictions
    predictions = [1 if float(p) >= 0.5 else 0 for p in probabilities]

    # Calculate evaluation metrics
    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions)
    recall = recall_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions)
  

    return accuracy, precision, recall, f1



if __name__ == "__main__":
    file_path = r'C:\Users\user\Desktop\Cancer_Prediciton\cancer_data.csv'

    # Load the dataset
    data = pd.read_csv(file_path)
    #print(data)
    # Drop the 'id' column as it is not relevant for classification
    data = data.drop('id', axis=1)

    #drop the rows with missing values 
    data = data.dropna()




def restart():
    # Clear the entry fields
    for entry in entries:
        entry.delete(0, tk.END)


def close_application():
    sys.exit()


# Create the main window
window = tk.Tk()
window.title("Calculating Probabilty of Having Malignant Breast Cancer")

# Set the window size
window.geometry("550x600")

# Set the window position in the top-left corner
window.geometry("+0+0")

# Set the background color of the window to green
window.configure(bg="white")

window.protocol("WM_DELETE_WINDOW", close_application)


bold_font = font.Font(weight="bold")  # Create a bold font

# Create labels and entry fields for each parameter
labels = [
    "Radius",
    "Texture",
    "Perimeter",
    "Area",
    "Smoothness",
    "Compactness",
    "Concavity",
    "Symmetry",
    "Fractal Dimension"
]
entries = []





def open_details_window():
    # Create a new window for displaying the evaluation metrics
    details_window = tk.Toplevel(window)
    details_window.title("Evaluation Metrics")
    details_window.geometry("300x200")
    details_window.configure(bg="white")

    # Calculate the evaluation metrics
    accuracy, precision, recall, f1 = evaluate_model(data)

    # Create labels to display the evaluation metrics in the new window
    accuracy_label = tk.Label(details_window, text=f"Accuracy: {accuracy:.4f}", font=bold_font, bg="white")
    precision_label = tk.Label(details_window, text=f"Precision: {precision:.4f}", font=bold_font, bg="white")
    recall_label = tk.Label(details_window, text=f"Recall: {recall:.4f}", font=bold_font, bg="white")
    f1_label = tk.Label(details_window, text=f"F1 Score: {f1:.4f}", font=bold_font, bg="white")

    # Position the labels in the new window
    accuracy_label.pack(pady=5)
    precision_label.pack(pady=5)
    recall_label.pack(pady=5)
    f1_label.pack(pady=5)



# Create a button to see the details
details_button = tk.Button(window, text="See Details", command=open_details_window, bd=2, highlightthickness=0)
details_button.grid(row=len(labels)+20, column=2, columnspan=1, padx=10, pady=10)












for i, label_text in enumerate(labels):
    label = tk.Label(window, text=label_text + ":", font=bold_font, bg="white", highlightbackground="white")
    entry = tk.Entry(window, font=bold_font, highlightbackground="white", bd=2, relief="solid")
    label.grid(row=i, column=0, padx=5, pady=5)
    entry.grid(row=i, column=1, padx=5, pady=5)
    entries.append(entry)


# Create a restart button
restart_button = tk.Button(window, text="Restart", command=restart, bd=2, highlightthickness=0)
restart_button.grid(row=len(labels), column=0, padx=5, pady=10)

# Create a button to trigger the calculation
calculate_button = tk.Button(window, text="Calculate Probability", command=calculate_number, bd=2, highlightthickness=0)
calculate_button.grid(row=len(labels), column=1, padx=5, pady=10)

# Create a label to display the calculated number
result_label = tk.Label(window, text="Probability of having malignant breast tumor: ")

result_label.config(text="Probability of having malignant cancer:")
result_label.grid(row=len(labels) + 1, column=0, columnspan=2, padx=5, pady=5, sticky="n")


# Set the font style for the result label
font_style = Font(family="Merriweather", size=12, weight="bold")
result_label.config(font=font_style)

# Start the GUI event loop
window.mainloop()