import tkinter as tk
from tkinter import messagebox
import subprocess

# Function to execute the selected scripts
def run_scripts():
    commands = []
    
    # If "Calculate Data" is checked, run script.py
    if calculate_data_var.get():
        commands.append(["python", "data.py"])
    
    # If "Test" is checked, run test_by_warehouse.py
    if test_var.get():
        commands.append(["python", "test_by_warehouse.py"])
    
    # Execute the selected commands
    for command in commands:
        try:
            subprocess.run(command, check=True)
            messagebox.showinfo("Success", f"{command[1]} executed successfully!")
        except subprocess.CalledProcessError:
            messagebox.showerror("Error", f"Failed to execute {command[1]}")

# Create main window
root = tk.Tk()
root.title("Data Processing & Testing")
root.geometry("400x300")

# Title Label
tk.Label(root, text="Select Actions to Perform", font=("Arial", 14)).pack(pady=10)

# Checkboxes
calculate_data_var = tk.BooleanVar()
test_var = tk.BooleanVar()

calculate_checkbox = tk.Checkbutton(root, text="Calculate Data", variable=calculate_data_var)
calculate_checkbox.pack(anchor="w", padx=20)

test_checkbox = tk.Checkbutton(root, text="Test", variable=test_var)
test_checkbox.pack(anchor="w", padx=20)

# Run Button
run_button = tk.Button(root, text="Run", command=run_scripts, font=("Arial", 12))
run_button.pack(pady=20)

# Start GUI
root.mainloop()
