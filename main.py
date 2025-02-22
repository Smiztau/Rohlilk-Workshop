import tkinter as tk
from tkinter import ttk

def toggle_check(event):
    item_id = tree.identify_row(event.y)
    if not item_id:
        return
    
    current_value = tree.item(item_id, 'values')[1]
    new_value = "V" if current_value != "V" else "X"
    
    tree.item(item_id, values=(tree.item(item_id, 'values')[0], new_value))

def run_code():
    print("hello world")

# Create main window
root = tk.Tk()
root.title("Task Selection")

# Create table
columns = ("Task", "Select")
tree = ttk.Treeview(root, columns=columns, show="headings")
tree.heading("Task", text="Task")
tree.heading("Select", text="Select")

tasks = ["Train with rolling avg", "Train without rolling avg", "Train by warehouse", "Train by uniqueIds"]
for task in tasks:
    tree.insert("", tk.END, values=(task, "X"))  # Default unchecked

tree.bind("<Button-1>", toggle_check)

tree.pack(expand=True, fill=tk.BOTH)

# Create button to run code
run_button = tk.Button(root, text="Run", command=run_code)
run_button.pack(pady=10)

root.mainloop()
