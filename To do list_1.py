import tkinter as tk
from tkinter import messagebox

class TaskManagerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Task Manager")
        self.tasks = []
        
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(pady=10)
        
        self.task_listbox = tk.Listbox(self.main_frame, width=50, height=10)
        self.task_listbox.pack(side=tk.LEFT)
        
        self.scrollbar = tk.Scrollbar(self.main_frame)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.task_listbox.config(yscrollcommand=self.scrollbar.set)
        self.scrollbar.config(command=self.task_listbox.yview)
        
        self.task_entry = tk.Entry(self.root, width=50)
        self.task_entry.pack(pady=10)
        
        self.add_task_button = tk.Button(self.root, text="Add Task", command=self.add_task)
        self.add_task_button.pack(pady=5)
        
        self.remove_task_button = tk.Button(self.root, text="Remove Task", command=self.remove_task)
        self.remove_task_button.pack(pady=5)
        
        self.complete_task_button = tk.Button(self.root, text="Mark as Completed", command=self.mark_task_completed)
        self.complete_task_button.pack(pady=5)
        
    def add_task(self):
        task_description = self.task_entry.get()
        if task_description != "":
            self.tasks.append({"description": task_description, "completed": False})
            self.task_listbox.insert(tk.END, task_description)
            self.task_entry.delete(0, tk.END)
        else:
            messagebox.showwarning("Warning", "You must enter a task.")
    
    def remove_task(self):
        try:
            selected_task_index = self.task_listbox.curselection()[0]
            self.task_listbox.delete(selected_task_index)
            del self.tasks[selected_task_index]
        except IndexError:
            messagebox.showwarning("Warning", "You must select a task.")
    
    def mark_task_completed(self):
        try:
            selected_task_index = self.task_listbox.curselection()[0]
            self.tasks[selected_task_index]["completed"] = True
            task_description = self.tasks[selected_task_index]["description"]
            self.task_listbox.delete(selected_task_index)
            self.task_listbox.insert(selected_task_index, f"{task_description} (Completed)")
        except IndexError:
            messagebox.showwarning("Warning", "You must select a task.")

if __name__ == "__main__":
    root = tk.Tk()
    app = TaskManagerApp(root)
    root.mainloop()
