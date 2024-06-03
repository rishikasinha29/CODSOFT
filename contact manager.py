import tkinter as tk
from tkinter import messagebox

# Contact Book List
contact_book = []

# Add Contact Function
def add_contact():
    name = name_entry.get()
    phone_number = phone_entry.get()
    if len(phone_number) != 10 or not phone_number.isdigit():
        messagebox.showwarning("Invalid Input", "Phone number must be 10 digits.")
        return
    contact_book.append([name, phone_number])
    messagebox.showinfo("Success", "Contact added successfully")
    update_contact_list()
    clear_entries()

# View Contact Function
def view_contact():
    phone_number = view_phone_entry.get()
    for contact in contact_book:
        if phone_number == contact[1]:
            messagebox.showinfo("Contact Found", f"Name: {contact[0]}, Phone Number: {contact[1]}")
            return
    messagebox.showwarning("Not Found", "Invalid phone number, please recheck.")

# Update Contact Function
def update_contact():
    old_phone_number = update_old_phone_entry.get()
    new_name = update_name_entry.get()
    new_phone_number = update_phone_entry.get()
    for contact in contact_book:
        if old_phone_number == contact[1]:
            contact[0] = new_name
            contact[1] = new_phone_number
            messagebox.showinfo("Success", "Contact updated successfully")
            update_contact_list()
            return
    messagebox.showwarning("Not Found", f"Contact not found with phone number: {old_phone_number}")

# Delete Contact Function
def delete_contact():
    phone_number = delete_phone_entry.get()
    for contact in contact_book:
        if phone_number == contact[1]:
            contact_book.remove(contact)
            messagebox.showinfo("Success", "Contact deleted successfully")
            update_contact_list()
            return
    messagebox.showwarning("Not Found", f"Contact not found with phone number: {phone_number}")

# Update Contact List Display
def update_contact_list():
    contact_list.delete(0, tk.END)
    for contact in contact_book:
        contact_list.insert(tk.END, f"{contact[0]} - {contact[1]}")

# Clear Entry Fields
def clear_entries():
    name_entry.delete(0, tk.END)
    phone_entry.delete(0, tk.END)
    view_phone_entry.delete(0, tk.END)
    update_old_phone_entry.delete(0, tk.END)
    update_name_entry.delete(0, tk.END)
    update_phone_entry.delete(0, tk.END)
    delete_phone_entry.delete(0, tk.END)

# GUI Setup
root = tk.Tk()
root.title("Contact Manager")

# Add Contact Frame
tk.Label(root, text="Add Contact").pack()
add_frame = tk.Frame(root)
add_frame.pack(pady=5)
tk.Label(add_frame, text="Name").grid(row=0, column=0)
tk.Label(add_frame, text="Phone Number").grid(row=1, column=0)
name_entry = tk.Entry(add_frame)
phone_entry = tk.Entry(add_frame)
name_entry.grid(row=0, column=1)
phone_entry.grid(row=1, column=1)
tk.Button(add_frame, text="Add", command=add_contact).grid(row=2, column=0, columnspan=2)

# View Contact Frame
tk.Label(root, text="View Contact").pack()
view_frame = tk.Frame(root)
view_frame.pack(pady=5)
tk.Label(view_frame, text="Phone Number").grid(row=0, column=0)
view_phone_entry = tk.Entry(view_frame)
view_phone_entry.grid(row=0, column=1)
tk.Button(view_frame, text="View", command=view_contact).grid(row=1, column=0, columnspan=2)

# Update Contact Frame
tk.Label(root, text="Update Contact").pack()
update_frame = tk.Frame(root)
update_frame.pack(pady=5)
tk.Label(update_frame, text="Old Phone Number").grid(row=0, column=0)
tk.Label(update_frame, text="New Name").grid(row=1, column=0)
tk.Label(update_frame, text="New Phone Number").grid(row=2, column=0)
update_old_phone_entry = tk.Entry(update_frame)
update_name_entry = tk.Entry(update_frame)
update_phone_entry = tk.Entry(update_frame)
update_old_phone_entry.grid(row=0, column=1)
update_name_entry.grid(row=1, column=1)
update_phone_entry.grid(row=2, column=1)
tk.Button(update_frame, text="Update", command=update_contact).grid(row=3, column=0, columnspan=2)

# Delete Contact Frame
tk.Label(root, text="Delete Contact").pack()
delete_frame = tk.Frame(root)
delete_frame.pack(pady=5)
tk.Label(delete_frame, text="Phone Number").grid(row=0, column=0)
delete_phone_entry = tk.Entry(delete_frame)
delete_phone_entry.grid(row=0, column=1)
tk.Button(delete_frame, text="Delete", command=delete_contact).grid(row=1, column=0, columnspan=2)

# Contact List Frame
tk.Label(root, text="Contact List").pack()
contact_list = tk.Listbox(root, width=50)
contact_list.pack(pady=5)

# Run the GUI
update_contact_list()
root.mainloop()
