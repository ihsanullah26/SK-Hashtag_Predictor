import tkinter as tk
from tkinter import messagebox, scrolledtext
import joblib
import re
import string
import emoji
import numpy as np
import pyperclip

# --- 1. DATA PREPROCESSING & MODEL LOADING ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|@\S+', '', text)
    text = emoji.replace_emoji(text, replace='')
    text = text.translate(str.maketrans('', '', string.punctuation))
    return ' '.join(text.split())

try:
    tfidf = joblib.load('tfidf.pkl')
    bln = joblib.load('binarizer.pkl')
    svc = joblib.load('model_svc.pkl')
    nb = joblib.load('model_nb.pkl')
    lr = joblib.load('model_lr.pkl')
except Exception as e:
    print(f"Error: {e}")

# --- 2. LOGIC ---
def get_predictions():
    raw_text = text_input.get("1.0", tk.END).strip()
    if not raw_text:
        messagebox.showwarning("Input Error", "Please enter some text!")
        return

    try:
        cleaned = clean_text(raw_text)
        vec = tfidf.transform([cleaned])
        
        if vec.nnz == 0:
            update_display("Model does not recognize these words.")
            return

        s_svc = svc.decision_function(vec)[0]
        s_nb = nb.predict_proba(vec)[0]
        s_lr = lr.predict_proba(vec)[0]
        
        combined_scores = s_svc + s_nb + s_lr
        top_indices = np.argsort(combined_scores)[-6:] 
        suggested_tags = [bln.classes_[i] for i in top_indices[::-1]]
        
        update_display(" ".join(suggested_tags))
    except Exception as e:
        messagebox.showerror("Error", str(e))

def update_display(content):
    result_display.config(state=tk.NORMAL)
    result_display.delete("1.0", tk.END)
    result_display.insert(tk.END, content)
    result_display.config(state=tk.DISABLED)

def copy_to_clipboard():
    content = result_display.get("1.0", tk.END).strip()
    if content and "Model does not" not in content:
        pyperclip.copy(content)
        copy_btn.config(text="✓ COPIED", bg="#7dd3fc")
        root.after(1500, lambda: copy_btn.config(text="📋 Copy to Clipboard", bg="#e0f2fe"))

def reset_all():
    text_input.delete("1.0", tk.END)
    update_display("")

# --- 3. GUI SETUP ---
root = tk.Tk()
root.title("SK Hashtag Generator")
root.geometry("650x850")
root.configure(bg="#ffffff") 

# Hover Styling
def on_enter_gen(e): predict_btn.config(bg="#0284c7")
def on_leave_gen(e): predict_btn.config(bg="#0ea5e9")

# 1. HEADER
header_frame = tk.Frame(root, bg="#0ea5e9", pady=40)
header_frame.pack(fill="x")

header_label = tk.Label(header_frame, text="✨ SK HASHTAG GENERATOR", 
                        font=("Arial Black", 28, "bold"), 
                        bg="#0ea5e9", fg="white")
header_label.pack()

# 2. MAIN AREA
main_frame = tk.Frame(root, bg="#ffffff")
main_frame.pack(padx=60, pady=30, fill="both", expand=True)

# BOLD INPUT LABEL
tk.Label(main_frame, text="ENTER YOUR POST CONTENT", font=("Helvetica", 14, "bold"), 
         bg="#ffffff", fg="#1e293b").pack(anchor="w", pady=(0, 10))

# --- IMPROVED TEXT INPUT ---
# Note: font=("Helvetica", 14, "bold") makes the user's typing very prominent
text_input = scrolledtext.ScrolledText(main_frame, height=6, width=50, 
                                      font=("Helvetica", 14, "bold"),
                                      bg="#f0f9ff", fg="#0369a1", 
                                      insertbackground="#0ea5e9",
                                      bd=0, highlightthickness=2, 
                                      highlightbackground="#bae6fd",
                                      padx=20, pady=20)
text_input.pack(fill="x")

# 3. GENERATE BUTTON
predict_btn = tk.Button(main_frame, text="GENERATE HASHTAGS", command=get_predictions, 
                       bg="#0ea5e9", fg="white", font=("Helvetica", 14, "bold"), 
                       relief="flat", pady=18, cursor="hand2", bd=0)
predict_btn.pack(fill="x", pady=25)
predict_btn.bind("<Enter>", on_enter_gen)
predict_btn.bind("<Leave>", on_leave_gen)

# 4. OUTPUT SECTION
tk.Label(main_frame, text="AI SUGGESTIONS", font=("Helvetica", 14, "bold"), 
         bg="#ffffff", fg="#0284c7").pack(anchor="w", pady=(10, 10))

# OUTPUT DISPLAY
result_display = tk.Text(main_frame, height=4, width=50, font=("Helvetica", 18, "bold"), 
                         fg="#0369a1", bg="#f0f9ff", bd=0, padx=25, pady=25, 
                         state=tk.DISABLED, wrap="word",
                         highlightthickness=2, highlightbackground="#bae6fd")
result_display.pack(fill="x")

# 5. BOTTOM BUTTONS
btn_frame = tk.Frame(main_frame, bg="#ffffff")
btn_frame.pack(fill="x", pady=20)

copy_btn = tk.Button(btn_frame, text="📋 Copy to Clipboard", command=copy_to_clipboard, 
                    bg="#e0f2fe", fg="#0369a1", font=("Helvetica", 11, "bold"), 
                    relief="flat", pady=12, cursor="hand2", width=25)
copy_btn.pack(side="left", expand=True, padx=5)

reset_btn = tk.Button(btn_frame, text="Clear", command=reset_all, 
                     bg="#f1f5f9", fg="#64748b", font=("Helvetica", 11, "bold"), 
                     relief="flat", pady=12, cursor="hand2", width=10)
reset_btn.pack(side="left", padx=5)

root.mainloop()