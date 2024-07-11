import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np

# Créer la fenêtre principale
root = tk.Tk()
root.title("Analyse sentiment")
root.geometry("1200x800")
root.configure(bg='#1c1c3c')

# Définir les couleurs
bg_color = '#1c1c3c'
fg_color = '#ffffff'
highlight_color = '#0e76a8'
button_color = '#2a2a4a'
button_active_color = '#0e76a8'
plot_bg_color = '#2a2a4a'
tab_selected_color = '#7c4dcf'
tab_unselected_color = '#190d78'
tab_text_color = '#ffffff'
tab_selected_text_color = '#f1dd5b'

# Style pour les onglets et les widgets
style = ttk.Style()
style.theme_use('clam')  # Utiliser un thème qui supporte mieux la personnalisation
style.configure('TNotebook', background=bg_color, foreground=fg_color, padding=10)
style.configure('TNotebook.Tab', background=tab_unselected_color, foreground=tab_text_color, padding=[20, 10], font=('Arial', 14, 'bold'))
style.map('TNotebook.Tab', background=[('selected', tab_selected_color)], foreground=[('selected', tab_selected_text_color)])

style.configure('TFrame', background=bg_color)
style.configure('TLabel', background=bg_color, foreground=fg_color)

# Ajouter du style pour les frames des onglets
style.configure('Custom.TFrame', background=bg_color, borderwidth=2, relief="groove")

# Fonction pour créer un graphique en ligne
def create_line_plot(frame, title):
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    t = np.arange(0., 5., 0.2)
    ax.plot(t, t, 'r-', t, t**2, 'bs', t, t**3, 'g^')
    ax.set_title(title, color=fg_color)
    ax.set_facecolor(plot_bg_color)
    fig.patch.set_facecolor(bg_color)
    ax.spines['bottom'].set_color(fg_color)
    ax.spines['top'].set_color(fg_color)
    ax.spines['right'].set_color(fg_color)
    ax.spines['left'].set_color(fg_color)
    ax.xaxis.label.set_color(fg_color)
    ax.yaxis.label.set_color(fg_color)
    ax.tick_params(axis='x', colors=fg_color)
    ax.tick_params(axis='y', colors=fg_color)
    
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Fonction pour créer un graphique en secteurs
def create_pie_chart(frame, title):
    fig, ax = plt.subplots(figsize=(6, 3), dpi=100)
    labels = 'A', 'B', 'C', 'D'
    sizes = [15, 30, 45, 10]
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
    ax.set_title(title, color=fg_color)
    fig.patch.set_facecolor(bg_color)
    
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Fonction pour créer un graphique en barres
def create_bar_plot(frame, title):
    fig, ax = plt.subplots(figsize=(6, 3), dpi=100)
    labels = ['G1', 'G2', 'G3', 'G4', 'G5']
    men_means = [20, 35, 30, 35, 27]
    women_means = [25, 32, 34, 20, 25]
    
    x = np.arange(len(labels))
    width = 0.35
    rects1 = ax.bar(x - width/2, men_means, width, label='Men', color='#ff9999')
    rects2 = ax.bar(x + width/2, women_means, width, label='Women', color='#66b3ff')
    
    ax.set_ylabel('Scores', color=fg_color)
    ax.set_title(title, color=fg_color)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=fg_color)
    ax.legend(facecolor=bg_color, edgecolor=fg_color)
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(plot_bg_color)
    
    for spine in ax.spines.values():
        spine.set_edgecolor(fg_color)
    
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Créer un notebook (onglets)
notebook_frame = tk.Frame(root, bg=bg_color)
notebook_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

notebook = ttk.Notebook(notebook_frame, style='TNotebook')
notebook.pack(expand=True)

# Créer des frames pour chaque onglet avec un style personnalisé
tab1 = ttk.Frame(notebook, width=1000, height=800, style='Custom.TFrame')
tab2 = ttk.Frame(notebook, width=1000, height=800, style='Custom.TFrame')
tab3 = ttk.Frame(notebook, width=1000, height=800, style='Custom.TFrame')

tab1.pack(fill='both', expand=True)
tab2.pack(fill='both', expand=True)
tab3.pack(fill='both', expand=True)

notebook.add(tab1, text='Title 01')
notebook.add(tab2, text='Title 02')
notebook.add(tab3, text='Title 03')

# Ajouter des widgets au premier onglet
line_plot_frame = ttk.Frame(tab1, width=600, height=200, style='Custom.TFrame')
line_plot_frame.pack(side=tk.TOP, pady=20, padx=20, fill=tk.BOTH, expand=True)
create_line_plot(line_plot_frame, "Line Plot Title Here")

pie_chart_frame = ttk.Frame(tab1, width=600, height=200, style='Custom.TFrame')
pie_chart_frame.pack(side=tk.LEFT, pady=20, padx=20, fill=tk.BOTH, expand=True)
create_pie_chart(pie_chart_frame, "Pie Chart Title Here")

bar_plot_frame = ttk.Frame(tab1, width=600, height=200, style='Custom.TFrame')
bar_plot_frame.pack(side=tk.RIGHT, pady=20, padx=20, fill=tk.BOTH, expand=True)
create_bar_plot(bar_plot_frame, "Bar Plot Title Here")

# Ajouter des widgets aux deuxième et troisième onglets de manière similaire...

root.mainloop()
