import random
import re
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from textblob import TextBlob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import emoji
from fpdf import FPDF
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from dateutil import parser
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Global variable to store the dataset and analysis results
donnees = None
trend_df = None
most_tweets_date = None
least_tweets_date = None
most_liked_tweet = None
most_likes_index = None
tzinfos = {
    "PDT": -7 * 3600,
    "PST": -8 * 3600,
    "CDT": -5 * 3600,
    "CST": -6 * 3600,
    "EDT": -4 * 3600,
    "EST": -5 * 3600,
    "MDT": -6 * 3600,
    "MST": -7 * 3600,
    "UTC": 0,
}

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

# Function to analyze sentiment

def analyze_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return 'Positive ' + emoji.emojize(':smile:')
    elif polarity < 0:
        return 'Negative ' + emoji.emojize(':frowning_face:')
    else:
        return 'Neutral ' + emoji.emojize(':neutral_face:')

# Function to display sentiment analysis result for a single tweet
def analyze_single_tweet():
    tweet = tweet_entry.get("1.0", tk.END).strip()
    if tweet:
        sentiment = analyze_sentiment(tweet)
        result_label.config(text=f"Sentiment: {sentiment}", font=("Arial", 12, "bold"))
    else:
        messagebox.showwarning("Input Error", "Please enter a tweet to analyze.")

# Function to save the results to a PDF
def save_to_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Add the text results
    pdf.cell(200, 10, txt="Sentiment Analysis Results", ln=True, align='C')

    # Add most and least tweets dates
    pdf.cell(200, 10, txt=f"Date with the most tweets: {most_tweets_date}", ln=True, align='L')
    pdf.cell(200, 10, txt=f"Date with the least tweets: {least_tweets_date}", ln=True, align='L')

    # Add most liked tweet
    pdf.cell(200, 10, txt=f"Most Liked Tweet: {most_liked_tweet}", ln=True, align='L')
    pdf.cell(200, 10, txt=f"Likes: {donnees.loc[most_likes_index, 'Number of Likes']}", ln=True, align='L')

    # Save the plots as images and add them to the PDF
    figs = [fig100,fig0,fig00,fig1, fig3, fig4, fig5, fig6, fig7, fig8]
    for i, fig in enumerate(figs):
        img_path = f'plot_{i}.png'
        fig.savefig(img_path, format='png')
        pdf.add_page()
        pdf.image(img_path, x=10, y=10, w=190)
        os.remove(img_path)

    # Save the PDF to the desktop
    desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
    pdf_path = os.path.join(desktop, 'sentiment_analysis_results.pdf')
    pdf.output(pdf_path)

    messagebox.showinfo("Success", f"Results successfully saved to PDF at {pdf_path}!")

# Function to load and analyze a dataset
def analyze_dataset():
    global train_accuracy, test_accuracy , donnees, trend_df, most_tweets_date, least_tweets_date, most_liked_tweet, most_likes_index,fig00,fig0 ,fig1, fig100, fig3, fig4, fig5, fig6, fig7, fig8
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        try:
            donnees = pd.read_csv(file_path)

            # Fill NaN values in 'Tweet' with empty strings
            donnees['Tweet'].fillna('', inplace=True)

            # Convert all values in 'Tweet' to strings
            donnees['Tweet'] = donnees['Tweet'].astype(str)

            # Apply sentiment analysis to the 'Tweet' column
            donnees['Sentiment'] = donnees['Tweet'].apply(analyze_sentiment)

            # Create trend_df
            if 'Date Created' in donnees.columns:
                donnees['Date'] = pd.to_datetime(donnees['Date Created']).dt.date
            elif 'created_at' in donnees.columns:
                donnees['Date'] = pd.to_datetime(donnees['created_at']).dt.date
            else:
                messagebox.showerror("Error", "Neither 'Date Created' nor 'created_at' column found in the DataFrame.")
                return

            # Group by date and sentiment
            sentiment_trend_df = donnees.groupby(['Date', 'Sentiment'])['Tweet'].count().reset_index(name='Count')

            # Clear the result frame (Assuming result_content_frame is defined in your actual GUI)
            # for widget in result_content_frame.winfo_children():
            #     widget.destroy()
            df = pd.DataFrame({
                'Tweet': donnees['Tweet'],
                'Sentiment': donnees['Sentiment']
            })
            # Plot various graphs
            def plot_and_show(fig):
                canvas = FigureCanvasTkAgg(fig, master=result_content_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(pady=10)

            # Prepare data for the bar plot
            df_for_plot = df.groupby('Sentiment').size().reset_index(name='Count')
            # Exemple de DataFrame avec des textes et des labels de sentiment
            # Prepare the features for the model
            vectorizer = TfidfVectorizer()
            features_array = vectorizer.fit_transform(donnees['Tweet']).toarray()

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(features_array, donnees['Sentiment'], test_size=0.1)

            # Instantiate the LogisticRegression model
            model = LogisticRegression()

            # Fit the model on the training data
            model.fit(X_train, y_train)

            # Evaluate the model on the testing data
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            # Print the accuracy score
            print("Accuracy:", accuracy)

            # Calculate the performance of the model on the training set
            train_accuracy = model.score(X_train, y_train)

            # Calculate the performance of the model on the test set
            test_accuracy = model.score(X_test, y_test)

            # Print the results
            print("Training Accuracy:", train_accuracy)
            print("Test Accuracy:", test_accuracy)

            # Plot the accuracy scores
            fig100=plt.figure(figsize=(10, 6))
            plt.bar(x=['Training Accuracy', 'Test Accuracy'], height=[train_accuracy, test_accuracy], color=['red', 'green'])
            plt.title('Model Accuracy Comparison')
            plt.xlabel('Accuracy Type')
            plt.ylabel('Accuracy Score')
            plt.tight_layout()
            plot_and_show(fig100)



            # Create a bar plot
            fig0 = plt.figure(figsize=(12, 8))
            sns.barplot(x='Sentiment', y='Count', data=df_for_plot)
            plt.title('Sentiment Analysis of Tweets')
            plt.xlabel('Sentiment')
            plt.ylabel('Number of Tweets')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plot_and_show(fig0)

            # Prepare data for the pie chart
            labels = df['Sentiment'].value_counts().index.to_list()
            sizes =  df['Sentiment'].value_counts().to_list()

            # Create a pie chart
            fig00, ax = plt.subplots()
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            ax.set_title('Sentiment Distribution')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plot_and_show(fig00)

            # Number of Tweets by Sentiment and Date
            fig1 = plt.figure(figsize=(12, 8))
            sns.barplot(data=sentiment_trend_df, x='Date', y='Count', hue='Sentiment')
            plt.title('Number of Tweets by Sentiment and Date')
            plt.xlabel('Date')
            plt.ylabel('Number of Tweets')
            plt.legend(title='Sentiment')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plot_and_show(fig1)

            # Number of Tweets by Sentiment and Date (Stacked)
            # fig2 = plt.figure(figsize=(12, 8))
            # sns.barplot(data=sentiment_trend_df, x='Date', y='Count', hue='Sentiment')
            # plt.title('Number of Tweets by Sentiment and Date (Stacked)')
            # plt.xlabel('Date')
            # plt.ylabel('Number of Tweets')
            # plt.legend(title='Sentiment')
            # plt.xticks(rotation=45, ha='right')
            # plt.tight_layout()
            # plot_and_show(fig2)

            # Likes trend
            if 'Number of Likes' in donnees.columns:
                likes_trend_df = donnees.groupby('Date')['Number of Likes'].sum().reset_index(name='Total Likes')
                likes_trend_df = likes_trend_df.sort_values('Date')
                fig3 = plt.figure(figsize=(12, 8))
                sns.lineplot(data=likes_trend_df, x='Date', y='Total Likes')
                plt.title('Number of Likes Over Time')
                plt.xlabel('Date')
                plt.ylabel('Number of Likes')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plot_and_show(fig3)

                likes_trend_df['Percentage Change'] = likes_trend_df['Total Likes'].pct_change() * 100
                fig4 = plt.figure(figsize=(12, 8))
                sns.lineplot(data=likes_trend_df, x='Date', y='Percentage Change')
                plt.title('Percentage Change in Likes Over Time')
                plt.xlabel('Date')
                plt.ylabel('Percentage Change')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plot_and_show(fig4)

            # Sentiment over time
            fig5 = plt.figure(figsize=(12, 8))
            for sentiment in ['Positive ' + emoji.emojize(':smile:'),
                              'Negative ' + emoji.emojize(':frowning_face:'),
                              'Neutral ' + emoji.emojize(':neutral_face:')]:
                sentiment_df = sentiment_trend_df[sentiment_trend_df['Sentiment'] == sentiment]
                sns.lineplot(data=sentiment_df, x='Date', y='Count', label=sentiment)
            plt.title('Sentiment of Tweets Over Time')
            plt.xlabel('Date')
            plt.ylabel('Number of Tweets')
            plt.legend()
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plot_and_show(fig5)

            # Number of tweets by date
            trend_df = donnees.groupby('Date')['Tweet'].count().reset_index(name='Count')
            trend_df = trend_df.sort_values('Date')
            fig6 = plt.figure(figsize=(12, 8))
            sns.barplot(data=trend_df, x='Date', y='Count')
            plt.title('Number of Tweets by Date')
            plt.xlabel('Date')
            plt.ylabel('Number of Tweets')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plot_and_show(fig6)

            fig7 = plt.figure(figsize=(12, 8))
            sns.lineplot(data=trend_df, x='Date', y='Count')
            plt.title('Number of Tweets Over Time')
            plt.xlabel('Date')
            plt.ylabel('Number of Tweets')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plot_and_show(fig7)

            trend_df['Percentage Change'] = trend_df['Count'].pct_change() * 100
            fig8 = plt.figure(figsize=(12, 8))
            sns.lineplot(data=trend_df, x='Date', y='Percentage Change')
            plt.title('Percentage Change in Tweets Over Time')
            plt.xlabel('Date')
            plt.ylabel('Percentage Change')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plot_and_show(fig8)

            most_tweets_date = trend_df['Date'][trend_df['Count'].idxmax()]
            least_tweets_date = trend_df['Date'][trend_df['Count'].idxmin()]
            result_label.config(text=f"Date with the most tweets: {most_tweets_date}\nDate with the least tweets: {least_tweets_date}", font=("Arial", 12, "bold"))

            # Generate word clouds for each sentiment
            for sentiment in ['Positive ' + emoji.emojize(':smile:'),
                              'Negative ' + emoji.emojize(':frowning_face:'),
                              'Neutral ' + emoji.emojize(':neutral_face:')]:
                df_filtered = donnees[donnees['Sentiment'] == sentiment]
                text = " ".join(df_filtered['Tweet'].tolist())
                wordcloud = WordCloud(width=800, height=600, background_color="black").generate(text)
                fig9 = plt.figure(figsize=(10, 6))
                plt.imshow(wordcloud, interpolation="bilinear")
                plt.axis("off")
                plt.title(f"WordCloud for {sentiment} Sentiment")
                plt.tight_layout()
                plot_and_show(fig9)

            # Display trend texts in a table format
            trend_texts_df = donnees.copy()
            trend_texts_df = trend_texts_df.groupby('Date')['Tweet'].apply(list).reset_index(name='Tweets')
            trend_texts_df = trend_texts_df.sort_values('Date')
            trend_tree.delete(*trend_tree.get_children())  # Clear the treeview
            for date, tweets in trend_texts_df.values:
                for i, tweet in enumerate(tweets[:5]):  # Display only the first 5 tweets
                    trend_tree.insert('', 'end', values=(i + 1, tweet))

            # Display the tweet with the most likes
            most_likes_index = donnees['Number of Likes'].idxmax()
            most_liked_tweet = donnees.loc[most_likes_index, 'Tweet']
            most_likes_label.config(text=f"Most Liked Tweet: {most_liked_tweet}\nLikes: {donnees.loc[most_likes_index, 'Number of Likes']}", font=("Arial", 12, "bold"))

            #display accuracy
            train_accuracy_label.config(text=f"Training Accuracy: {train_accuracy*100}%", font=("Arial", 12, "bold"))
            test_accuracy_label.config(text=f"Test Accuracy: {test_accuracy*100}%", font=("Arial", 12, "bold"))

            # Enable and display the save to PDF button
            save_pdf_button.config(state='normal')
            # display_trends_button.config(state='normal')  # Enable the display trends button

            # Show success message
            messagebox.showinfo("Success", "Dataset successfully uploaded and analyzed!")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

def clean_tweet(tweet):
        if pd.isna(tweet):
            tweet = ''
        tweet = re.sub(r'http\S+', '', tweet)
        tweet = re.sub(r'@\w+', '', tweet)
        tweet = re.sub(r'#\w+', '', tweet)
        tweet = re.sub(r'\W', ' ', tweet)
        tweet = re.sub(r'\d', '', tweet)
        tweet = re.sub(r'\s+', ' ', tweet).strip()
        return tweet
def remove_stopwords(tweet):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(tweet)
    filtered_tweet = [word for word in word_tokens if word.lower() not in stop_words]
    return ' '.join(filtered_tweet)


def load_dataset():
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        try:
            data = pd.read_csv(file_path, encoding='utf-8', header=None, low_memory=False)
        except UnicodeDecodeError:
            try:
                data = pd.read_csv(file_path, encoding='latin1', header=None, low_memory=False)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {e}")
                return

        # Assigning column names
        data.columns = ['Unused', 'ID', 'Timestamp', 'Query', 'Username', 'Content', 'cleaned_tweet', 'sentiment']

        # Transforming the dataset to match the desired format
        data_transformed = data[['ID', 'Timestamp', 'Content']].copy()
        data_transformed.columns = ['ID', 'Date Created', 'Tweet']

        # Adding random 'Number of Likes' column
        data_transformed['Number of Likes'] = [random.randint(2, 2000) for _ in range(len(data_transformed))]

        # Filling NaN values in 'Tweet' with empty strings
        data_transformed['Tweet'].fillna('', inplace=True)

        # Adding additional columns
        data_transformed['Tweet Length'] = data_transformed['Tweet'].apply(len)
        
        # Convert 'Date Created' to desired format
        data_transformed['Date Created'] = data_transformed['Date Created'].apply(convert_date)
        
        # Drop rows with NaN dates
        data_transformed.dropna(subset=['Date Created'], inplace=True)

        # Clean tweets and remove stopwords
        data_transformed['Tweet'] = data_transformed['Tweet'].apply(clean_tweet).apply(remove_stopwords)

        # Check if there is any data left after dropping NaNs
        if data_transformed.empty:
            messagebox.showwarning("Warning", "No valid dates found. Dataset is empty after dropping NaNs.")
            return

        if data_transformed is not None:
            desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
            save_path = os.path.join(desktop_path, "transformed_dataset.csv")
            data_transformed.to_csv(save_path, index=False, encoding='utf-8')
            messagebox.showinfo("Info", "Dataset saved successfully to Desktop")
        else:
            messagebox.showerror("Error", "No data to save")
        
        messagebox.showinfo("Info", f"Dataset loaded and transformed with {len(data)} records")
    else:
        messagebox.showwarning("Warning", "No file selected")
        return None
def convert_date(date_str):
    try:
        dt = parser.parse(date_str, tzinfos=tzinfos)
        return dt.isoformat()
    except Exception as e:
        return None

# def output_dataset(data_transformed):
#     if data_transformed is not None:
#         file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
#         print(file_path)
#         if file_path:
#             data_transformed.to_csv(file_path, index=False, encoding='utf-8')
#             messagebox.showinfo("Info", "Dataset saved successfully")
#         else:
#             messagebox.showwarning("Warning", "No file selected")
#     else:
#         messagebox.showerror("Error", "No data to save")

# Function to display trend texts when the button is clicked
# def display_trends():
#     global trend_df
#     trend_texts_output.config(state='normal')
#     trend_texts_output.delete("1.0", tk.END)
#     for date, count in trend_df.values:
#         trend_texts_output.insert(tk.END, f"{date}: {count} tweets\n")
#     trend_texts_output.config(state='disable')

# Initialize the main application window
root = tk.Tk()
root.title("Sentiment Analysis Tool")
root.geometry("1200x800")  # Set window size

# Apply a theme
style = ttk.Style(root)
style.theme_use('clam')  # You can choose between 'clam', 'alt', 'default', and 'classic'
style.configure('TNotebook', background=bg_color, foreground=fg_color, padding=10)
style.configure('TNotebook.Tab', background=tab_unselected_color, foreground=tab_text_color, padding=[10, 10], font=('Arial', 14, 'bold'))
style.map('TNotebook.Tab', background=[('selected', tab_selected_color)], foreground=[('selected', tab_selected_text_color)])

style.configure('TFrame', background=bg_color)
style.configure('TLabel', background=bg_color, foreground=fg_color)

# Ajouter du style pour les frames des onglets
style.configure('Custom.TFrame', background=bg_color, borderwidth=2, relief="groove")
# Create notebook for tabs
notebook = ttk.Notebook(root,style='TNotebook')
notebook.pack(pady=10, expand=True)

# Create frames for each tab
home_frame = ttk.Frame(notebook, width=1200, height=800, padding=10)
result_frame = ttk.Frame(notebook, width=1200, height=800, padding=10)

home_frame.pack(fill='both', expand=True)
result_frame.pack(fill='both', expand=True)

notebook.add(home_frame, text='Accueil')
notebook.add(result_frame, text='Résultats')

# Home tab widgets
tweet_label = ttk.Label(home_frame, text="Enter a tweet for analysis:", font=("Arial", 12))
tweet_label.pack(pady=10)

tweet_entry = tk.Text(home_frame, height=2, width=50, font=("Arial", 12))
tweet_entry.pack(pady=10)

analyze_tweet_button = ttk.Button(home_frame, text="Analyze Tweet", command=analyze_single_tweet)
analyze_tweet_button.pack(pady=5)

load_button = tk.Button(home_frame, text="Clean Data", command=load_dataset)
load_button.pack(pady=20)

result_label = ttk.Label(home_frame, text="", font=("Arial", 12, "bold"))
result_label.pack(pady=10)

upload_button = ttk.Button(home_frame, text="Upload Dataset and Analyze", command=analyze_dataset)
upload_button.pack(pady=20)

# Button to save results to PDF (initially disabled)
save_pdf_button = ttk.Button(home_frame, text="Save Results to PDF", command=save_to_pdf)
save_pdf_button.pack(pady=5)
save_pdf_button.config(state='disabled')

# Button to display trends (initially disabled)
# display_trends_button = ttk.Button(home_frame, text="Afficher les tendances des tweets (textes uniquement)", command="display_trends")
# display_trends_button.pack(pady=5)
# display_trends_button.config(state='disabled')

# Scrollable result frame
canvas = tk.Canvas(result_frame, width=1200, height=800)
scrollbar = ttk.Scrollbar(result_frame, orient="vertical", command=canvas.yview)
scrollable_frame = ttk.Frame(canvas)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(
        scrollregion=canvas.bbox("all")
    )
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

# Result content frame inside the scrollable frame
result_content_frame = ttk.Frame(scrollable_frame)
result_content_frame.pack(fill="both", expand=True)

# Label to display trend texts in the results tab
trend_texts_label = ttk.Label(result_content_frame, text="", font=("Arial", 10), justify=tk.LEFT)
trend_texts_label.pack(pady=10)

# Treeview widget to display trend texts in a table format
columns = ('#1', '#2')
trend_tree = ttk.Treeview(result_content_frame, columns=columns, show='headings', height=5)
trend_tree.heading('#1', text='#', anchor=tk.W)
trend_tree.heading('#2', text='tendency Tweet', anchor=tk.W)
trend_tree.column('#1', width=50, anchor=tk.W)
trend_tree.column('#2', width=1050, anchor=tk.W)
trend_tree.pack(pady=10)

# # Text widget to display trend texts output
# trend_texts_output = tk.Text(result_content_frame, height=10, width=100, state='disabled', wrap='word', font=("Arial", 10))
# trend_texts_output.pack(pady=10)

# Label to display the most liked tweet
most_likes_label = ttk.Label(result_content_frame, text="", font=("Arial", 12, "bold"), justify=tk.LEFT, wraplength=1050)
most_likes_label.pack(pady=10)

train_accuracy_label = ttk.Label(result_content_frame, text="", font=("Arial", 12, "bold"), justify=tk.LEFT, wraplength=1050)
train_accuracy_label.pack(pady=10)

test_accuracy_label = ttk.Label(result_content_frame, text="", font=("Arial", 12, "bold"), justify=tk.LEFT, wraplength=1050)
test_accuracy_label.pack(pady=10)

trend_texts_label = ttk.Label(result_content_frame, text="", font=("Arial", 12, "bold"), justify=tk.LEFT, wraplength=1050)
trend_texts_label.pack(pady=10)
# Run the application
root.mainloop()

# After analyzing the sentiments and storing results in a dataframe (df)


