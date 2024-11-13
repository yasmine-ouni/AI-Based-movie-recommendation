import os
from flask import Flask, render_template, request
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__)

# Step 1: Perform EDA on the DataFrame
def perform_eda(df):
    """
    Perform Exploratory Data Analysis on the DataFrame.
    """
    missing_values = df.isnull().sum()
    numerical_descriptive = df.describe()
    genres_distribution = df['genres'].value_counts()
    release_year_distribution = df['release_year'].value_counts().head(10)
    imdb_score_distribution = df['imdb_score'].describe()
    unique_values_in_categorical = {col: df[col].nunique() for col in df.select_dtypes(include=['object']).columns}
    # Identify and drop duplicate rows based on the 'ID' column (or whatever column represents the ID)
    df = df.drop_duplicates(subset='id', keep='first')


# Step 2: Initialize Chroma DB and Add Movie Embeddings
import chromadb.errors  # Make sure to import chromadb errors if not already

# Step 2: Initialize Chroma DB and Add Movie Embeddings
def initialize_chroma_db(df):
    """
    Initialize Chroma DB and add movie embeddings for retrieval.
    """
    client = chromadb.Client()
    collection_name = "movies_collection"
    
    # Check if collection exists; if not, create it
    try:
        collection = client.get_collection(collection_name)
    except chromadb.errors.InvalidCollectionException:
        collection = client.create_collection(collection_name)
    
    sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def combine_features(row):
        description = row['description'] if pd.notnull(row['description']) else 'No description available'
        genres = ', '.join(row['genres']) if isinstance(row['genres'], list) else (row['genres'] if pd.notnull(row['genres']) else 'Unknown genres')
        imdb_score = row['imdb_score'] if pd.notnull(row['imdb_score']) else 'Unknown'
        release_year = row['release_year'] if pd.notnull(row['release_year']) else 'Unknown'
        return f"{description} | Genres: {genres} | IMDb Score: {imdb_score} | Release Year: {release_year}"

    # Handle missing values in 'title' and 'combined_features' columns
    df['title'] = df['title'].fillna("unknown")
    df['combined_features'] = df.apply(combine_features, axis=1)
    df = df.drop_duplicates(subset='id', keep='first')

    # Generate embeddings for combined features
    embeddings = [sentence_model.encode(text).tolist() for text in df['combined_features']]

    # Ensure unique IDs by appending an index to duplicates
    id_counts = {}
    ids = []
    for title in df['title']:
        title_key = str(title).replace(" ", "_").lower()
        if title_key in id_counts:
            id_counts[title_key] += 1
            unique_id = f"{title_key}_{id_counts[title_key]}"
        else:
            id_counts[title_key] = 1
            unique_id = title_key
        ids.append(unique_id)

    collection.add(
        ids=ids,
        documents=df['combined_features'].tolist(),
        metadatas=df[['title', 'description', 'genres', 'imdb_score', 'release_year']].to_dict(orient='records'),
        embeddings=embeddings
    )

    return client, collection

# Step 3: Retrieve the most relevant movies from Chroma DB based on user input
def get_relevant_movies(user_input, collection, top_k=5):
    """
    Retrieve the top_k most relevant movies from Chroma DB based on cosine similarity to user input.
    """
    sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    user_embedding = sentence_model.encode(user_input).tolist()
    results = collection.query(query_embeddings=[user_embedding], n_results=top_k)

    recommendations = pd.DataFrame(results['documents'])
    recommendations['score'] = results['distances']
    return recommendations[['title', 'description', 'genres', 'imdb_score', 'release_year', 'score']]

# Step 4: Generate the final response with the recommended movies using RAG
def generate_response(user_input, collection, top_k=5):
    """
    Generate a response using RAG (Retrieval-Augmented Generation) with the top-k relevant movie context.
    """
    context = get_relevant_movies(user_input, collection, top_k)
    context_str = "\n".join([f"Movie: {row['title']} | Genres: {row['genres']} | IMDb Score: {row['imdb_score']} | Release Year: {row['release_year']}\nDescription: {row['description']}" for _, row in context.iterrows()])
    final_prompt = f"User Query: {user_input}\n\nHere are some movie recommendations based on your request:\n{context_str}\n\nProvide me with the best recommendation."

    model_name = "google/gemma-2-2b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    inputs = tokenizer(final_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(inputs['input_ids'], max_length=250, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7, do_sample=True)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Step 5: Flask app to handle user input and display recommendations
@app.route("/", methods=["GET", "POST"])
def index():
    df = pd.read_csv('D:/movie recommendation/titles.csv')
    perform_eda(df)
    client, collection = initialize_chroma_db(df)

    if request.method == "POST":
        user_input = request.form["user_input"]
        response = generate_response(user_input, collection)
        return render_template("index.html", response=response)

    return render_template("index.html", response="")

if __name__ == "__main__":
    app.run(debug=True)
