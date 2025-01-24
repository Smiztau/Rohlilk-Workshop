from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
import csv

# Download necessary NLTK datasets
nltk.download('punkt')
nltk.download('stopwords')

# Load your corpus (the .txt file)
with open("corpus.txt", "r", encoding="utf-8") as file:
    text = file.read()

text = text.lower()
text = ''.join([char if char.isalpha() or char == ' ' or char == '\n' else ' ' for char in text])

items = [
    "Pastry", "Herb", "Beet", "Chicken", "Chicory", "Donut", "Salmon", "Physalis", "Grape", "Apple", 
    "Cabbage", "Pumpkin", "Mango", "Pepper", "Bread", "Toust", "Bell Pepper", "Tortilla", "Tomato", "Waffle", 
    "Celery", "Plum", "Beef", "Nectarine", "Roll", "Blueberry", "Carrot", "Pear", "Croissant", "Date", "Biscuit", 
    "Salad", "Potato", "Cucumber", "Sweet Potato", "Orange", "Pea", "Onion", "Turkey", "Avocado", "Tulip", "Corn", 
    "Pineapple", "Spinach", "Eggplant", "Lettuce", "Strawberry", "Mandarin", "Parsley", "Pork", "Pappudia", "Basil", 
    "Raspberry", "Baguette", "Shallot", "Melon", "Peach", "Pretzel", "Passion fruit", "Chili", "Sandwich", "Pomegranate", 
    "Wrap", "Arugula", "Radish", "Muffin", "Tangerine", "Grapefruit", "Kohlrabi", "Zucchini", "Asparagus", "Leek", "Cone", 
    "Mushroom", "Lime", "Bagel", "Cake", "Carp", "Apricot", "Breadcrumbs", "Banana", "Scallion", "Watermelon", "Radicchio", 
    "Vegetable", "Rosemary", "Garlic", "Kaki", "Kiwi", "Broccoli", "Cactus Fruit", "Nut", "Breadcrumb", "Green Bean", "Cherry", 
    "Cantaloupe", "Clementine", "Snack", "Coriander", "Parsnip", "Mint", "Bun", "Cauliflower", "Berry", "Bean", "Thyme", 
    "Drink", "Lemon", "Flatbread", "Decoration", "Shrimp", "Cracker", "Pomelo", "Paprika", "Brioche", "Cookie", "Mix meat", 
    "Ginger", "Fennel", "Pita", "Brussels sprout", "Fig", "Blackberry", "Pak choi", "Plantain", "Chive", "Bag", "Persimmon", 
    "Chestnut", "Cornmeal", "Flower", "Squash", "Chrysanthemum", "Currant", "Juice", "Milling", "Sprout", "Kale", "Easter decoration", 
    "Soup", "Cream", "Yogurt", "Duck", "Plant meat", "Litchi", "Granadilla", "Cheese", "Plant", "Cactus", "Cereal", "Gooseberry", 
    "Rice Cake", "Pepperoni", "Focaccia", "Pizza", "Hot Dog", "Rambutan", "Burger", "Grain", "Panini", "Mangosteen", "Satsuma", 
    "Bakery", "Pasta", "Chard", "Parsley Root", "Cress", "Celeriac", "Dessert", "Soil", "Endive", "Lambs lettuce", "Soybean sprout", 
    "Lucki", "Surimi"
]


items_dict_plural = {}

# Step 3: Loop through each item in the list
for item in items:
    # Clean the item: lowercase and remove spaces or underscores
    cleaned_item = item.lower().replace(" ", "").replace("_", "")
    
    # Step 4: Convert the cleaned item to its plural form
    # Basic pluralization logic (you can use libraries for better pluralization handling)
    # Handle some basic cases like "chicken" to "chickens"
    if cleaned_item[-1] == 'y':
        plural_item = cleaned_item[:-1] + 'ies'  # Change "cherry" to "cherries"
    elif cleaned_item.endswith('s') or cleaned_item in ['breadcrumb', 'bread', 'fruit', 'scarf', 'mangosteen', 'soybeansprout']:
        plural_item = cleaned_item  # If it already ends in 's', keep it (e.g. "scarf" stays "scarfs")
    else:
        plural_item = cleaned_item + 's'  # Add 's' for regular plural forms
    
    # Step 5: Add the plural form as the key and the singular form as the value in the dictionary
    items_dict_plural[plural_item] = cleaned_item

for plural, singular in items_dict_plural.items():
        # Replace plural with singular, ensuring correct case
        text = text.replace(plural, singular)

# Tokenize the text into sentences
sentences = text.splitlines()

# Preprocess each sentence: lowercasing, removing punctuation, stopwords
def preprocess_sentence(sentence):
    tokens = word_tokenize(sentence.lower())  # Tokenize and lowercase
    tokens = [word for word in tokens if word.isalpha()]  # Remove non-alphabetic characters
    stop_words = set(stopwords.words('english'))  # Set of English stopwords
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    return tokens

# Preprocess all sentences
print("start preprocessing")
processed_sentences = [preprocess_sentence(sentence) for sentence in sentences]

# Train a word2vec model
print("start word to vec")
model = Word2Vec(processed_sentences, vector_size=2, window=5, min_count=1, sg=0)


# Create dictionary with keys and values as lowercase letters only, and no spaces or underscores
items_dict = {item: item.lower().replace(" ", "").replace("_", "") for item in items}

items_dict["Mix meat"] = "meat"
items_dict["Mangosteen"] = "mangoteen"
items_dict["Soybean sprout"] = "soybeanprout"
items_dict["Toust"] = "toast"
data_to_write = []

for key, value in items_dict.items():
    try:
        # Get the word embedding for the value (which is the normalized food name)
        embedding = model.wv[value]
        
        # Take the first 2 components of the embedding
        # You can change this depending on the size of the embeddings (default 100 dimensions)
        embedding_first_2 = embedding[:2]
        
        # Prepare row data (original name, first 2 components of the embedding)
        data_to_write.append([key, embedding_first_2[0], embedding_first_2[1]])
    except KeyError:
        print(f"Embedding for '{value}' not found in the model!")

# Write data to CSV
output_csv = 'csv_junk/food_embeddings.csv'

with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    
    # Write the header row
    writer.writerow(['name_only', 'Embedding Component 1', 'Embedding Component 2'])
    
    # Write the data
    writer.writerows(data_to_write)

print(f"Data written to {output_csv}")