from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
import sqlite3
import os
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import WordNetLemmatizer
from datetime import datetime

# Download required NLTK data
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
DATABASE = 'lost_found.db'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

synonyms = {
    "bag": ["backpack", "sack", "pack", "handbag"],
    "wallet": ["purse", "cardholder", "billfold"],
    "books": ["notebooks", "journals", "textbooks"],
    "id": ["identification", "identity card", "license"],
    "laptop": ["computer", "notebook", "pc", "macbook"],
    "phone": ["mobile", "cellphone", "smartphone", "iphone"],
    "keys": ["keyring", "car keys"],
    "glasses": ["spectacles", "eyewear", "sunglasses"],
    "watch": ["timepiece", "smartwatch"],
    "umbrella": ["parasol", "rain cover"],
    "jacket": ["coat", "sweater", "hoodie"]
}

lemmatizer = WordNetLemmatizer()

def init_db():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS lost_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            description TEXT NOT NULL,
            location TEXT NOT NULL,
            date_time TEXT NOT NULL,
            contact_info TEXT,
            image_path TEXT,
            timestamp TEXT NOT NULL
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS found_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            description TEXT NOT NULL,
            location TEXT NOT NULL,
            date_time TEXT NOT NULL,
            contact_info TEXT,
            image_path TEXT,
            timestamp TEXT NOT NULL
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS matches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lost_item_id INTEGER,
            found_item_id INTEGER,
            similarity_score REAL,
            timestamp TEXT NOT NULL
        )
    ''')
    
    conn.commit()
    conn.close()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_description(description):
    if not description:
        return ""
    words = description.lower().split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    replaced_words = []
    for word in lemmatized_words:
        if word in synonyms:
            replaced_words.extend(synonyms[word])
        else:
            replaced_words.append(word)
    return " ".join(replaced_words)

def save_image(file):
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_")
        filename = timestamp + filename
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        return file_path
    return None

init_db()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit_item', methods=['GET', 'POST'])
def submit_item():
    if request.method == 'POST':
        description = request.form.get('description')
        location = request.form.get('location')
        date_time = request.form.get('date_time')
        contact_info = request.form.get('contact_info', '')
        
        if not description or not location or not date_time:
            flash('Please fill in all required fields.', 'error')
            return render_template('submit_lost.html')
        
        image_path = None
        if 'image' in request.files:
            file = request.files['image']
            if file.filename != '':
                image_path = save_image(file)
        
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO lost_items (description, location, date_time, contact_info, image_path, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (description, location, date_time, contact_info, image_path, str(datetime.now())))
        conn.commit()
        conn.close()
        
        flash('Lost item submitted successfully!', 'success')
        return redirect(url_for('index'))
    
    return render_template('submit_lost.html')

@app.route('/upload_found', methods=['GET', 'POST'])
def upload_found():
    if request.method == 'POST':
        description = request.form.get('description')
        location = request.form.get('location')
        date_time = request.form.get('date_time')
        contact_info = request.form.get('contact_info', '')
        
        if not description or not location or not date_time:
            flash('Please fill in all required fields.', 'error')
            return render_template('upload_found.html')
        
        image_path = None
        if 'image' in request.files:
            file = request.files['image']
            if file.filename != '':
                image_path = save_image(file)
        
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO found_items (description, location, date_time, contact_info, image_path, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (description, location, date_time, contact_info, image_path, str(datetime.now())))
        conn.commit()
        conn.close()
        
        flash('Found item uploaded successfully!', 'success')
        return redirect(url_for('index'))
    
    return render_template('upload_found.html')

@app.route('/view_lost_items')
def view_lost_items():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM lost_items ORDER BY timestamp DESC')
    lost_items = cursor.fetchall()
    conn.close()
    
    lost_items_list = []
    for item in lost_items:
        lost_items_list.append({
            'id': item[0], 'description': item[1], 'location': item[2],
            'date_time': item[3], 'contact_info': item[4], 'image_path': item[5],
            'timestamp': item[6]
        })
    
    return render_template('lost_items.html', lost_items=lost_items_list)

@app.route('/view_found_items')
def view_found_items():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM found_items ORDER BY timestamp DESC')
    found_items = cursor.fetchall()
    conn.close()
    
    found_items_list = []
    for item in found_items:
        found_items_list.append({
            'id': item[0], 'description': item[1], 'location': item[2],
            'date_time': item[3], 'contact_info': item[4], 'image_path': item[5],
            'timestamp': item[6]
        })
    
    return render_template('found_items.html', found_items=found_items_list)

@app.route('/match_items')
def match_items():
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM lost_items')
        lost_items = cursor.fetchall()
        cursor.execute('SELECT * FROM found_items')
        found_items = cursor.fetchall()
        
        if not lost_items or not found_items:
            flash('No items available for matching.', 'info')
            return redirect(url_for('get_matches'))
        
        cursor.execute('DELETE FROM matches')
        
        lost_descriptions = [preprocess_description(item[1]) for item in lost_items]
        found_descriptions = [preprocess_description(item[1]) for item in found_items]
        
        vectorizer = TfidfVectorizer(ngram_range=(1, 3), stop_words='english')
        all_descriptions = lost_descriptions + found_descriptions
        
        if len(all_descriptions) < 2:
            flash('Not enough items for matching.', 'info')
            return redirect(url_for('get_matches'))
        
        tfidf_matrix = vectorizer.fit_transform(all_descriptions)
        
        matches_found = 0
        for i, lost_item in enumerate(lost_items):
            lost_vector = tfidf_matrix[i]
            found_vectors = tfidf_matrix[len(lost_items):]
            similarities = cosine_similarity(lost_vector, found_vectors).flatten()
            
            threshold = 0.1
            for j, similarity in enumerate(similarities):
                if similarity > threshold:
                    cursor.execute('''
                        INSERT INTO matches (lost_item_id, found_item_id, similarity_score, timestamp)
                        VALUES (?, ?, ?, ?)
                    ''', (lost_item[0], found_items[j][0], float(similarity), str(datetime.now())))
                    matches_found += 1
        
        conn.commit()
        conn.close()
        
        if matches_found > 0:
            flash(f'Found {matches_found} potential matches!', 'success')
        else:
            flash('No matches found.', 'info')
        
        return redirect(url_for('get_matches'))
        
    except Exception as e:
        print(f"Error: {str(e)}")
        flash('Error during matching.', 'error')
        return redirect(url_for('index'))

@app.route('/get_matches')
def get_matches():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT 
            m.similarity_score, m.timestamp,
            l.id, l.description, l.location, l.date_time, l.contact_info, l.image_path,
            f.id, f.description, f.location, f.date_time, f.contact_info, f.image_path
        FROM matches m
        JOIN lost_items l ON m.lost_item_id = l.id
        JOIN found_items f ON m.found_item_id = f.id
        ORDER BY m.similarity_score DESC
    ''')
    
    matches = cursor.fetchall()
    conn.close()
    
    matches_list = []
    for match in matches:
        matches_list.append({
            'similarity_score': round(match[0] * 100, 2),
            'timestamp': match[1],
            'lost_item': {
                'id': match[2], 'description': match[3], 'location': match[4],
                'date_time': match[5], 'contact_info': match[6], 'image_path': match[7]
            },
            'found_item': {
                'id': match[8], 'description': match[9], 'location': match[10],
                'date_time': match[11], 'contact_info': match[12], 'image_path': match[13]
            }
        })
    
    return render_template('results.html', matches=matches_list)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
