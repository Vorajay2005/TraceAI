# Lost & Found AI System

A Flask-based web application that uses AI to match lost and found items using natural language processing and machine learning.

## Features

- **Submit Lost Items**: Users can report lost items with descriptions, locations, dates, and images
- **Upload Found Items**: Users can upload found items with similar details
- **AI-Powered Matching**: Uses TF-IDF vectorization and cosine similarity to match lost and found items
- **Smart Preprocessing**: Includes lemmatization and synonym replacement for better matching
- **Image Support**: Upload and display images for both lost and found items
- **Responsive Design**: Modern, mobile-friendly interface

## Technology Stack

- **Backend**: Flask (Python)
- **Database**: SQLite
- **AI/ML**: scikit-learn, NLTK
- **Frontend**: HTML, CSS, JavaScript
- **Image Processing**: Werkzeug file handling

## Installation

1. Clone or download the project
2. Navigate to the project directory:

   ```bash
   cd lost_found_ai
   ```

3. Activate the virtual environment:

   ```bash
   source venv/bin/activate
   ```

4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Run the application:

   ```bash
   python app.py
   ```

6. Open your browser and go to: `http://localhost:5000`

## Usage

1. **Home Page**: Navigate between different sections
2. **Submit Lost Item**: Fill out the form with item details and optional image
3. **Upload Found Item**: Similar form for found items
4. **View Items**: Browse all lost and found items
5. **AI Matching**: Click "Find Matches" to run the AI algorithm
6. **View Results**: See matched items with similarity scores

## AI Matching Algorithm

The system uses:

- **Text Preprocessing**: Lemmatization and synonym replacement
- **TF-IDF Vectorization**: Converts text to numerical vectors
- **Cosine Similarity**: Measures similarity between item descriptions
- **Smart Thresholding**: Filters matches above 10% similarity

## File Structure

```
lost_found_ai/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── lost_found.db         # SQLite database (created automatically)
├── templates/            # HTML templates
│   ├── index.html
│   ├── submit_lost.html
│   ├── upload_found.html
│   ├── lost_items.html
│   ├── found_items.html
│   └── results.html
├── static/
│   ├── css/
│   │   └── style.css     # Styling
│   └── uploads/          # Uploaded images
└── venv/                 # Virtual environment
```

## Contributing

Feel free to fork this project and submit pull requests for improvements!

## License

This project is open source and available under the MIT License.
