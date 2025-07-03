from flask import Flask, request, render_template, jsonify
import os
from werkzeug.utils import secure_filename
from src.predictor import EWastePredictor
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize predictor
model_path = os.path.join("models", "e_waste_model.h5")
predictor = None

if os.path.exists(model_path):
    predictor = EWastePredictor(model_path)
else:
    print("⚠️ Model not found. Please train the model first.")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>E-Waste Type Classifier</title>
        <style>
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                max-width: 900px; 
                margin: 0 auto; 
                padding: 20px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container {
                background: white;
                border-radius: 15px;
                padding: 30px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            }
            h1 { 
                color: #333; 
                text-align: center; 
                margin-bottom: 10px;
                font-size: 2.5em;
            }
            .subtitle {
                text-align: center;
                color: #666;
                margin-bottom: 30px;
                font-size: 1.1em;
            }
            .upload-area { 
                border: 3px dashed #4CAF50; 
                padding: 50px; 
                text-align: center; 
                margin: 30px 0; 
                border-radius: 10px;
                background: #f9f9f9;
                transition: all 0.3s ease;
            }
            .upload-area:hover {
                background: #f0f8f0;
                border-color: #45a049;
            }
            .categories {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
                gap: 15px;
                margin: 30px 0;
            }
            .category {
                text-align: center;
                padding: 15px;
                background: #f8f9fa;
                border-radius: 8px;
                border: 1px solid #e9ecef;
            }
            .category-icon {
                font-size: 2em;
                margin-bottom: 5px;
            }
            button { 
                background: linear-gradient(45deg, #4CAF50, #45a049);
                color: white; 
                padding: 15px 30px; 
                border: none; 
                border-radius: 25px; 
                cursor: pointer; 
                font-size: 16px;
                font-weight: bold;
                transition: transform 0.2s ease;
            }
            button:hover { 
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(76, 175, 80, 0.3);
            }
            input[type="file"] {
                margin: 20px 0;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
                width: 100%;
                max-width: 300px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔋 E-Waste Type Classifier</h1>
            <p class="subtitle">Upload an image to identify the type of electronic waste</p>
            
            <div class="categories">
                <div class="category">
                    <div class="category-icon">⌨️</div>
                    <div>Keyboards</div>
                </div>
                <div class="category">
                    <div class="category-icon">🖱️</div>
                    <div>Mouse</div>
                </div>
                <div class="category">
                    <div class="category-icon">🔋</div>
                    <div>Battery</div>
                </div>
                <div class="category">
                    <div class="category-icon">📱</div>
                    <div>Mobiles</div>
                </div>
                <div class="category">
                    <div class="category-icon">🔌</div>
                    <div>PCB</div>
                </div>
                <div class="category">
                    <div class="category-icon">📺</div>
                    <div>Microwave</div>
                </div>
            </div>
            
            <form action="/predict" method="post" enctype="multipart/form-data">
                <div class="upload-area">
                    <p>📸 Choose an image file</p>
                    <input type="file" name="file" accept="image/*" required>
                    <br><br>
                    <button type="submit">🔍 Classify E-Waste Type</button>
                </div>
            </form>
        </div>
    </body>
    </html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    if predictor is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    # Save uploaded file
    filename = secure_filename(str(uuid.uuid4()) + '_' + file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        # Make prediction
        result = predictor.predict_image(filepath)
        
        # Clean up uploaded file
        os.remove(filepath)
        
        if 'error' in result:
            return jsonify(result), 500
        
        # Return HTML response
        ewaste_type = result['ewaste_type']
        confidence = result['confidence']
        top_3 = result['top_3_predictions']
        
        # Category icons mapping
        category_icons = {
            'keyboards': '⌨️',
            'mouse': '🖱️',
            'battery': '🔋',
            'mobiles': '📱',
            'pcb': '🔌',
            'microwave': '📺'
        }
        
        top_3_html = ""
        for i, pred in enumerate(top_3, 1):
            icon = category_icons.get(pred['class'], '📦')
            top_3_html += f"<div class='prediction-item'>{icon} {pred['class'].title()}: {pred['confidence']:.1%}</div>"
        
        return f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>E-Waste Classifier - Result</title>
            <style>
                body {{ 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                    max-width: 900px; 
                    margin: 0 auto; 
                    padding: 20px; 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                }}
                .container {{
                    background: white;
                    border-radius: 15px;
                    padding: 30px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                }}
                .result {{ 
                    margin: 30px 0; 
                    padding: 25px; 
                    border-radius: 15px; 
                    background: linear-gradient(45deg, #4CAF50, #45a049);
                    color: white;
                    text-align: center;
                }}
                .main-prediction {{
                    font-size: 2.5em;
                    margin: 15px 0;
                }}
                .confidence {{
                    font-size: 1.3em;
                    margin: 10px 0;
                    opacity: 0.9;
                }}
                .top-predictions {{
                    background: #f8f9fa;
                    padding: 20px;
                    border-radius: 10px;
                    margin: 20px 0;
                }}
                .prediction-item {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 10px;
                    margin: 5px 0;
                    background: white;
                    border-radius: 8px;
                    border-left: 4px solid #4CAF50;
                }}
                .back-btn {{ 
                    background: linear-gradient(45deg, #2196F3, #21CBF3);
                    color: white; 
                    padding: 15px 30px; 
                    text-decoration: none; 
                    border-radius: 25px;
                    display: inline-block;
                    font-weight: bold;
                    transition: transform 0.2s ease;
                }}
                .back-btn:hover {{
                    transform: translateY(-2px);
                    box-shadow: 0 5px 15px rgba(33, 150, 243, 0.3);
                }}
                h1 {{ color: #333; text-align: center; }}
                h2 {{ color: #333; margin-bottom: 15px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🔋 E-Waste Type Classifier - Result</h1>
                <div class="result">
                    <div class="main-prediction">
                        {category_icons.get(ewaste_type, '📦')} {ewaste_type.title()}
                    </div>
                    <div class="confidence">Confidence: {confidence:.1%}</div>
                </div>
                
                <div class="top-predictions">
                    <h2>📊 Top 3 Predictions:</h2>
                    {top_3_html}
                </div>
                
                <div style="text-align: center;">
                    <a href="/" class="back-btn">← Classify Another Image</a>
                </div>
            </div>
        </body>
        </html>
        '''
        
    except Exception as e:
        # Clean up uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)