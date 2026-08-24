import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
import json
import os
from datetime import datetime
import time

# ============== CONFIGURATION ==============
MODEL_PATH = "best_resnet50_food_subset.pth"
CSV_PATH = "food_metadata.csv"
NEW_FOOD_CSV = "new_food_data.csv"  # Separate file for user-contributed foods
HEALTHY_FOOD_CSV = "1763360300482_healthy_food_dataset_appended.csv"  # Healthy alternatives dataset
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============== DIET PLAN DEFINITIONS ==============
DIET_PLANS = {
    "Weight Loss": {
        "description": "Calorie-deficit diet focused on lean proteins, vegetables, and portion control",
        "daily_calories": "1500-1800 kcal",
        "macros": {"protein": "30%", "carbs": "40%", "fat": "30%"},
        "guidelines": [
            "Focus on high-protein, low-calorie foods",
            "Limit refined carbs and sugary foods",
            "Eat plenty of vegetables and lean proteins",
            "Stay hydrated with 8-10 glasses of water daily",
            "Avoid processed and fried foods"
        ],
        "suitable_foods": ["Grilled chicken", "Fish", "Vegetables", "Salads", "Fruits (moderate)"],
        "avoid_foods": ["Fried foods", "Sugary desserts", "White bread", "Processed snacks", "High-calorie drinks"],
        "meal_combos": [
            {"name": "Grilled Turkey Breast (100g) + Quinoa Salad", "calories": 355, "protein": 42.0, "carbs": 37.5, "fat": 15.5},
            {"name": "Steamed White Fish (100g) + Mixed Green Salad", "calories": 150, "protein": 28.5, "carbs": 7.5, "fat": 3.0},
            {"name": "Grilled Chicken Salad + Apple", "calories": 208, "protein": 30.8, "carbs": 21.0, "fat": 4.0},
            {"name": "Lentil Soup + Carrot Sticks", "calories": 252, "protein": 14.4, "carbs": 37.0, "fat": 2.9},
            {"name": "Baked Falafel (3 pcs) + Greek Salad", "calories": 320, "protein": 18.4, "carbs": 57.5, "fat": 18.4}
        ]
    },
    "Muscle Gain": {
        "description": "High-protein, calorie-surplus diet to support muscle growth and recovery",
        "daily_calories": "2500-3000 kcal",
        "macros": {"protein": "35%", "carbs": "45%", "fat": "20%"},
        "guidelines": [
            "Consume 1.6-2.2g protein per kg body weight",
            "Eat complex carbs for sustained energy",
            "Include healthy fats from nuts and avocados",
            "Time meals around workouts",
            "Eat 5-6 smaller meals throughout the day"
        ],
        "suitable_foods": ["Chicken breast", "Eggs", "Rice", "Oats", "Fish", "Greek yogurt", "Nuts"],
        "avoid_foods": ["Empty calories", "Excessive sugar", "Trans fats", "Alcohol"],
        "meal_combos": [
            {"name": "Grilled Chicken Salad + Overnight Oats (with milk)", "calories": 440, "protein": 54.0, "carbs": 48.0, "fat": 13.5},
            {"name": "Grilled Salmon (100g) + Brown Rice & Veg Bowl", "calories": 478, "protein": 37.7, "carbs": 37.5, "fat": 18.4},
            {"name": "Grilled Turkey Breast (100g) + Sushi - Salmon Roll (6 pcs)", "calories": 375, "protein": 45.0, "carbs": 37.5, "fat": 21.5},
            {"name": "Greek Yogurt Parfait + Almonds (10 pcs) + Banana", "calories": 435, "protein": 30.9, "carbs": 35.6, "fat": 16.9},
            {"name": "Sprouted Moong Salad + Whole Wheat Pasta", "calories": 450, "protein": 22.0, "carbs": 95.0, "fat": 19.0}
        ]
    },
    "Keto/Low Carb": {
        "description": "Very low-carb, high-fat diet that promotes ketosis for fat burning",
        "daily_calories": "1800-2200 kcal",
        "macros": {"protein": "25%", "carbs": "5%", "fat": "70%"},
        "guidelines": [
            "Keep net carbs under 20-50g per day",
            "Focus on healthy fats (avocado, olive oil, nuts)",
            "Moderate protein intake",
            "Avoid all grains, sugar, and starchy vegetables",
            "Stay hydrated and supplement electrolytes"
        ],
        "suitable_foods": ["Avocado", "Eggs", "Cheese", "Fatty fish", "Nuts", "Leafy greens", "Butter"],
        "avoid_foods": ["Bread", "Rice", "Pasta", "Potatoes", "Sugar", "Most fruits", "Beans"],
        "meal_combos": [
            {"name": "Grilled Salmon (100g) + Avocado Toast (no bread, just avocado)", "calories": 333, "protein": 29.7, "carbs": 12.5, "fat": 19.9},
            {"name": "Grilled Salmon (100g) + Spinach + Walnuts (7 halves)", "calories": 397, "protein": 32.1, "carbs": 5.8, "fat": 21.9},
            {"name": "Steamed White Fish + Broccoli + Almonds (10 pcs)", "calories": 339, "protein": 29.4, "carbs": 12.4, "fat": 16.3},
            {"name": "Paneer Tikka (grilled) + Mixed Green Salad", "calories": 200, "protein": 13.0, "carbs": 28.7, "fat": 11.7},
            {"name": "Grilled Chicken Salad + Cottage Cheese (low-fat)", "calories": 228, "protein": 42.0, "carbs": 4.0, "fat": 4.0}
        ]
    },
    "Mediterranean": {
        "description": "Heart-healthy diet rich in olive oil, fish, vegetables, and whole grains",
        "daily_calories": "2000-2400 kcal",
        "macros": {"protein": "20%", "carbs": "50%", "fat": "30%"},
        "guidelines": [
            "Use extra virgin olive oil as primary fat source",
            "Eat fish and seafood at least twice weekly",
            "Include plenty of fruits, vegetables, and whole grains",
            "Enjoy moderate amounts of dairy (especially yogurt and cheese)",
            "Limit red meat to occasional consumption"
        ],
        "suitable_foods": ["Fish", "Olive oil", "Vegetables", "Whole grains", "Legumes", "Nuts", "Fruits"],
        "avoid_foods": ["Processed meats", "Refined grains", "Trans fats", "Sugary beverages"],
        "meal_combos": [
            {"name": "Grilled Salmon (100g) + Quinoa Salad + Greek Salad", "calories": 548, "protein": 49.7, "carbs": 75.0, "fat": 30.4},
            {"name": "Hummus with Carrot Sticks + Whole Wheat Pasta with Tomato Sauce", "calories": 420, "protein": 10.8, "carbs": 60.0, "fat": 7.8},
            {"name": "Chickpea Salad + Tomato + Orange", "calories": 293, "protein": 15.1, "carbs": 62.7, "fat": 12.9},
            {"name": "Tuna Salad (light mayo) + Lentil Soup", "calories": 440, "protein": 26.2, "carbs": 67.5, "fat": 20.5},
            {"name": "Greek Yogurt Parfait + Mixed Berry Bowl + Walnuts", "calories": 413, "protein": 25.2, "carbs": 25.5, "fat": 16.9}
        ]
    },
    "Vegan": {
        "description": "Plant-based diet excluding all animal products",
        "daily_calories": "1800-2200 kcal",
        "macros": {"protein": "20%", "carbs": "55%", "fat": "25%"},
        "guidelines": [
            "Focus on whole plant foods",
            "Ensure adequate protein from legumes, tofu, tempeh",
            "Supplement B12, Vitamin D, and potentially iron",
            "Include omega-3 sources (flax, chia, walnuts)",
            "Eat a variety of colorful vegetables and fruits"
        ],
        "suitable_foods": ["Legumes", "Tofu", "Tempeh", "Quinoa", "Vegetables", "Fruits", "Nuts", "Seeds"],
        "avoid_foods": ["All meat", "Dairy", "Eggs", "Honey", "Gelatin", "Animal-derived ingredients"],
        "meal_combos": [
            {"name": "Tofu Stir-fry + Brown Rice & Veg Bowl", "calories": 443, "protein": 27.0, "carbs": 75.0, "fat": 30.0},
            {"name": "Lentil Soup + Quinoa Salad + Apple", "calories": 498, "protein": 23.2, "carbs": 88.5, "fat": 14.5},
            {"name": "Chickpea Salad + Baked Sweet Potato + Smoothie (Green)", "calories": 619, "protein": 14.5, "carbs": 95.1, "fat": 12.9},
            {"name": "Moong Dal + Idli (2 pcs) + Banana", "calories": 407, "protein": 18.3, "carbs": 76.4, "fat": 10.4},
            {"name": "Veggie Wrap (whole wheat) + Sambar + Orange", "calories": 441, "protein": 23.9, "carbs": 85.7, "fat": 14.9}
        ]
    },
    "Diabetic-Friendly": {
        "description": "Blood sugar control diet with low glycemic index foods",
        "daily_calories": "1800-2200 kcal",
        "macros": {"protein": "25%", "carbs": "45%", "fat": "30%"},
        "guidelines": [
            "Choose low glycemic index foods",
            "Control portion sizes, especially carbs",
            "Eat regular meals at consistent times",
            "Include fiber-rich foods",
            "Monitor blood sugar levels regularly"
        ],
        "suitable_foods": ["Non-starchy vegetables", "Whole grains", "Lean proteins", "Legumes", "Nuts"],
        "avoid_foods": ["Sugary foods", "White bread", "Sugary drinks", "Processed snacks", "High-sugar fruits"],
        "meal_combos": [
            {"name": "Grilled Turkey Breast + Lentil Soup + Broccoli", "calories": 486, "protein": 44.6, "carbs": 38.2, "fat": 6.4},
            {"name": "Steamed White Fish + Quinoa Salad + Spinach", "calories": 347, "protein": 36.0, "carbs": 39.1, "fat": 13.6},
            {"name": "Moong Dal + Oatmeal (rolled oats, water) + Carrot", "calories": 362, "protein": 20.2, "carbs": 91.0, "fat": 6.4},
            {"name": "Sprouted Moong Salad + Greek Yogurt Parfait", "calories": 310, "protein": 40.0, "carbs": 58.0, "fat": 17.0},
            {"name": "Grilled Chicken Salad + Black Bean Salad", "calories": 355, "protein": 42.0, "carbs": 37.5, "fat": 15.5}
        ]
    },
    "Heart-Healthy": {
        "description": "Low-sodium, low-saturated fat diet to support cardiovascular health",
        "daily_calories": "2000-2300 kcal",
        "macros": {"protein": "25%", "carbs": "50%", "fat": "25%"},
        "guidelines": [
            "Limit sodium to under 2000mg per day",
            "Choose unsaturated fats over saturated fats",
            "Eat plenty of fruits and vegetables (5-9 servings)",
            "Include omega-3 rich fish twice weekly",
            "Avoid trans fats completely"
        ],
        "suitable_foods": ["Salmon", "Oatmeal", "Berries", "Leafy greens", "Nuts", "Whole grains"],
        "avoid_foods": ["Fried foods", "Processed meats", "High-sodium foods", "Trans fats", "Excessive red meat"],
        "meal_combos": [
            {"name": "Grilled Salmon (100g) + Oatmeal + Mixed Berry Bowl", "calories": 429, "protein": 27.1, "carbs": 78.5, "fat": 7.2},
            {"name": "Steamed White Fish + Brown Rice & Veg Bowl + Apple", "calories": 468, "protein": 36.0, "carbs": 67.5, "fat": 13.5},
            {"name": "Vegetable Soup + Whole Wheat Pasta + Grapes", "calories": 476, "protein": 17.8, "carbs": 92.5, "fat": 6.3},
            {"name": "Kale Caesar (light dressing) + Banana + Walnuts", "calories": 469, "protein": 18.9, "carbs": 69.1, "fat": 27.8},
            {"name": "Miso Soup + Buckwheat Pancake + Strawberry", "calories": 268, "protein": 12.8, "carbs": 46.3, "fat": 6.9}
        ]
    },
    "High Protein": {
        "description": "Protein-rich diet for satiety, muscle maintenance, and metabolic health",
        "daily_calories": "2000-2400 kcal",
        "macros": {"protein": "40%", "carbs": "30%", "fat": "30%"},
        "guidelines": [
            "Include protein source at every meal",
            "Aim for 1.8-2.5g protein per kg body weight",
            "Balance with vegetables and healthy fats",
            "Stay well hydrated (increased water needs)",
            "Choose lean protein sources"
        ],
        "suitable_foods": ["Chicken", "Fish", "Eggs", "Greek yogurt", "Cottage cheese", "Lean beef", "Tofu"],
        "avoid_foods": ["Empty carbs", "Excessive sugar", "Processed foods"],
        "meal_combos": [
            {"name": "Grilled Chicken Salad + Greek Yogurt Parfait + Cottage Cheese", "calories": 388, "protein": 66.0, "carbs": 12.0, "fat": 5.5},
            {"name": "Grilled Salmon (100g) + Grilled Turkey Breast + Spinach", "calories": 350, "protein": 56.5, "carbs": 1.6, "fat": 10.0},
            {"name": "Steamed White Fish + Tuna Salad + Broccoli", "calories": 411, "protein": 39.0, "carbs": 45.7, "fa0t": 19.9},
            {"name": "Sprouted Moong Salad + Greek Yogurt Parfait + Almonds", "calories": 478, "protein": 45.6, "carbs": 62.2, "fat": 18.4},
            {"name": "Grilled Turkey Breast + Cottage Cheese + Chickpea Salad", "calories": 443, "protein": 54.0, "carbs": 41.5, "fat": 12.5}
        ]
    },
    "Balanced/General Health": {
        "description": "Well-rounded diet with all food groups in moderation",
        "daily_calories": "2000-2400 kcal",
        "macros": {"protein": "25%", "carbs": "50%", "fat": "25%"},
        "guidelines": [
            "Eat a variety of foods from all food groups",
            "Practice portion control",
            "Include 5+ servings of fruits and vegetables daily",
            "Choose whole grains over refined grains",
            "Limit processed foods and added sugars"
        ],
        "suitable_foods": ["Variety of vegetables", "Fruits", "Whole grains", "Lean proteins", "Dairy", "Healthy fats"],
        "avoid_foods": ["Excessive processed foods", "Trans fats", "Added sugars", "Excessive sodium"],
        "meal_combos": [
            {"name": "Grilled Chicken Salad + Brown Rice & Veg Bowl + Apple", "calories": 478, "protein": 42.0, "carbs": 58.5, "fat": 15.5},
            {"name": "Grilled Salmon (100g) + Quinoa Salad + Orange", "calories": 489, "protein": 38.7, "carbs": 55.7, "fat": 12.8},
            {"name": "Greek Yogurt Parfait + Overnight Oats + Banana + Walnuts", "calories": 629, "protein": 37.9, "carbs": 75.6, "fat": 27.7},
            {"name": "Lentil Soup + Whole Wheat Pasta + Mixed Berry Bowl", "calories": 571, "protein": 17.8, "carbs": 107.5, "fat": 6.3},
            {"name": "Tofu Stir-fry + Buckwheat Pancake + Grapes", "calories": 447, "protein": 21.0, "carbs": 73.8, "fat": 24.4}
        ]
    }
}


# ============== MODEL LOADING ==============
class FoodDetectionModel:
    def __init__(self, model_path):
        self.device = DEVICE
        self.model = None
        self.classes = []
        self.metadata_df = None
        self.new_food_df = None
        self.healthy_food_df = None
        self.load_model(model_path)
        
    def load_model(self, model_path):
        """Load the trained PyTorch model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.classes = checkpoint.get('classes', [])
        num_classes = len(self.classes)
        
        # Build model architecture
        self.model = models.resnet50(pretrained=False)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully with {num_classes} classes")
    
    def load_metadata(self, csv_path):
        """Load food metadata CSV"""
        if not os.path.exists(csv_path):
            print(f"Warning: Metadata CSV not found at {csv_path}")
            return False
        
        self.metadata_df = pd.read_csv(csv_path)
        print(f"Original metadata loaded: {len(self.metadata_df)} food items")
        
        # Load new food data if exists
        if os.path.exists(NEW_FOOD_CSV):
            self.new_food_df = pd.read_csv(NEW_FOOD_CSV)
            print(f"User-contributed food data loaded: {len(self.new_food_df)} items")
        else:
            self.new_food_df = pd.DataFrame()
        
        # Load healthy food dataset
        if os.path.exists(HEALTHY_FOOD_CSV):
            self.healthy_food_df = pd.read_csv(HEALTHY_FOOD_CSV)
            print(f"Healthy food dataset loaded: {len(self.healthy_food_df)} items")
        else:
            print(f"Warning: Healthy food dataset not found at {HEALTHY_FOOD_CSV}")
            self.healthy_food_df = pd.DataFrame()
        
        return True
    
    def match_metadata(self, predicted_class):
        """Match predicted class to metadata with robust strategy"""
        # First check original metadata
        metadata = self._search_in_dataframe(self.metadata_df, predicted_class)
        if metadata:
            metadata['source'] = 'original'
            return metadata
        
        # Then check user-contributed data
        if not self.new_food_df.empty:
            metadata = self._search_in_dataframe(self.new_food_df, predicted_class)
            if metadata:
                metadata['source'] = 'user_contributed'
                return metadata
        
        return None
    
    def _search_in_dataframe(self, df, predicted_class):
        """Search for food in a dataframe"""
        if df is None or df.empty:
            return None
        
        # Strategy 1: Exact match
        exact_match = df[df['Food Item'] == predicted_class]
        if not exact_match.empty:
            return exact_match.iloc[0].to_dict()
        
        # Strategy 2: Case-insensitive match
        case_insensitive = df[df['Food Item'].str.lower() == predicted_class.lower()]
        if not case_insensitive.empty:
            return case_insensitive.iloc[0].to_dict()
        
        # Strategy 3: Normalized match
        normalized_pred = predicted_class.strip().replace('_', ' ')
        normalized_match = df[
            df['Food Item'].str.strip().str.replace('_', ' ').str.lower() 
            == normalized_pred.lower()
        ]
        if not normalized_match.empty:
            return normalized_match.iloc[0].to_dict()
        
        return None
    
    def preprocess_image(self, image):
        """Preprocess image for model input"""
        preprocess = transforms.Compose([
            transforms.Resize(int(IMG_SIZE * 1.1)),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])
        
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        elif isinstance(image, str):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")
        
        return preprocess(image).unsqueeze(0).to(self.device)
    
    def predict(self, image):
        """Predict food class from image"""
        x = self.preprocess_image(image)
        
        with torch.no_grad():
            out = self.model(x)
            probs = torch.nn.functional.softmax(out, dim=1)
            conf, idx = torch.max(probs, 1)
        
        predicted_class = self.classes[idx.item()]
        confidence = conf.item()
        
        return predicted_class, confidence
    
    def predict_top_k(self, image, k=5):
        """Predict top k food classes"""
        x = self.preprocess_image(image)
        
        with torch.no_grad():
            out = self.model(x)
            probs = torch.nn.functional.softmax(out, dim=1)
            top_probs, top_indices = torch.topk(probs, k)
        
        results = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            results.append({
                'class': self.classes[idx.item()],
                'confidence': prob.item()
            })
        
        return results

# ============== GLOBAL MODEL INSTANCE ==============
food_model = None

def initialize_model():
    """Initialize the global model instance"""
    global food_model
    try:
        food_model = FoodDetectionModel(MODEL_PATH)
        food_model.load_metadata(CSV_PATH)
        return "✅ Model initialized successfully!"
    except Exception as e:
        return f"❌ Error initializing model: {str(e)}"

# ============== DIET PLAN FUNCTIONS ==============
def find_healthy_alternatives(predicted_class, metadata, diet_name):
    """Find healthy alternatives from the healthy food dataset"""
    if food_model is None or food_model.healthy_food_df is None or food_model.healthy_food_df.empty:
        return ""
    
    df = food_model.healthy_food_df
    
    if not metadata:
        # If no metadata, suggest based on diet plan only
        alternatives = []
        if diet_name == "Weight Loss":
            alternatives = df[df['Calories (kcal)'] < 200].head(5)
        elif diet_name == "Muscle Gain":
            alternatives = df[df['Protein (g)'] > 15].head(5)
        elif diet_name == "Keto/Low Carb":
            alternatives = df[df['Carbohydrates (g)'] < 10].head(5)
        elif diet_name == "Diabetic-Friendly":
            alternatives = df[df['Sugars (g)'] < 8].head(5)
        elif diet_name == "Heart-Healthy":
            alternatives = df[df['Sodium (mg)'] < 300].head(5)
        elif diet_name == "High Protein":
            alternatives = df[df['Protein (g)'] > 20].head(5)
        elif diet_name == "Vegan":
            vegan_categories = ['Vegetables', 'Fruits', 'Legumes', 'Grains', 'Nuts']
            alternatives = df[df['Category'].isin(vegan_categories)].head(5)
        else:
            alternatives = df.head(5)
    else:
        # Get nutritional values from detected food
        calories = metadata.get('energy_kcal', 0)
        protein = metadata.get('protein_g', 0)
        carbs = metadata.get('carbs_g', 0)
        fat = metadata.get('fat_g', 0)
        sugar = metadata.get('sugar_g', 0)
        
        # Find healthier alternatives based on diet plan criteria
        if diet_name == "Weight Loss":
            # Lower calorie, similar or higher protein
            alternatives = df[
                (df['Calories (kcal)'] < calories * 0.8) & 
                (df['Protein (g)'] >= protein * 0.8)
            ].nsmallest(5, 'Calories (kcal)')
        
        elif diet_name == "Muscle Gain":
            # Higher protein, moderate calories
            alternatives = df[
                (df['Protein (g)'] > protein) | 
                ((df['Protein (g)'] >= protein * 0.9) & (df['Calories (kcal)'] > calories))
            ].nlargest(5, 'Protein (g)')
        
        elif diet_name == "Keto/Low Carb":
            # Much lower carbs, higher fat
            alternatives = df[
                (df['Carbohydrates (g)'] < 10) & 
                (df['Fat (g)'] > 5)
            ].nsmallest(5, 'Carbohydrates (g)')
        
        elif diet_name == "Diabetic-Friendly":
            # Lower sugar, lower carbs
            alternatives = df[
                (df['Sugars (g)'] < sugar * 0.7) & 
                (df['Carbohydrates (g)'] < carbs * 0.8)
            ].nsmallest(5, 'Sugars (g)')
        
        elif diet_name == "Heart-Healthy":
            # Lower sodium, lower cholesterol
            alternatives = df[
                (df['Sodium (mg)'] < 300) & 
                (df['Cholesterol (mg)'] < 50)
            ].nsmallest(5, 'Sodium (mg)')
        
        elif diet_name == "High Protein":
            # Much higher protein
            alternatives = df[df['Protein (g)'] > max(protein, 15)].nlargest(5, 'Protein (g)')
        
        elif diet_name == "Vegan":
            # Plant-based categories
            vegan_categories = ['Vegetables', 'Fruits', 'Legumes', 'Grains', 'Nuts', 'Snack']
            alternatives = df[df['Category'].isin(vegan_categories)].head(5)
        
        else:  # Balanced/General Health
            # Balanced nutrition, moderate calories
            alternatives = df[
                (df['Calories (kcal)'].between(150, 400)) & 
                (df['Protein (g)'] > 5) & 
                (df['Fiber (g)'] > 2)
            ].head(5)
    
    if alternatives.empty:
        return ""
    
    html = "<div style='padding: 20px; background-color: #f0f8ff; border-radius: 10px; margin-top: 20px;'>"
    html += f"<h3 style='color: #1565c0; margin-top: 0;'>🥗 Healthier Alternatives for Your {diet_name} Diet</h3>"
    html += "<p style='color: #424242; font-size: 14px; margin-bottom: 15px;'>Consider these healthier options from our database:</p>"
    
    for idx, row in alternatives.iterrows():
        food_name = row['Food_Item']
        category = row['Category']
        cal = row['Calories (kcal)']
        protein = row['Protein (g)']
        carbs = row['Carbohydrates (g)']
        fat = row['Fat (g)']
        
        html += f"<div style='background-color: springgreen; padding: 12px; margin: 10px 0; border-left: 4px solid #4caf50; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>"
        html += f"<div style='display: flex; justify-content: space-between; align-items: center;'>"
        html += f"<div><strong style='color: #2e7d32; font-size: 16px;'>{food_name}</strong>"
        html += f"<span style='background-color: #e8f5e9; color: #2e7d32; padding: 2px 8px; border-radius: 12px; font-size: 12px; margin-left: 8px;'>{category}</span></div>"
        html += "</div>"
        html += f"<div style='color: #616161; font-size: 13px; margin-top: 8px;'>"
        html += f"<span style='margin-right: 15px;'>🔥 {cal} kcal</span>"
        html += f"<span style='margin-right: 15px;'>💪 {protein}g protein</span>"
        html += f"<span style='margin-right: 15px;'>🍞 {carbs}g carbs</span>"
        html += f"<span>🥑 {fat}g fat</span>"
        html += "</div></div>"
    
    html += "</div>"
    return html

def format_meal_recommendations(diet_name):
    """Format meal combination recommendations for the diet plan"""
    if diet_name not in DIET_PLANS:
        return ""
    
    meal_combos = DIET_PLANS[diet_name].get('meal_combos', [])
    if not meal_combos:
        return ""
    
    emoji_map = {
        "Weight Loss": "🥗",
        "Muscle Gain": "💪",
        "Keto/Low Carb": "🥑",
        "Mediterranean": "🌿",
        "Vegan": "🌱",
        "Diabetic-Friendly": "🩸",
        "Heart-Healthy": "❤️",
        "High Protein": "🥩",
        "Balanced/General Health": "👍"
    }
    
    emoji = emoji_map.get(diet_name, "🍽️")
    
    html = f"<div style='padding: 20px; background-color: #fafafa; border-radius: 10px; margin-top: 20px;'>"
    html += f"<h3 style='color: #6a1b9a; margin-top: 0;'>{emoji} Recommended Meal Combinations</h3>"
    html += "<p style='color: #424242; font-size: 14px; margin-bottom: 15px;'>Try these balanced meal combos optimized for your diet plan:</p>"
    
    for combo in meal_combos:
        html += f"<div style='background-color: orchid; padding: 15px; margin: 12px 0; border-left: 4px solid #9c27b0; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>"
        html += f"<div style='color: #4a148c; font-weight: bold; font-size: 15px; margin-bottom: 8px;'>{combo['name']}</div>"
        html += f"<div style='color: #616161; font-size: 13px;'>"
        html += f"<span style='margin-right: 15px;'>📊 {combo['calories']} kcal</span>"
        html += f"<span style='margin-right: 15px;'>💪 Protein: {combo['protein']}g</span>"
        html += f"<span style='margin-right: 15px;'>🍞 Carbs: {combo['carbs']}g</span>"
        html += f"<span>🥑 Fat: {combo['fat']}g</span>"
        html += "</div></div>"
    
    html += "</div>"
    return html

def get_recommended_foods_from_dataset(diet_name):
    """Get recommended foods from the actual dataset based on diet plan"""
    if food_model is None or food_model.metadata_df is None:
        return []
    
    recommendations = []
    df = food_model.metadata_df
    
    # Also check user-contributed data
    if food_model.new_food_df is not None and not food_model.new_food_df.empty:
        df = pd.concat([df, food_model.new_food_df], ignore_index=True)
    
    if df.empty:
        return []
    
    # Diet-specific filtering logic
    if diet_name == "Weight Loss":
        # Low calorie (<300), high protein (>10g), low sugar (<10g)
        filtered = df[
            (df['energy_kcal'] < 300) & 
            (df['protein_g'] > 10) & 
            (df['sugar_g'] < 10)
        ]
        recommendations = filtered.nsmallest(10, 'energy_kcal')['Food Item'].tolist()
        
    elif diet_name == "Muscle Gain":
        # High protein (>15g), moderate-high calories (>200)
        filtered = df[
            (df['protein_g'] > 15) & 
            (df['energy_kcal'] > 200)
        ]
        recommendations = filtered.nlargest(10, 'protein_g')['Food Item'].tolist()
        
    elif diet_name == "Keto/Low Carb":
        # Very low carbs (<10g), high fat (>10g)
        filtered = df[
            (df['carbs_g'] < 10) & 
            (df['fat_g'] > 10)
        ]
        recommendations = filtered.nsmallest(10, 'carbs_g')['Food Item'].tolist()
        
    elif diet_name == "Mediterranean":
        # Look for fish, vegetables, olive oil-based foods, moderate fat
        keywords = ['fish', 'salmon', 'salad', 'vegetable', 'olive', 'tomato', 'lentil', 'chickpea']
        filtered = df[df['Food Item'].str.lower().str.contains('|'.join(keywords), na=False)]
        recommendations = filtered.head(10)['Food Item'].tolist()
        
    elif diet_name == "Vegan":
        # Plant-based: look for vegetables, legumes, grains
        keywords = ['vegetable', 'salad', 'bean', 'lentil', 'chickpea', 'tofu', 'rice', 'quinoa', 'fruit']
        filtered = df[df['Food Item'].str.lower().str.contains('|'.join(keywords), na=False)]
        # Also exclude obvious animal products
        exclude_keywords = ['chicken', 'fish', 'meat', 'beef', 'pork', 'egg', 'cheese', 'milk', 'yogurt']
        filtered = filtered[~filtered['Food Item'].str.lower().str.contains('|'.join(exclude_keywords), na=False)]
        recommendations = filtered.head(10)['Food Item'].tolist()
        
    elif diet_name == "Diabetic-Friendly":
        # Low sugar (<8g), low-moderate carbs (<40g), high fiber (>3g)
        filtered = df[
            (df['sugar_g'] < 8) & 
            (df['carbs_g'] < 40) & 
            (df['fiber_g'] > 3)
        ]
        recommendations = filtered.nsmallest(10, 'sugar_g')['Food Item'].tolist()
        
    elif diet_name == "Heart-Healthy":
        # Low sodium (<400mg), low cholesterol (<100mg), moderate-low fat
        filtered = df[
            (df['sodium_mg'] < 400) & 
            (df['cholesterol_mg'] < 100)
        ]
        recommendations = filtered.nsmallest(10, 'sodium_mg')['Food Item'].tolist()
        
    elif diet_name == "High Protein":
        # Very high protein (>20g)
        filtered = df[df['protein_g'] > 20]
        recommendations = filtered.nlargest(10, 'protein_g')['Food Item'].tolist()
        
    elif diet_name == "Balanced/General Health":
        # Balanced macros, moderate calories (150-400), good fiber (>2g)
        filtered = df[
            (df['energy_kcal'].between(150, 400)) & 
            (df['fiber_g'] > 2) &
            (df['protein_g'] > 5)
        ]
        recommendations = filtered.head(10)['Food Item'].tolist()
    
    return recommendations[:10]  # Limit to top 10

def format_diet_plan(diet_name):
    """Format diet plan information as HTML"""
    if diet_name not in DIET_PLANS:
        return "<p>Please select a diet plan</p>"
    
    plan = DIET_PLANS[diet_name]
    
    html = f"<div style='padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 12px; color: white;'>"
    html += f"<h2 style='margin-top: 0; font-size: 28px;'>🍽️ {diet_name} Diet Plan</h2>"
    html += f"<p style='font-size: 16px; opacity: 0.95;'>{plan['description']}</p>"
    html += "</div>"
    
    # Macros and Calories
    html += "<div style='display: flex; gap: 15px; margin: 20px 0;'>"
    html += f"<div style='flex: 1; background-color: #e3f2fd; padding: 15px; border-radius: 8px; border-left: 4px solid #2196f3;'>"
    html += f"<strong style='color: #1976d2;'>📊 Daily Calories</strong><br>"
    html += f"<span style='font-size: 20px; color: #0d47a1;'>{plan['daily_calories']}</span>"
    html += "</div>"
    
    html += "<div style='flex: 1; background-color: #f3e5f5; padding: 15px; border-radius: 8px; border-left: 4px solid #9c27b0;'>"
    html += "<strong style='color: #7b1fa2;'>🎯 Macros</strong><br>"
    html += f"<span style='color: #4a148c;'>P: {plan['macros']['protein']} | C: {plan['macros']['carbs']} | F: {plan['macros']['fat']}</span>"
    html += "</div></div>"
    
    # Guidelines
    html += "<div style='background-color: #fff3e0; padding: 15px; border-radius: 8px; margin: 15px 0; border-left: 4px solid #ff9800;'>"
    html += "<h3 style='color: #e65100; margin-top: 0;'>📋 Key Guidelines</h3><ul style='margin: 10px 0 10px 20px; padding-left: 20px; list-style-type: disc;'>"
    for guideline in plan['guidelines']:
        html += f"<li style='margin: 8px 0; line-height: 1.6; color: #5d4037;'>{guideline}</li>"
    html += "</ul></div>"
    
    # Suitable Foods
    html += "<div style='background-color: #e8f5e9; padding: 15px; border-radius: 8px; margin: 15px 0; border-left: 4px solid #4caf50;'>"
    html += "<h3 style='color: #2e7d32; margin-top: 0;'>✅ Recommended Foods</h3>"
    html += "<p style='color: #1b5e20; font-size: 15px; font-weight: 500;'>" + ", ".join(plan['suitable_foods']) + "</p>"
    html += "</div>"
    
    # Foods to Avoid
    html += "<div style='background-color: #ffebee; padding: 15px; border-radius: 8px; margin: 15px 0; border-left: 4px solid #f44336;'>"
    html += "<h3 style='color: #c62828; margin-top: 0;'>❌ Foods to Avoid</h3>"
    html += "<p style='color: #b71c1c; font-size: 15px; font-weight: 500;'>" + ", ".join(plan['avoid_foods']) + "</p>"
    html += "</div>"
    
    # Get recommended foods from actual dataset
    dataset_recommendations = get_recommended_foods_from_dataset(diet_name)
    
    if dataset_recommendations:
        html += "<div style='background-color: #e1f5fe; padding: 15px; border-radius: 8px; margin: 15px 0; border-left: 4px solid #0288d1;'>"
        html += "<h3 style='color: #01579b; margin-top: 0;'>🎯 Recommended Foods from Your Dataset</h3>"
        html += "<div style='display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px;'>"
        
        for food in dataset_recommendations:
            html += f"<span style='background-color: #b3e5fc; color: #01579b; padding: 6px 12px; border-radius: 20px; font-size: 14px; font-weight: 500;'>{food}</span>"
        
        html += "</div>"
        html += f"<p style='color: #0277bd; font-size: 13px; margin-top: 10px; font-style: italic;'>Found {len(dataset_recommendations)} matching foods in your database</p>"
        html += "</div>"
    else:
        html += "<div style='background-color: #fff9c4; padding: 15px; border-radius: 8px; margin: 15px 0; border-left: 4px solid #fbc02d;'>"
        html += "<p style='color: #f57f17; margin: 0;'>⚠️ No matching foods found in your dataset for this diet plan. Try adding more foods or initialize the model first.</p>"
        html += "</div>"
    
    # Add meal combination recommendations
    html += format_meal_recommendations(diet_name)
    
    return html

def analyze_food_for_diet(predicted_class, metadata, diet_name):
    """Analyze if a food fits the selected diet plan"""
    if diet_name not in DIET_PLANS:
        return ""
    
    plan = DIET_PLANS[diet_name]
    
    html = f"<div style='padding: 20px; background-color: #f8f9fa; border-radius: 10px; margin-top: 20px;'>"
    html += f"<h3 style='color: #2c3e50; margin-top: 0;'>🔍 Diet Compatibility: {diet_name}</h3>"
    
    if not metadata:
        html += "<p style='color: #495057;'>No nutritional data available for compatibility analysis.</p></div>"
        return html
    
    # Extract nutritional values
    calories = metadata.get('energy_kcal', 0)
    protein = metadata.get('protein_g', 0)
    carbs = metadata.get('carbs_g', 0)
    fat = metadata.get('fat_g', 0)
    sugar = metadata.get('sugar_g', 0)
    sodium = metadata.get('sodium_mg', 0)
    
    # Compatibility scoring
    compatible = True
    warnings = []
    recommendations = []
    
    # Diet-specific analysis
    if diet_name == "Weight Loss":
        if calories > 300:
            warnings.append(f"High calorie content ({calories} kcal) - consume in moderation")
        if sugar > 15:
            warnings.append(f"High sugar ({sugar}g) - may hinder weight loss")
        if fat > 15:
            warnings.append("High fat content - watch portion sizes")
        recommendations.append("Pair with vegetables for added volume and nutrients")
        
    elif diet_name == "Muscle Gain":
        if protein < 10:
            warnings.append(f"Low protein ({protein}g) - consider adding protein-rich sides")
        if calories < 200:
            recommendations.append("Light food - you may need larger portions")
        recommendations.append("Consume within 2 hours post-workout for best results")
        
    elif diet_name == "Keto/Low Carb":
        if carbs > 10:
            compatible = False
            warnings.append(f"Too high in carbs ({carbs}g) - NOT keto-friendly")
        elif carbs > 5:
            warnings.append(f"Moderate carbs ({carbs}g) - track carefully")
        if fat < 5:
            recommendations.append("Add healthy fats (avocado, olive oil)")
            
    elif diet_name == "Diabetic-Friendly":
        if sugar > 10:
            compatible = False
            warnings.append(f"High sugar ({sugar}g) - may spike blood glucose")
        if carbs > 30:
            warnings.append(f"High carbs ({carbs}g) - monitor portions carefully")
        recommendations.append("Check blood sugar 2 hours after eating")
        
    elif diet_name == "Heart-Healthy":
        if sodium > 400:
            warnings.append(f"High sodium ({sodium}mg) - limit frequency")
        if fat > 20:
            warnings.append("High fat - ensure it's from healthy sources")
        recommendations.append("Pair with potassium-rich foods (bananas, spinach)")
    
    # Display compatibility status
    if compatible:
        if len(warnings) == 0:
            html += "<div style='background-color: #d4edda; padding: 12px; border-radius: 6px; border-left: 4px solid #28a745; margin: 10px 0;'>"
            html += "✅ <strong style='color: #155724;'>EXCELLENT FIT</strong> for your diet plan!"
            html += "</div>"
        else:
            html += "<div style='background-color: #fff3cd; padding: 12px; border-radius: 6px; border-left: 4px solid #ffc107; margin: 10px 0;'>"
            html += "⚠️ <strong style='color: #856404;'>ACCEPTABLE</strong> with considerations"
            html += "</div>"
    else:
        html += "<div style='background-color: #f8d7da; padding: 12px; border-radius: 6px; border-left: 4px solid #dc3545; margin: 10px 0;'>"
        html += "❌ <strong style='color: #721c24;'>NOT RECOMMENDED</strong> for this diet"
        html += "</div>"
    
    # Display warnings
    if warnings:
        html += "<div style='margin: 15px 0;'><strong style='color: #d35400;'>⚠️ Cautions:</strong><ul style='margin: 5px 0; color: #7d4419;'>"
        for warning in warnings:
            html += f"<li style='margin: 5px 0;'>{warning}</li>"
        html += "</ul></div>"
    
    # Display recommendations
    if recommendations:
        html += "<div style='margin: 15px 0;'><strong style='color: #229954;'>💡 Tips:</strong><ul style='margin: 5px 0; color: #196f3d;'>"
        for rec in recommendations:
            html += f"<li style='margin: 5px 0;'>{rec}</li>"
        html += "</ul></div>"
    
    html += "</div>"
    return html

# ============== FORMATTING FUNCTIONS ==============
def format_nutritional_info(metadata):
    """Format nutritional information as HTML"""
    if not metadata:
        return "<p>Metadata not found for this food item.</p>"
    
    source_badge = ""
    if metadata.get('source') == 'user_contributed':
        source_badge = "<span style='background-color: #3498db; color: white; padding: 3px 8px; border-radius: 4px; font-size: 12px; margin-left: 10px;'>👥 User Contributed</span>"
    
    html = "<div style='padding: 10px;'>"
    html += f"<h3 style='color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 5px;'>Nutritional Information{source_badge}</h3>"
    html += "<table style='width: 100%; border-collapse: collapse;'>"
    
    nutrients = [
        ('Protein ', 'protein_g', 'g'),
        ('Fat', 'fat_g', 'g'),
        ('Carbohydrates', 'carbs_g', 'g'),
        ('Energy', 'energy_kcal', 'kcal'),
        ('Sugar', 'sugar_g', 'g'),
        ('Fiber', 'fiber_g', 'g'),
        ('Sodium', 'sodium_mg', 'mg'),
        ('Cholesterol', 'cholesterol_mg', 'mg'),
        ('Iron', 'iron_mg', 'mg'),
    ]
    
    for label, key, unit in nutrients:
        value = metadata.get(key, 'N/A')
        html += f"<tr style='border-bottom: 1px solid #ecf0f1;'>"
        html += f"<td style='padding: 8px; font-weight: bold;'>{label}:</td>"
        html += f"<td style='padding: 8px;'>{value} {unit}</td>"
        html += "</tr>"
    
    html += "</table></div>"
    return html

def format_health_info(metadata):
    """Format health information as HTML"""
    if not metadata:
        return "<p>Metadata not found for this food item.</p>"
    
    html = "<div style='padding: 10px;'>"
    html += "<h3 style='color: #e74c3c; border-bottom: 2px solid #e74c3c; padding-bottom: 5px;'>Health Information</h3>"
    
    html += f"<div style='background-color: #fff8e1; padding: 10px; margin: 10px 0; border-left: 4px solid #ff9800; border-radius: 4px; color: #663c00;'>"
    html += f"<strong style='color: #e65100;'>Primary Risk Factor:</strong><br>{metadata.get('Primary Risk Factor', 'N/A')}"
    html += "</div>"
    
    html += f"<div style='background-color: #ffebee; padding: 10px; margin: 10px 0; border-left: 4px solid #f44336; border-radius: 4px; color: #8b0000;'>"
    html += f"<strong style='color: #c62828;'>Disease/Condition on Excess:</strong><br>{metadata.get('Disease/Condition on Excess', 'N/A')}"
    html += "</div>"
    
    html += "</div>"
    return html

def format_diet_recommendation(metadata):
    """Format diet recommendation as HTML"""
    if not metadata:
        return "<p>Metadata not found for this food item.</p>"
    
    html = "<div style='padding: 10px;'>"
    html += "<h3 style='color: #27ae60; border-bottom: 2px solid #27ae60; padding-bottom: 5px;'>Dietary Recommendation</h3>"
    
    html += f"<div style='background-color: #d4edda; padding: 15px; margin: 10px 0; border-left: 4px solid #28a745; border-radius: 4px;'>"
    html += f"<p style='margin: 0; font-size: 15px; color: #155724; font-weight: 500;'>{metadata.get('Dietary Role/Recommendation', 'N/A')}</p>"
    html += "</div>"
    
    html += "<h4 style='color: #2c3e50; margin-top: 15px;'>Additional Details</h4>"
    html += "<table style='width: 100%; border-collapse: collapse;'>"
    
    details = [
        ('Category', 'Food Category'),
        ('Cuisine', 'Cuisine'),
        ('Meal Type', 'Meal Type'),
        ('Diet Tags', 'Diet Tags'),
    ]
    
    for label, key in details:
        value = metadata.get(key, 'N/A')
        html += f"<tr style='border-bottom: 1px solid #ecf0f1;'>"
        html += f"<td style='padding: 8px; font-weight: bold;'>{label}:</td>"
        html += f"<td style='padding: 8px;'>{value}</td>"
        html += "</tr>"
    
    html += "</table></div>"
    return html

# ============== PREDICTION FUNCTIONS ==============
def predict_with_diet_plan(image, diet_name, show_alternatives):
    """Predict food and analyze for diet plan compatibility"""
    if food_model is None:
        return None, "Please initialize the model first", "", "", "", "", "", gr.update(visible=False)
    
    if image is None:
        return None, "Please upload an image", "", "", "", "", "", gr.update(visible=False)
    
    try:
        # Predict
        predicted_class, confidence = food_model.predict(image)
        
        # Check confidence threshold
        if confidence < 0.6:
            result_text = f"<div style='padding: 15px; background-color: #fff3cd; border-radius: 8px; border: 2px solid #ffc107;'>"
            result_text += f"<h2 style='color: #856404; margin-top: 0;'>Unknown Food Item</h2>"
            result_text += f"<p style='font-size: 16px;'>Confidence is too low ({confidence:.2%}) to make a reliable prediction.</p>"
            result_text += f"<p style='font-size: 14px; color: #666;'>Best guess was: {predicted_class}</p>"
            result_text += "</div>"
            
            unknown_message = "<div style='padding: 15px; background-color: #e8f4f8; border-radius: 8px; text-align: center;'>"
            unknown_message += "<p style='font-size: 16px; color: #2c3e50;'>Would you like to add this food item to the dataset?</p>"
            unknown_message += "<p style='font-size: 14px; color: #7f8c8d;'>Go to the 'Add New Food' tab to contribute this food item.</p>"
            unknown_message += "</div>"
            
            return image, result_text, unknown_message, "", "", "", "", gr.update(visible=True)
        
        # Get metadata
        metadata = food_model.match_metadata(predicted_class)
        
        # Format result
        result_text = f"<div style='padding: 15px; background-color: #e8f4f8; border-radius: 8px;'>"
        result_text += f"<h2 style='color: #2c3e50; margin-top: 0;'>Predicted Food: {predicted_class}</h2>"
        result_text += f"<p style='font-size: 18px;'><strong>Confidence:</strong> <span style='color: #27ae60; font-size: 22px;'>{confidence:.2%}</span></p>"
        result_text += "</div>"
        
        # Format outputs
        nutrition = format_nutritional_info(metadata)
        health = format_health_info(metadata)
        recommendation = format_diet_recommendation(metadata)
        
        # Diet plan analysis
        diet_analysis = analyze_food_for_diet(predicted_class, metadata, diet_name)
        
        # Find healthy alternatives only if checkbox is checked
        healthy_alternatives = ""
        if show_alternatives:
            healthy_alternatives = find_healthy_alternatives(predicted_class, metadata, diet_name)
        
        return image, result_text, nutrition, health, recommendation, diet_analysis, healthy_alternatives, gr.update(visible=False)
    
    except Exception as e:
        return None, f"Error during prediction: {str(e)}", "", "", "", "", "", gr.update(visible=False)

def predict_single_image(image):
    """Predict food item from uploaded image (without diet plan)"""
    if food_model is None:
        return None, "Please initialize the model first", "", "", "", gr.update(visible=False)
    
    if image is None:
        return None, "Please upload an image", "", "", "", gr.update(visible=False)
    
    try:
        # Predict
        predicted_class, confidence = food_model.predict(image)
        
        # Check confidence threshold
        if confidence < 0.6:
            result_text = f"<div style='padding: 15px; background-color: #fff3cd; border-radius: 8px; border: 2px solid #ffc107;'>"
            result_text += f"<h2 style='color: #856404; margin-top: 0;'>Unknown Food Item</h2>"
            result_text += f"<p style='font-size: 16px;'>Confidence is too low ({confidence:.2%}) to make a reliable prediction.</p>"
            result_text += f"<p style='font-size: 14px; color: #666;'>Best guess was: {predicted_class}</p>"
            result_text += "</div>"
            
            unknown_message = "<div style='padding: 15px; background-color: #e8f4f8; border-radius: 8px; text-align: center;'>"
            unknown_message += "<p style='font-size: 16px; color: #2c3e50;'>Would you like to add this food item to the dataset?</p>"
            unknown_message += "<p style='font-size: 14px; color: #7f8c8d;'>Go to the 'Add New Food' tab to contribute this food item.</p>"
            unknown_message += "</div>"
            
            return image, result_text, unknown_message, "", "", gr.update(visible=True)
        
        # Get metadata
        metadata = food_model.match_metadata(predicted_class)
        
        # Format result
        result_text = f"<div style='padding: 15px; background-color: #e8f4f8; border-radius: 8px;'>"
        result_text += f"<h2 style='color: #2c3e50; margin-top: 0;'>Predicted Food: {predicted_class}</h2>"
        result_text += f"<p style='font-size: 18px;'><strong>Confidence:</strong> <span style='color: #27ae60; font-size: 22px;'>{confidence:.2%}</span></p>"
        result_text += "</div>"
        
        # Format outputs
        nutrition = format_nutritional_info(metadata)
        health = format_health_info(metadata)
        recommendation = format_diet_recommendation(metadata)
        
        return image, result_text, nutrition, health, recommendation, gr.update(visible=False)
    
    except Exception as e:
        return None, f"Error during prediction: {str(e)}", "", "", "", gr.update(visible=False)

def predict_batch_images(files):
    """Predict food items from multiple uploaded images"""
    if food_model is None:
        return "Please initialize the model first"
    
    if not files:
        return "Please upload at least one image"
    
    html = "<div style='padding: 10px;'>"
    html += "<h3 style='color: #2c3e50;'>Batch Processing Results</h3>"
    
    for i, file in enumerate(files):
        try:
            image = Image.open(file.name).convert("RGB")
            predicted_class, confidence = food_model.predict(image)
            
            color = "#27ae60" if confidence > 0.7 else "#f39c12" if confidence > 0.5 else "#e74c3c"
            
            html += f"<div style='background-color: #f8f9fa; padding: 10px; margin: 10px 0; border-left: 4px solid {color}; border-radius: 4px;'>"
            html += f"<strong>Image {i+1}:</strong> {predicted_class} "
            html += f"<span style='color: {color}; font-weight: bold;'>(Confidence: {confidence:.2%})</span>"
            html += "</div>"
        except Exception as e:
            html += f"<div style='background-color: #f8d7da; padding: 10px; margin: 10px 0; border-left: 4px solid #dc3545; border-radius: 4px;'>"
            html += f"<strong>Image {i+1}:</strong> Error: {str(e)}"
            html += "</div>"
    
    html += "</div>"
    return html

def predict_top_k_foods(image, k=5):
    """Show top k predictions"""
    if food_model is None:
        return "Please initialize the model first"
    
    if image is None:
        return "Please upload an image"
    
    try:
        results = food_model.predict_top_k(image, k)
        
        html = "<div style='padding: 10px;'>"
        html += "<h3 style='color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 5px;'>Top Predictions</h3>"
        
        for i, result in enumerate(results, 1):
            confidence = result['confidence']
            color = "#27ae60" if confidence > 0.7 else "#f39c12" if confidence > 0.5 else "#e74c3c"
            width = int(confidence * 100)
            
            html += f"<div style='margin: 15px 0;'>"
            html += f"<div style='display: flex; justify-content: space-between; margin-bottom: 5px;'>"
            html += f"<span style='font-weight: bold;'>{i}. {result['class']}</span>"
            html += f"<span style='color: {color}; font-weight: bold;'>{confidence:.2%}</span>"
            html += "</div>"
            html += f"<div style='background-color: #ecf0f1; border-radius: 4px; overflow: hidden;'>"
            html += f"<div style='background-color: {color}; height: 20px; width: {width}%;'></div>"
            html += "</div></div>"
        
        html += "</div>"
        return html
    except Exception as e:
        return f"Error: {str(e)}"

# ============== WEBCAM LIVE DETECTION ==============
def webcam_live_detect(image, show_alternatives, diet_name):
    """Live webcam detection with full information"""
    if food_model is None:
        return None, "Please initialize the model first", "", "", "", "", gr.update(visible=False)
    
    if image is None:
        return None, "Waiting for camera feed...", "", "", "", "", gr.update(visible=False)
    
    try:
        # Predict
        predicted_class, confidence = food_model.predict(image)
        
        # Draw on image
        img_with_text = image.copy()
        img_array = np.array(img_with_text)
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Check confidence threshold
        if confidence < 0.6:
            text = f"Unknown ({confidence:.1%})"
            color = (0, 0, 255)  # Red
            
            # Calculate text size for background
            (text_width, text_height), baseline = cv2.getTextSize(text, font, 1.2, 2)
            
            # Draw background rectangle
            cv2.rectangle(img_array, (10, 10), (20 + text_width, 50 + text_height), (0, 0, 0), -1)
            cv2.putText(img_array, text, (15, 40), font, 1.2, color, 2, cv2.LINE_AA)
            
            img_with_text = Image.fromarray(img_array)
            
            result_text = f"<div style='padding: 15px; background-color: #fff3cd; border-radius: 8px; border: 2px solid #ffc107;'>"
            result_text += f"<h2 style='color: #856404; margin-top: 0;'>Unknown Food Item</h2>"
            result_text += f"<p style='font-size: 16px;'>Confidence is too low ({confidence:.2%}) to make a reliable prediction.</p>"
            result_text += f"<p style='font-size: 14px; color: #666;'>Best guess was: {predicted_class}</p>"
            result_text += "</div>"
            
            unknown_message = "<div style='padding: 15px; background-color: #e8f4f8; border-radius: 8px; text-align: center;'>"
            unknown_message += "<p style='font-size: 16px; color: #2c3e50;'>Would you like to add this food item to the dataset?</p>"
            unknown_message += "<p style='font-size: 14px; color: #7f8c8d;'>Go to the 'Add New Food' tab to contribute this food item.</p>"
            unknown_message += "</div>"
            
            return img_with_text, result_text, unknown_message, "", "", "", gr.update(visible=True)
        
        # Get metadata
        metadata = food_model.match_metadata(predicted_class)
        
        text = f"{predicted_class} ({confidence:.1%})"
        
        # Calculate text size for background
        (text_width, text_height), baseline = cv2.getTextSize(text, font, 1.2, 2)
        
        # Draw background rectangle
        cv2.rectangle(img_array, (10, 10), (20 + text_width, 50 + text_height), (0, 0, 0), -1)
        
        # Draw text
        color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255) if confidence > 0.5 else (0, 0, 255)
        cv2.putText(img_array, text, (15, 40), font, 1.2, color, 2, cv2.LINE_AA)
        
        img_with_text = Image.fromarray(img_array)
        
        # Format result
        result_text = f"<div style='padding: 15px; background-color: #e8f4f8; border-radius: 8px;'>"
        result_text += f"<h2 style='color: #2c3e50; margin-top: 0;'>Detected: {predicted_class}</h2>"
        result_text += f"<p style='font-size: 18px;'><strong>Confidence:</strong> <span style='color: #27ae60; font-size: 22px;'>{confidence:.2%}</span></p>"
        result_text += "</div>"
        
        # Format outputs
        nutrition = format_nutritional_info(metadata)
        health = format_health_info(metadata)
        recommendation = format_diet_recommendation(metadata)
        
        # Find healthy alternatives only if checkbox is checked
        healthy_alternatives = ""
        if show_alternatives:
            healthy_alternatives = find_healthy_alternatives(predicted_class, metadata, diet_name)
        
        return img_with_text, result_text, nutrition, health, recommendation, healthy_alternatives, gr.update(visible=False)
    
    except Exception as e:
        return image, f"<p style='color: red;'>Error: {str(e)}</p>", "", "", "", "", gr.update(visible=False)

# ============== ADD NEW FOOD ==============
def add_new_food(food_name, risk_factor, disease, recommendation, 
                 protein, fat, carbs, energy, sugar, fiber, sodium, 
                 cholesterol, iron, category, cuisine, meal_type, diet_tags):
    """Add a new food item to the NEW_FOOD_CSV file"""
    if not food_name:
        return "❌ Please provide a food name"
    
    try:
        # Create new row
        new_row = {
            'Food Item': food_name,
            'Primary Risk Factor': risk_factor or '',
            'Disease/Condition on Excess': disease or '',
            'Dietary Role/Recommendation': recommendation or '',
            'protein_g': protein or 0,
            'fat_g': fat or 0,
            'carbs_g': carbs or 0,
            'energy_kcal': energy or 0,
            'sugar_g': sugar or 0,
            'fiber_g': fiber or 0,
            'sodium_mg': sodium or 0,
            'cholesterol_mg': cholesterol or 0,
            'iron_mg': iron or 0,
            'Food Category': category or '',
            'Cuisine': cuisine or '',
            'Meal Type': meal_type or '',
            'Diet Tags': diet_tags or ''
        }
        
        # Load existing NEW_FOOD_CSV or create new
        if os.path.exists(NEW_FOOD_CSV):
            df = pd.read_csv(NEW_FOOD_CSV)
            # Check for duplicates
            if food_name in df['Food Item'].values:
                return f"⚠️ '{food_name}' already exists in the user-contributed database!"
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        else:
            df = pd.DataFrame([new_row])
        
        # Save to NEW_FOOD_CSV (separate file)
        df.to_csv(NEW_FOOD_CSV, index=False)
        
        # Reload metadata in model
        if food_model:
            food_model.load_metadata(CSV_PATH)
        
        return f"✅ Successfully added '{food_name}' to the user-contributed database!\n📁 Saved to: {NEW_FOOD_CSV}\n📊 Total user-contributed items: {len(df)}"
    
    except Exception as e:
        return f"❌ Error adding food: {str(e)}"

# ============== GRADIO INTERFACE ==============
def create_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(title="Food Detection & Diet Recommendation", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🍽️ Food Detection & Diet Recommendation App")
        gr.Markdown("Upload food images or use live webcam to get predictions and personalized diet recommendations.")
        
        # Initialize button at the top
        with gr.Row():
            init_btn = gr.Button("🚀 Initialize Model", variant="primary", scale=1)
            init_status = gr.Textbox(label="Status", scale=3, interactive=False)
        
        init_btn.click(initialize_model, outputs=init_status)
        
        # Tabs for different functionalities
        with gr.Tabs():
            # Tab 1: Diet Plan Selection & Food Analysis
            with gr.TabItem("🎯 Diet Plan Analysis"):
                gr.Markdown("### Choose your diet plan and analyze foods for compatibility")
                
                with gr.Row():
                    diet_selector = gr.Dropdown(
                        choices=list(DIET_PLANS.keys()),
                        label="Select Your Diet Plan",
                        value="Balanced/General Health",
                        interactive=True
                    )
                
                diet_plan_info = gr.HTML(label="Diet Plan Information")
                
                # Update diet plan info when selection changes
                diet_selector.change(
                    format_diet_plan,
                    inputs=diet_selector,
                    outputs=diet_plan_info
                )
                
                # Load initial diet plan
                app.load(
                    lambda: format_diet_plan("Balanced/General Health"),
                    outputs=diet_plan_info
                )
                
                gr.Markdown("---")
                gr.Markdown("### Upload food image to check compatibility")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        diet_input_image = gr.Image(type="pil", label="Upload Food Image")
                        diet_analyze_btn = gr.Button("🔍 Analyze Food for Diet", variant="primary", size="lg")
                        show_alternatives_checkbox = gr.Checkbox(label="Show Healthy Alternatives", value=True)
                    
                    with gr.Column(scale=1):
                        diet_output_image = gr.Image(label="Input Image")
                        diet_prediction_result = gr.HTML(label="Prediction Result")
                
                with gr.Row():
                    with gr.Column():
                        diet_nutrition_output = gr.HTML(label="Nutritional Information")
                    with gr.Column():
                        diet_health_output = gr.HTML(label="Health Risks")
                
                with gr.Row():
                    diet_recommendation_output = gr.HTML(label="Diet Recommendation")
                
                diet_compatibility_output = gr.HTML(label="Diet Compatibility Analysis")
                
                healthy_alternatives_output = gr.HTML(label="Healthy Alternatives")
                
                diet_unknown_food_notice = gr.HTML(visible=False, label="Add Food Notice")
                
                diet_analyze_btn.click(
                    predict_with_diet_plan,
                    inputs=[diet_input_image, diet_selector, show_alternatives_checkbox],
                    outputs=[diet_output_image, diet_prediction_result, diet_nutrition_output, 
                            diet_health_output, diet_recommendation_output, diet_compatibility_output, 
                            healthy_alternatives_output, diet_unknown_food_notice]
                )
            
            # Tab 2: Single Image Prediction
            '''with gr.TabItem("📸 Single Image Detection"):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_image = gr.Image(type="pil", label="Upload Food Image")
                        predict_btn = gr.Button("🔎 Analyze Food", variant="primary", size="lg")
                        single_show_alternatives = gr.Checkbox(label="Show Healthy Alternatives", value=False)
                        single_diet_selector = gr.Dropdown(
                            choices=list(DIET_PLANS.keys()),
                            label="Select Diet Plan (for alternatives)",
                            value="Balanced/General Health",
                            visible=False
                        )
                    
                    with gr.Column(scale=1):
                        output_image = gr.Image(label="Input Image")
                        prediction_result = gr.HTML(label="Prediction Result")
                
                # Show diet selector when alternatives checkbox is checked
                single_show_alternatives.change(
                    lambda x: gr.update(visible=x),
                    inputs=single_show_alternatives,
                    outputs=single_diet_selector
                )
                
                with gr.Row():
                    with gr.Column():
                        nutrition_output = gr.HTML(label="Nutritional Information")
                    with gr.Column():
                        health_output = gr.HTML(label="Health Risks")
                
                with gr.Row():
                    recommendation_output = gr.HTML(label="Diet Recommendation")
                
                single_alternatives_output = gr.HTML(label="Healthy Alternatives")
                
                unknown_food_notice = gr.HTML(visible=False, label="Add Food Notice")
                
                predict_btn.click(
                    predict_single_image,
                    inputs=[input_image, single_show_alternatives, single_diet_selector],
                    outputs=[output_image, prediction_result, nutrition_output, health_output, recommendation_output, single_alternatives_output, unknown_food_notice]
                )'''
            
            # Tab 3: Live Webcam Detection
            with gr.TabItem("📹 Live Webcam Detection"):
                gr.Markdown("### Real-time food detection from your camera")
                gr.Markdown("The detection updates automatically as you show different foods to the camera.")
                
                with gr.Row():
                    webcam_show_alternatives = gr.Checkbox(label="Show Healthy Alternatives", value=False)
                    webcam_diet_selector = gr.Dropdown(
                        choices=list(DIET_PLANS.keys()),
                        label="Select Diet Plan (for alternatives)",
                        value="Balanced/General Health",
                        visible=False
                    )
                
                # Show diet selector when alternatives checkbox is checked
                webcam_show_alternatives.change(
                    lambda x: gr.update(visible=x),
                    inputs=webcam_show_alternatives,
                    outputs=webcam_diet_selector
                )
                
                with gr.Row():
                    with gr.Column(scale=1):
                        webcam_input = gr.Image(sources=["webcam"], type="pil", label="Webcam Feed", streaming=True)
                    
                    with gr.Column(scale=1):
                        webcam_output_image = gr.Image(label="Detection Result")
                        webcam_prediction = gr.HTML(label="Detected Food")
                
                with gr.Row():
                    with gr.Column():
                        webcam_nutrition = gr.HTML(label="Nutritional Information")
                    with gr.Column():
                        webcam_health = gr.HTML(label="Health Risks")
                
                with gr.Row():
                    webcam_recommendation = gr.HTML(label="Diet Recommendation")
                
                webcam_alternatives_output = gr.HTML(label="Healthy Alternatives")
                
                webcam_unknown_notice = gr.HTML(visible=False, label="Add Food Notice")
                
                # Live streaming detection
                webcam_input.stream(
                    webcam_live_detect,
                    inputs=webcam_input,
                    outputs=[webcam_output_image, webcam_prediction, webcam_nutrition, webcam_health, webcam_recommendation, webcam_unknown_notice],
                    time_limit=300,
                    stream_every=0.5
                )
            
            # Tab 4: Top-K Predictions
            with gr.TabItem("📊 Top-K Predictions"):
                with gr.Row():
                    with gr.Column(scale=1):
                        topk_image = gr.Image(type="pil", label="Upload Food Image")
                        k_slider = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Number of Predictions (K)")
                        topk_btn = gr.Button("🎯 Get Top Predictions", variant="primary")
                    
                    with gr.Column(scale=1):
                        topk_output = gr.HTML(label="Top Predictions")
                
                topk_btn.click(
                    predict_top_k_foods,
                    inputs=[topk_image, k_slider],
                    outputs=topk_output
                )
            
            # Tab 5: Batch Processing
            with gr.TabItem("📦 Batch Processing"):
                with gr.Row():
                    with gr.Column(scale=1):
                        batch_files = gr.File(file_count="multiple", label="Upload Multiple Images")
                        batch_btn = gr.Button("⚡ Process Batch", variant="primary")
                    
                    with gr.Column(scale=1):
                        batch_output = gr.HTML(label="Batch Results")
                
                batch_btn.click(
                    predict_batch_images,
                    inputs=batch_files,
                    outputs=batch_output
                )
            
            # Tab 6: Add New Food
            with gr.TabItem("➕ Add New Food"):
                gr.Markdown("### Add a new food item to the user-contributed database")
                gr.Markdown(f"**Note:** New foods will be saved to `{NEW_FOOD_CSV}` (separate from the original database)")
                
                with gr.Row():
                    with gr.Column():
                        new_food_name = gr.Textbox(label="Food Name *", placeholder="e.g., chocolate cake")
                        new_risk_factor = gr.Textbox(label="Primary Risk Factor", placeholder="e.g., High Sugar")
                        new_disease = gr.Textbox(label="Disease/Condition on Excess", placeholder="e.g., Type 2 Diabetes")
                        new_recommendation = gr.Textbox(label="Dietary Recommendation", placeholder="e.g., Occasional treat only", lines=3)
                    
                    with gr.Column():
                        new_protein = gr.Number(label="Protein (g)", value=0)
                        new_fat = gr.Number(label="Fat (g)", value=0)
                        new_carbs = gr.Number(label="Carbohydrates (g)", value=0)
                        new_energy = gr.Number(label="Energy (kcal)", value=0)
                
                with gr.Row():
                    with gr.Column():
                        new_sugar = gr.Number(label="Sugar (g)", value=0)
                        new_fiber = gr.Number(label="Fiber (g)", value=0)
                        new_sodium = gr.Number(label="Sodium (mg)", value=0)
                        new_cholesterol = gr.Number(label="Cholesterol (mg)", value=0)
                    
                    with gr.Column():
                        new_iron = gr.Number(label="Iron (mg)", value=0)
                        new_category = gr.Textbox(label="Food Category", placeholder="e.g., Dessert")
                        new_cuisine = gr.Textbox(label="Cuisine", placeholder="e.g., Italian")
                        new_meal_type = gr.Textbox(label="Meal Type", placeholder="e.g., Dinner")
                
                new_diet_tags = gr.Textbox(label="Diet Tags", placeholder="e.g., High Protein, Low Carb")
                
                add_food_btn = gr.Button("✅ Add Food to Database", variant="primary", size="lg")
                add_food_output = gr.Textbox(label="Status", interactive=False, lines=3)
                
                add_food_btn.click(
                    add_new_food,
                    inputs=[new_food_name, new_risk_factor, new_disease, new_recommendation,
                           new_protein, new_fat, new_carbs, new_energy, new_sugar, new_fiber,
                           new_sodium, new_cholesterol, new_iron, new_category, new_cuisine,
                           new_meal_type, new_diet_tags],
                    outputs=add_food_output
                )
        
        gr.Markdown("---")
        gr.Markdown(f"""
        **📝 Notes:** 
        - Original data is stored in `{CSV_PATH}` (read-only)
        - User-contributed foods are saved to `{NEW_FOOD_CSV}` (your additions)
        """)
    
    return app

# ============== LAUNCH APP ==============
if __name__ == "__main__":
    app = create_interface()
    app.launch(share=True, server_name="0.0.0.0", server_port=7860)