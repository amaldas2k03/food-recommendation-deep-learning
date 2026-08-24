# Food Recommendation with Deep Learning

A deep-learning food-recognition and health-recommendation app. Upload a photo of a
dish and the app classifies it with a fine-tuned **ResNet-50** model, looks up its
nutritional profile, and recommends healthier alternatives and meal combos based on a
chosen diet plan (Weight Loss, Muscle Gain, Keto/Low Carb, Balanced, and more). Built
with [Gradio](https://www.gradio.app/).

## Features

- **Food image classification** using a fine-tuned ResNet-50 model.
- **Nutritional lookup** — protein, fat, carbs, calories, sugar, fiber, sodium, etc.
- **Health insights** — primary risk factors and conditions linked to overconsumption.
- **Diet-plan recommendations** — healthier alternatives and meal combos tailored to
  the selected diet plan.
- **User-contributed foods** — add new foods to the local dataset from the UI.
- Interactive web UI powered by Gradio.

## Project structure

| File | Description |
|------|-------------|
| `app4.py` | Main Gradio application (latest version). |
| `app3.py` | Earlier version of the app. |
| `train.ipynb` | Notebook used to train / fine-tune the ResNet-50 model. |
| `food_metadata.csv` | Nutrition + health metadata for recognised foods. |
| `1763360300482_healthy_food_dataset_appended.csv` | Healthy-alternatives dataset. |
| `new_food_data.csv` | User-contributed foods added from the app. |
| `requirements.txt` | Python dependencies. |

> **Note:** The trained model weights (`best_resnet50_food_subset.pth`, ~270 MB) are
> **not** included in this repository due to their size. Train the model with
> `train.ipynb` or obtain the `.pth` file separately and place it in the project root
> before running the app.

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/amaldas2k03/food-recommendation-deep-learning.git
   cd food-recommendation-deep-learning
   ```

2. (Recommended) Create a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   # source venv/bin/activate # macOS / Linux
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Place the trained model file `best_resnet50_food_subset.pth` in the project root.

## Running the app

```bash
python app4.py
```

The Gradio interface will be served locally (default `http://0.0.0.0:7860`) and a public
share link will be generated.

## Model

The classifier is a ResNet-50 pretrained on ImageNet and fine-tuned on a food-image
subset. The checkpoint stores both the model weights (`model_state_dict`) and the list
of class labels (`classes`).

## License

This project was created for academic / educational purposes.
