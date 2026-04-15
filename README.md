# Multi Hospital Lab Result Prediction

## Run Locally
pip install -r requirements.txt
python train.py
streamlit run streamlit_app.py

## Run Flask API
python app.py

## Deployment (Render)
- Push to GitHub
- Connect to Render
- Use:
  Build: pip install -r requirements.txt
  Start: gunicorn app:app
