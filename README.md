# Customer Churn Predictor

Predicts telecom customer churn using XGBoost with 81% accuracy and 0.86 ROC-AUC.

## Live Demo
Frontend: https://shagun-gaur-official.github.io/churn-predictor/
API Docs: https://churn-predictor-api-k8ed.onrender.com/docs
<img width="1532" height="1020" alt="image" src="https://github.com/user-attachments/assets/1b20b2a0-b534-4c34-845b-d2b47232a75d" />

## Tech Stack
- Python, Pandas, Scikit-learn, XGBoost
- FastAPI, Uvicorn
- HTML, Tailwind CSS
- Render (backend), GitHub Pages (frontend)
- 
## Model Performance
| Metric | Score |
|--------|-------|
| Accuracy | 0.81 |
| F1 Score | 0.61 |
| ROC-AUC | 0.86 |

## Run locally
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```
Then open frontend-html/index.html in browser.
