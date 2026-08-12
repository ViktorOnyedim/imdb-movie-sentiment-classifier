from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import gradio as gr
import re

app = FastAPI(title="IMDB Sentiment Classifier API", description="API for predicting sentiment of IMDB movie reviews.")

clf = joblib.load("logistic_regression_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def clean_text(text: str) -> str:
    # Remove HTML tags (e.g. "<br /><br />")
    text = re.sub(r"<.*?>", " ", text)
    # Collapse repeated whitespace introduced by tag removal
    text = re.sub(r"\s+", " ", text).strip()
    return text


class ReviewRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    label: str
    confidence: float | None = None


@app.get("/")
async def root():
    return {"message": "IMDB Sentiment Classifier API. POST to /predict with {'text': '...'}"}

@app.post("/predict")
def predict(request: ReviewRequest):
    cleaned_text = clean_text(request.text)
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = clf.predict(vectorized_text)[0]

    label = "Positive" if prediction == 1 else "Negative"

    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(vectorized_text)[0]
        class_index = list(clf.classes_).index(prediction)
        confidence = proba[class_index]
        # return {"label": label, "confidence": f"{confidence:.1%}"}

    # return {"label": label}
    return PredictionResponse(label=label, confidence=confidence)