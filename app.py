import joblib
import gradio as gr
import re

def clean_text(text: str) -> str:

    # Remove HTML tags (e.g. "<br /><br />")
    text = re.sub(r"<.*?>", " ", text)
    # Collapse repeated whitespace introduced by tag removal
    text = re.sub(r"\s+", " ", text).strip()
    return text

clf = joblib.load("logistic_regression_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def predict(text):
    cleaned_text = clean_text(text)
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = clf.predict(vectorized_text)[0]

    label = "Positive" if prediction == 1 else "Negative"


    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(vectorized_text)[0]
        class_index = list(clf.classes_).index(prediction)
        confidence = proba[class_index]
        
        return f"{label} ({confidence:.1%} confidence)"

    return label

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=5, placeholder="Paste a movie review here..."),
    outputs="text",
    title="IMDB Movie Review Sentiment Classifier",
    description="Enter text to predict its sentiment (Positive or Negative)."
)

if __name__ == "__main__":
    demo.launch()