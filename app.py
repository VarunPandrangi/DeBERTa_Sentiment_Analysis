import gradio as gr
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import emoji
import re
import html
import time

# Configuration
MODEL_PATH = "./output/saved_model"

print(f"Loading model from {MODEL_PATH}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Ensure the model is saved in './saved_model'. Run the notebook first if needed.")
    exit(1)

# Preprocessing function (Mirroring the notebook)
def clean_text(text):
    text = str(text)
    text = html.unescape(text)
    text = emoji.demojize(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text) # Remove URLs
    text = re.sub(r'@\w+', '', text) # Remove Mentions
    return text.strip()

def predict_sentiment(text):
    if not text:
        return None, None, None
    
    start_time = time.time()
    clean_input = clean_text(text)
    
    # Tokenize
    inputs = tokenizer(clean_input, return_tensors="pt", truncation=True, max_length=128)
    token_count = len(inputs['input_ids'][0])
    
    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=-1)
    
    # Extract scores
    neg_score = probabilities[0][0].item()
    pos_score = probabilities[0][1].item()
    
    end_time = time.time()
    duration = f"{end_time - start_time:.4f}s"

    # Determine Label
    if pos_score > neg_score:
        label = "POSITIVE 😃"
        color = "#28a745" # Green
        confidence = pos_score
        sentiment_desc = "Positive"
    else:
        label = "NEGATIVE 😞"
        color = "#dc3545" # Red
        confidence = neg_score
        sentiment_desc = "Negative"
        
    # Create a styled HTML output for the label
    html_label = f"<div style='width:100%; text-align:center; padding: 10px; background-color: {color}20; border-radius: 8px; border: 2px solid {color};'><h2 style='color: {color}; margin:0;'>{label}</h2></div>"
    
    scores = {
        "Negative": neg_score,
        "Positive": pos_score
    }
    
    # Analysis Logic HTML (with inline styles for visibility)
    analysis_html = f"""
    <div style='background-color: #f8f9fa; border-radius: 8px; border: 1px solid #e9ecef; padding: 20px; margin-top: 15px; color: #212529;'>
        <h4 style='margin-top: 0; color: #495057; border-bottom: 2px solid #dee2e6; padding-bottom: 8px;'>🔍 Analysis Breakdown</h4>
        
        <p style='margin: 15px 0; line-height: 1.6;'>
            <strong style='color: #212529;'>1. Preprocessing:</strong><br>
            <span style='color: #666;'>Raw Input:</span> <em style='color: #495057;'>"{text}"</em><br>
            <span style='color: #666;'>Cleaned Text:</span> <code style='background-color: #e9ecef; padding: 3px 6px; border-radius: 4px; color: #d63384; font-family: monospace;'>{clean_input}</code><br>
            <small style='color: #6c757d;'>(HTML decoded, URLs removed, Emojis converted to text)</small>
        </p>
        
        <p style='margin: 15px 0; line-height: 1.6;'>
            <strong style='color: #212529;'>2. Tokenization:</strong><br>
            The text was split into <strong style='color: #0d6efd;'>{token_count}</strong> tokens for the DeBERTa model.
        </p>
        
        <p style='margin: 15px 0; line-height: 1.6;'>
            <strong style='color: #212529;'>3. Model Decision:</strong><br>
            The model is <strong style='color: {color};'>{confidence:.1%}</strong> confident this is <strong style='color: {color};'>{sentiment_desc}</strong>.<br>
            <small style='color: #6c757d;'>⚡ Inference time: {duration}</small>
        </p>
    </div>
    """
    
    return html_label, scores, analysis_html

# Custom CSS for a professional look
custom_css = """
.container { max-width: 1000px; margin: auto; padding-top: 20px; }
.footer { text-align: center; margin-top: 40px; font-size: 0.8em; color: #666; }
.analysis-box { background-color: #f8f9fa; border-radius: 8px; border: 1px solid #e9ecef; padding: 15px; margin-top: 15px; }
.analysis-box h4 { margin-top: 0; color: #495057; border-bottom: 1px solid #dee2e6; padding-bottom: 5px; }
.analysis-box code { background-color: #e9ecef; padding: 2px 4px; border-radius: 4px; color: #d63384; font-family: monospace; }
"""

# Gradio UI
with gr.Blocks(title="Sentiment Analysis") as demo:
    gr.HTML(f"<style>{custom_css}</style>")
    with gr.Column(elem_classes=["container"]):
        gr.Markdown(
            """
            #  DeBERTa Sentiment Analysis
            ### Enterprise-Grade Sentiment Classification
            
            This dashboard uses a fine-tuned **DeBERTa V3** model to classify text as **Positive** or **Negative**.
            It handles emojis, slang, and complex sentence structures.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=2):
                input_text = gr.Textbox(
                    label="Input Text", 
                    placeholder="Type your review, tweet, or comment here...",
                    lines=4
                )
                submit_btn = gr.Button("Analyze Sentiment", variant="primary", size="lg")
                
            with gr.Column(scale=1):
                label_output = gr.HTML(label="Sentiment")
                score_output = gr.Label(label="Confidence Scores", num_top_classes=2)
                details_output = gr.HTML(label="Analysis Details")

        with gr.Accordion("ℹ️ How it works & Confidence Scores", open=False):
            gr.Markdown(
                """
                ### Logic Behind Confidence Scores
                1. **Logits**: The model produces raw numerical scores (logits) for each class.
                2. **Softmax**: We apply the Softmax function to convert logits into probabilities:
                   $$ P(class) = \\frac{e^{logit}}{\\sum e^{logits}} $$
                3. **Interpretation**: The score represents the model's certainty. A score of 0.95 means the model is 95% sure of its prediction.
                
                ### Preprocessing Pipeline
                * **HTML Decoding**: Converts `&amp;` to `&`.
                * **Demojization**: Converts 😃 to `:smiley:`.
                * **Cleaning**: Removes URLs and @mentions.
                """
            )

        gr.Markdown("### 🧪 Test Examples")
        gr.Examples(
            examples=[
                # 1. Long Positive Paragraph
                ["I had such an incredible day at the festival. Every performer brought something unique, the food stalls were delicious, and everyone I met was warm and welcoming. I can't wait to go next year!"],

                # 2. Long Negative Paragraph
                ["Nothing about this dinner was enjoyable. The waiter was rude, my order got mixed up, and the food tasted awful. I left feeling frustrated and let down."],

                # 3. Mixed Sentiment Paragraph
                ["The new gadget looks cool and has some nice features, but the battery dies quickly and customer support hasn’t been helpful. I’m happy with it sometimes, but it’s not perfect."],

                # 4. Short Positive Sentences
                ["Best purchase ever."],
                ["Totally loved it!"],
                ["Super happy!"],

                # 5. Short Negative Sentences
                ["This is terrible."],
                ["Complete waste of time."],
                ["Not impressed."],

                # 6. Neutral Sentences
                ["The train arrives at 7am."],
                ["I will attend the meeting tomorrow."],
                ["It's just another day."],

                # 7. Ambiguous/Contextual
                ["Interesting..."],
                ["Sure, whatever."],
                ["If you say so."],

                # 8. Sarcasm/Irony
                ["Great job... as usual."],
                ["Just what I needed, another error."],
                ["Oh perfect, my phone died again."],

                # 9. Strong Emotions (Single Words)
                ["Amazing!"],
                ["Disappointing."],
                ["Meh."],

                # 10. Just Letters
                ["A"],
                ["z"],

                # 11. Only Symbols
                ["!!!"],
                ["..."],
                ["???"],
                [":)"],
                [":("],
                [":/"],

                # 12. Only Emojis
                ["🥳😍"],
                ["😢😠"],
                ["😐"],
                ["🤔"],
                ["💩"],
                ["😂🤣"],

                # 13. Emoji + Text
                ["Wow, loved it! 😍"],
                ["Ugh, this sucks. 😞"],
                ["Nothing makes sense anymore 😂"],

                # 15. Social Media Formats
                ["LMAO that’s wild bro 😂😂"],
                ["Shoutout @johndoe for the help!"],
                ["IDK what’s up with this tbh…"],

                # 16. Keyword Trigger
                ["Scam"],
                ["Refund"],
                ["Support"],
                ["Love"],
                ["Hate"],

                # 17. Factual/Neutral Statement
                ["The sky is blue."],
                ["There are 12 months in a year."],
                ["Cats are animals."],

                # 18. Code, Numbers, Random Input
                ["print('Hello world')"],
                ["12345"],
                ["asdfghjkl"],
                ["!@#$%^&*()"],

                # 19. Non-English Phrases
                ["C’est la vie"],
                ["¡Feliz cumpleaños!"],
                ["Das ist gut."],

                # 20. Edge Case Mix
                ["Absolutely no words... 😶"],
                ["BEST. DAY. EVER."],
                ["I seriously can't believe this happened to me 😭"]
            ],
            inputs=input_text,
            outputs=[label_output, score_output, details_output],
            fn=predict_sentiment,
            run_on_click=True,
            examples_per_page=50
        )
        
        submit_btn.click(
            fn=predict_sentiment, 
            inputs=input_text, 
            outputs=[label_output, score_output, details_output]
        )
        
        gr.Markdown("<div class='footer'>Powered by DeBERTa V3 & Gradio</div>")

if __name__ == "__main__":
    demo.launch()
