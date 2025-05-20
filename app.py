from fastai.vision.all import *
import gradio as gr

# Modeli yükle
learn = load_learner("model (1).pkl")

# Tahmin fonksiyonu
def predict(img):
    pred, pred_idx, probs = learn.predict(img)
    return {learn.dls.vocab[i]: float(probs[i]) for i in range(len(probs))}

# Gradio arayüzü
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Evcil Hayvan Sınıflandırıcı",
    description="Bir kedi veya köpek görseli yükleyin, model türünü tahmin etsin.",
    
)

# Uygulamayı çalıştır
demo.launch()
