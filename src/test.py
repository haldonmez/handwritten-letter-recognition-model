from src.predict import load_model, predict_image

model = load_model("models/emnist_model_4_ver2.pth", input_size=1, output_size=27)
predicted_class, confidence = predict_image(model, "samples/last_inverted_cropped.png")

print(f"Predicted class: {predicted_class}")
print(f"Confidence: {confidence*100:.2f}%")
