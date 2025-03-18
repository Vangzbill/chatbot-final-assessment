from flask import Flask, request, jsonify
from flask_cors import CORS
from nlp_preprocessor import NLPPreprocessor

app = Flask(__name__)
CORS(app)  

model_dir = "./saved_model"
dataset_path = './sampel.csv'
preprocessor = NLPPreprocessor(model_dir=model_dir, dataset_path=dataset_path)

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    question = data.get("question")

    if not question:
        return jsonify({"error": "Pertanyaan tidak boleh kosong"}), 400

    try:
        result = preprocessor.process_query(question)
        
        return jsonify({
            "question": result["original_query"],
            "context": result["context"],
            "answer": result["optimized_answer"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
