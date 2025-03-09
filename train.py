from nlp_preprocessor import NLPPreprocessor

def train_model():
    preprocessor = NLPPreprocessor()
    
    preprocessor.train_model('sampel.csv', output_dir="./saved_model")

if __name__ == "__main__":
    train_model()