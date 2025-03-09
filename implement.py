import os
from nlp_preprocessor import NLPPreprocessor

def interactive_qa():
    model_dir = "./saved_model"
    
    dataset_path = './sampel.csv'
    
    preprocessor = NLPPreprocessor(model_dir=model_dir, dataset_path=dataset_path)
    
    print("Selamat datang di Sistem Tanya Jawab Interaktif!")
    print("Ketik 'keluar' untuk mengakhiri program.")
    print("[DEBUG] Dataset Columns:", preprocessor.dataset.columns)  
    print("[DEBUG] Sample Data:\n", preprocessor.dataset.head())

    while True:
        try:
            user_query = input("\nMasukkan pertanyaan Anda: ")
            
            if user_query.lower() in ['keluar', 'exit', 'quit']:
                print("Terima kasih telah menggunakan sistem tanya jawab.")
                break
            
            result = preprocessor.process_query(user_query)
            print("[DEBUG] Result:", result)
            print("\nPertanyaan:", result['original_query'])
            print("Konteks:", result['context'])
            print("Jawaban:", result['optimized_answer'])
        
        except KeyboardInterrupt:
            print("\n\nProgram dihentikan oleh pengguna.")
            break
        except Exception as e:
            print(f"Terjadi kesalahan: {e}")
            continue

if __name__ == "__main__":
    interactive_qa()