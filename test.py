import os
import sys

# Sostituisci con il tuo percorso reale dell'ambiente tesi_test
env_path = r"C:\Users\federico.mantoni\AppData\Local\miniconda3\envs\tesi_test"
lib_path = os.path.join(env_path, "Lib", "site-packages", "torch", "lib")

# Forza Python a caricare le DLL da questa cartella
if os.path.exists(lib_path):
    os.add_dll_directory(lib_path)
    print(f"Cartella DLL aggiunta: {lib_path}")

try:
    import torch
    print("--- Risultato ---")
    print(f"Torch caricato correttamente!")
    print(f"GPU disponibile: {torch.cuda.is_available()}")
except Exception as e:
    print(f"Errore durante l'import: {e}")