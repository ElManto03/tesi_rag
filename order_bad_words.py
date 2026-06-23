def pulisci_e_ordina_badwords(file_input: str, file_output: str):
    try:
        # 1. Legge il file eliminando spazi bianchi e righe vuote
        with open(file_input, 'r', encoding='utf-8') as f:
            # .strip() rimuove i \n e gli spazi a inizio/fine riga
            parole = {line.strip().lower() for line in f if line.strip()}
        
        # 2. Ordina la lista alfabeticamente
        parole_ordinate = sorted(list(parole))
        
        # 3. Riscrive le parole nel nuovo file, una per riga
        with open(file_output, 'w', encoding='utf-8') as f:
            for parola in parole_ordinate:
                f.write(f"{parola}\n")
                
        print(f"[OK] Elaborazione completata! Salvale {len(parole_ordinate)} parole uniche in: {file_output}")
        
    except FileNotFoundError:
        print(f"[ERRORE] Impossibile trovare il file di input: {file_input}")
    except Exception as e:
        print(f"[ERRORE] Si è verificato un problema: {str(e)}")

if __name__ == "__main__":
# Esempio di utilizzo:
# Assicurati che 'lista_grezza.txt' sia nella stessa cartella dello script
    pulisci_e_ordina_badwords("bad_words.txt", "bad_wordss.txt")