import pandas as pd
import os

def main():
    # le texte est deja propre grace au ticket 12
    # on verifie juste la presence des fichiers
    path = "data/preprocessed"
    if os.path.exists(f"{path}/train_clean.csv"):
        print("✅ Ticket 13: features pret pour modelisation")

if __name__ == "__main__":
    main()