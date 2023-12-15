import PoemesProfonds.preprocessing as pp
from PoemesProfonds.lecture import *
from keras.models import load_model
import pandas as pd
import pickle
import os
import jiwer




def phonetisor_save(txt, lecteur, namefile):
    phonemes = lecteur.lire_vers(txt)
    with open("../datasets/phonemes/" + namefile + ".txt", "w") as f:
        f.write(phonemes)
    return phonemes


def phonetisor(text, lecteur): # save phonemes file if not exists
    text = text.lower()
    # check if files exist in dataset/phonemes
    namefile = text
    # remove unexpected character
    accepted = "abcdefghijklmnopqrstuvwxyzéèêëàâäôöûüùîïç_"
    for x in namefile:
        if x not in accepted:
            namefile = namefile.replace(x, "_")
    # if phonemes file does not exists
    if not os.path.isfile("../datasets/phonemes/" + namefile + ".txt"):
        phonemes = phonetisor_save(text, lecteur, namefile)
    else:
        with open("../datasets/phonemes/" + namefile + ".txt", "r") as f:
            phonemes = f.read()
    # load audio file
    return phonemes


def PhonemeErrorRate(ref, hyp, lecteur):
    ref = phonetisor(ref, lecteur)
    hyp = phonetisor(hyp, lecteur)
    return jiwer.cer(ref, hyp)

def test():
    dico_u, dico_m, df_w2p = pd.read_pickle(os.path.join(".", "PoemesProfonds", "data", "dicos.pickle"))
    ltr2idx, phon2idx, Tx, Ty = pp.chars2idx(df_w2p)
    model_lire = load_model(os.path.join(".", "PoemesProfonds", "models", "lecteur", "lecteur_mdl.h5")) #"CE1_T12_l10.h5"))
    lecteur = Lecteur(Tx, Ty, ltr2idx, phon2idx, dico_u, dico_m, n_brnn1=90, n_h1=80, net=model_lire, blank="_")


    memory = lecteur
    text = "Bonjour, comment allez-vous ?"
    phonemes = phonetisor(text, memory)
    print(phonemes)

if __name__ == "__main__":
    test()