from deep_translator import GoogleTranslator as Translator
import progressbar
import random
from utils.utils import corrector
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer
from bert_score import BERTScorer
import os
# from TTS.api import TTS
# from speechbrain.lobes.models.huggingface_wav2vec import HuggingFaceWav2Vec2
import torch
import torchaudio
# from transformers import pipeline
import jiwer
# import tts.PoemesProfonds.preprocessing as pp
# from tts.PoemesProfonds.lecture import *
# from keras.models import load_model
# import pandas as pd
import pickle
import nltk



def read_hats():
    # dataset = [{"reference": ref, "hypA": hypA, "nbrA": nbrA, "hypB": hypB, "nbrB": nbrB}, ...]
    dataset = []
    with open("datasets/hats.txt", "r", encoding="utf8") as file:
        next(file)
        for line in file:
            line = line[:-1].split("\t")
            dictionary = dict()
            dictionary["ref"] = line[0]
            dictionary["hypA"] = line[1]
            dictionary["nbrA"] = int(line[2])
            dictionary["hypB"] = line[3]
            dictionary["nbrB"] = int(line[4])
            dataset.append(dictionary)
    return dataset


# -------------------------- Intermediate data -------------------------- #

def get_memory_generate(task):
    if task == "traduction":
        translator = Translator(source='fr', target='en')
        return translator
    elif task == "tts":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tts = TTS(model_path="./tts/model/tts_models--multilingual--multi-dataset--xtts_v2", config_path="./tts/model/tts_models--multilingual--multi-dataset--xtts_v2/config.json").to(device)
        return tts
    elif task == "ner":
        ner_model = pipeline(
            task='ner',
            model="cmarkea/distilcamembert-base-ner",
            tokenizer="cmarkea/distilcamembert-base-ner",
            aggregation_strategy="simple"
        )
        return ner_model
    else:
        raise ValueError("task is not recognized:", task)

def generate(text, task, memory):
    if task == "traduction":
        translator = memory
        return translator.translate(text)
    elif task == "tts":
        tts = memory
        get_wav(text, tts)
        return text
    elif task == "ner":
        ner_model = memory
        ners = NER_bio(text, ner_model)
        return ners
    else:
        raise ValueError("task is not recognized:", task)

def intermediate_data(dataset, task, verbose=True):
    # generate the intermediate data set

    # if file already exists, skip generation
    try:
        with open("datasets/intermediaire_" + task + ".txt", "r", encoding="utf8") as file:
            print("File already exists. Skipped generation.")
            return
    except:
        pass
    memory_generate = get_memory_generate(task) # translator = Translator(source='fr', target='en')
    txt = ""
    # progressbar
    if verbose:
        bar = progressbar.ProgressBar(maxval=len(dataset))
        bar.start()
        i = 0
    for dictionary in dataset:
        ref = dictionary["ref"]
        hypA = dictionary["hypA"]
        hypB = dictionary["hypB"]
        genref = generate(ref, task, memory_generate)
        genhypA = generate(hypA, task, memory_generate)
        genhypB = generate(hypB, task, memory_generate)
        txt += ref + "\t" + hypA + "\t" + genref + "\t" + genhypA + "\n"
        txt += ref + "\t" + hypB + "\t" + genref + "\t" + genhypB + "\n"

        if verbose:
            bar.update(i)
            i += 1

    with open("datasets/intermediaire_" + task + ".txt", "w", encoding="utf8") as file:
        file.write(txt)

def load_intermediate_data(task):
    dataset = []
    with open("datasets/intermediaire_" + task + ".txt", "r", encoding="utf8") as file:
        for line in file:
            line = line[:-1].split("\t")
            dictionary = dict()
            dictionary["ref"] = line[0]
            dictionary["hyp"] = line[1]
            dictionary["genref"] = line[2]
            dictionary["genhyp"] = line[3]
            dataset.append(dictionary)
    return dataset



# -------------------------- Metrics -------------------------- #


# ----- Translation ------ #

def semdist(ref, hyp, memory):
    model = memory
    ref_projection = model.encode(ref).reshape(1, -1)
    hyp_projection = model.encode(hyp).reshape(1, -1)
    score = cosine_similarity(ref_projection, hyp_projection)[0][0]
    return (1-score)*100 # lower is better

def bertscore(ref, hyp, memory):
    scorer = memory
    P, R, F1 = scorer.score([hyp], [ref])
    return 100-F1.numpy()[0]*100 # lower is better

# ----- TTS ------ #

def load_wav(path):
    waveform, sample_rate = torchaudio.load(path)
    # Resample if needed
    target_sample_rate = 16000  # Wav2Vec 2.0's expected sample rate
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
        waveform = resampler(waveform)
    # Ensure mono-channel (1 channel)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    return waveform

def get_wav(text, tts): # save audio file if not exists
    text = text.lower()
    # check if files exist in dataset/audiofiles
    namefile = text
    # remove unexpected character
    accepted = "abcdefghijklmnopqrstuvwxyzéèêëàâäôöûüùîïç_"
    for x in namefile:
        if x not in accepted:
            namefile = namefile.replace(x, "_")
    # if audio file does not exists
    if not os.path.isfile("datasets/audiofiles/" + namefile + ".wav"):
        tts.tts_to_file(text, speaker_wav="datasets/audiofiles/bfm15.wav", language="fr", file_path="datasets/audiofiles/" + namefile + ".wav")
    # load audio file
    return load_wav("datasets/audiofiles/" + namefile + ".wav")

def get_features(wav, model_w2v2):
    features = model_w2v2.forward(wav)
    # compute only the average of the features in order to have the same shape
    average_features = torch.mean(features[0], dim=0, keepdim=True)[0]
    return average_features

def speech_difference(ref, hyp, memory):
    tts, model_w2v2 = memory
    # get audio files
    ref_wav = get_wav(ref, tts)
    hyp_wav = get_wav(hyp, tts)
    # get features
    ref_features = get_features(ref_wav, model_w2v2).detach().numpy().reshape(1, -1)
    hyp_features = get_features(hyp_wav, model_w2v2).detach().numpy().reshape(1, -1)
    # compute cosine similarity
    cs = cosine_similarity(ref_features, hyp_features)[0][0]
    return (1-cs)*100 # lower is better

def load_w2v_model():
    # HuggingFace model hub
    model_hub_w2v2 = "LeBenchmark/wav2vec2-FR-7K-large"
    model_w2v2 = HuggingFaceWav2Vec2(model_hub_w2v2, save_path='./save')
    return model_w2v2

# ----- TTS/Phoneme Error Rate ------ #

def phonetisor_save(txt, lecteur, namefile):
    phonemes = lecteur.lire_vers(txt)
    with open("datasets/phonemes/" + namefile + ".txt", "w") as f:
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
    if not os.path.isfile("datasets/phonemes/" + namefile + ".txt"):
        phonemes = phonetisor_save(text, lecteur, namefile)
    else:
        with open("datasets/phonemes/" + namefile + ".txt", "r") as f:
            phonemes = f.read()
    # load audio file
    return phonemes

def PhonemeErrorRate(ref, hyp, lecteur):
    ref = phonetisor(ref, lecteur)
    hyp = phonetisor(hyp, lecteur)
    return jiwer.cer(ref, hyp)

# ----- NER ------ #

def NER_bio(text, ner_model):
    ners = ner_model(text)
    bio_tags = ['O'] * len(text.split())

    for ner_entity in ners:
        entity_group = ner_entity['entity_group']
        start_token = ner_entity['start']
        end_token = ner_entity['end']

        # Convert entity start and end indices to token indices
        start_token_index = len(text[:start_token].split())
        end_token_index = len(text[:end_token].split())

        # Assign BIO tags to the tokens
        if start_token_index < len(bio_tags):
            bio_tags[start_token_index] = f'B-{entity_group}'

        for i in range(start_token_index + 1, min(end_token_index + 1, len(bio_tags))):
            bio_tags[i] = f'I-{entity_group}'

    return " ".join(bio_tags)

def ner_error_rate(ref, hyp, memory): # ner error rate
    # pipeline = memory
    # ref_ner = get_ner(ref, pipeline)
    # hyp_ner = get_ner(hyp, pipeline)
    # score = jiwer.wer(ref_ner, hyp_ner)
    score = jiwer.wer(ref, hyp)
    return score*100 # lower is better

# ----- Character Error Rate ------ #

def character_error_rate(ref, hyp, memory):
    return jiwer.cer(ref, hyp)*100 # lower is better

# ----- BLEU ------ #

def bleu(ref, hyp, memory):
    ref = ref.split(" ")
    hyp = hyp.split(" ")
    BLEUscore = nltk.translate.bleu_score.sentence_bleu([ref], hyp)
    return (1-BLEUscore)*100 # lower is better


# ----- Common ------ #

def load_metric(metric):
    if metric == "semdist":
        return semdist, SentenceTransformer('dangvantuan/sentence-camembert-large')
    elif metric == "bertscore":
        return bertscore, BERTScorer(lang="fr")
    elif metric == "speech_difference":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tts = TTS(model_path="./tts/model/tts_models--multilingual--multi-dataset--xtts_v2", config_path="./tts/model/tts_models--multilingual--multi-dataset--xtts_v2/config.json").to(device)
        model_w2v2 = load_w2v_model()
        return speech_difference, (tts, model_w2v2)
    elif metric == "ner_error_rate":
        return ner_error_rate, 0 # memory is null since ner are already obtained
    elif metric == "phoneme_error_rate":
        dico_u, dico_m, df_w2p = pd.read_pickle(os.path.join(".", "tts", "PoemesProfonds", "data", "dicos.pickle"))
        ltr2idx, phon2idx, Tx, Ty = pp.chars2idx(df_w2p)
        model_lire = load_model(os.path.join(".", "tts", "PoemesProfonds", "models", "lecteur", "lecteur_mdl.h5")) #"CE1_T12_l10.h5"))
        lecteur = Lecteur(Tx, Ty, ltr2idx, phon2idx, dico_u, dico_m, n_brnn1=90, n_h1=80, net=model_lire, blank="_")
        memory = lecteur
        return PhonemeErrorRate, memory
    elif metric == "cer":
        return character_error_rate, 0 # memory is null
    elif metric == "bleu":
        return bleu, 0 # memory is null
    else:
        raise ValueError("metric is not recognized:", metric)

def compute_metrics(task, metric1, metric2, verbose=True):
    # hats must have been translated

    # if file already exists, skip generation
    try:
        with open("datasets/metrics_" + task + "_" + metric1 + "_" + metric2 + ".txt", "r", encoding="utf8") as file:
            print("File already exists. Skipped generation.")
            return
    except:
        pass

    function_metric1, memory_metric1 = load_metric(metric1)
    function_metric2, memory_metric2 = load_metric(metric2)

    dataset = load_intermediate_data(task)
    txt = ""

    if verbose:
        # progressbar
        bar = progressbar.ProgressBar(maxval=len(dataset))
        bar.start()
        i = 0
    for dictionary in dataset:
        ref = dictionary["ref"]
        hyp = dictionary["hyp"]
        genref = dictionary["genref"]
        genhyp = dictionary["genhyp"]
        score1 = function_metric1(ref, hyp, memory_metric1)
        score2 = function_metric2(genref, genhyp, memory_metric2)
        txt += ref + "\t" + hyp + "\t" + genref + "\t" + genhyp + "\t" + str(score1) + "\t" + str(score2) + "\n"

        if verbose:
            bar.update(i)
            i += 1
    
    # save data set
    with open("datasets/metrics_" + task + "_" + metric1 + "_" + metric2 + ".txt", "w", encoding="utf8") as file:
        file.write(txt)

def load_computed_metrics(task, metric1, metric2):
    dataset = []
    with open("datasets/metrics_" + task + "_" + metric1 + "_" + metric2 + ".txt", "r", encoding="utf8") as file:
        for line in file:
            line = line[:-1].split("\t")
            dictionary = dict()
            dictionary["ref"] = line[0]
            dictionary["hyp"] = line[1]
            dictionary["genref"] = line[2]
            dictionary["genhyp"] = line[3]
            dictionary[metric1] = line[4]
            dictionary[metric2] = line[5]
            dataset.append(dictionary)
    return dataset


def correct_and_save(task, metric1, metric2, verbose=False):
    # correct word error in the hypothesis and compute the improvements

    # if file already exists
    try:
        with open("datasets/corrections_" + task + "_" + metric1 + "_" + metric2 + ".txt", "r", encoding="utf8") as file:
            print("File already exists. Skipped generation.")
            return
    except:
        pass

    dataset = load_computed_metrics(task, metric1, metric2)

    function_metric1, memory_metric1 = load_metric(metric1)
    function_metric2, memory_metric2 = load_metric(metric2)
    memory_generate = get_memory_generate(task)

    txt = ""
    if verbose:
        # progressbar
        bar = progressbar.ProgressBar(maxval=len(dataset))
        bar.start()
        i = 0
    for dictionary in dataset:
        ref = dictionary["ref"]
        hyp = dictionary["hyp"]
        genref = dictionary["genref"]
        genhyp = dictionary["genhyp"]
        score1 = float(dictionary[metric1])
        score2 = float(dictionary[metric2])
        corrections = corrector(ref, hyp) # list of possible word corrections
        txt += ref + "\t" + hyp + "\t" + genref + "\t" + genhyp + "\t" + str(score1) + "\t" + str(score2)
        for correction in corrections:
            score1_correction = function_metric1(ref, correction, memory_metric1)
            gencorrection = generate(correction, task, memory_generate)
            score2_correction = function_metric2(genref, gencorrection, memory_metric2)
            txt += "\t" + correction + "," + gencorrection + "," + str(score1_correction) + "," + str(score2_correction)
        txt += "\n"
        if verbose:
            bar.update(i)
            i += 1
    with open("datasets/corrections_" + task + "_" + metric1 + "_" + metric2 + ".txt", "w", encoding="utf8") as file:
        file.write(txt)
    print("Function worked properly.")


def load_corrected_hats(task, metric1, metric2):
    dataset = []
    with open("datasets/corrections_" + task + "_" + metric1 + "_" + metric2 + ".txt", "r", encoding="utf8") as file:
        for line in file:
            line = line[:-1].split("\t")
            dictionary = dict()
            dictionary["ref"] = line[0]
            dictionary["hyp"] = line[1]
            dictionary["genref"] = line[2]
            dictionary["genhyp"] = line[3]
            dictionary[metric1] = line[4]
            dictionary[metric2] = line[5]
            dictionary["corrections"] = []
            for i in range(6, len(line)):
                dictionary["corrections"].append(line[i].split(","))
            dataset.append(dictionary)
    return dataset

def load_only_improvements(task, metric1, metric2, soustraction=True):
    zero = 0
    improvements_intrinsic = []
    improvements_extrinsic = []
    dataset = load_corrected_hats(task, metric1, metric2)
    for dictionary in dataset:
        score1 = float(dictionary[metric1])
        score2 = float(dictionary[metric2])
        if score1 == 0 or score2 == 0:
            zero += 1
            if not soustraction:
                continue
        for correction in dictionary["corrections"]:
            score1_correction = float(correction[-2])
            score2_correction = float(correction[-1])
            if soustraction:
                improvement_intrinsic = score1 - score1_correction
                improvement_extrinsic = score2 - score2_correction
            else:
                improvement_intrinsic = score1_correction/score1*100 - 100
                improvement_extrinsic = score2_correction/score2*100 - 100
            improvements_intrinsic.append(improvement_intrinsic)
            improvements_extrinsic.append(improvement_extrinsic)
    # print("Correct generations despite transcription errors:", zero, "out of", len(dataset), "times.")
    return improvements_intrinsic, improvements_extrinsic



# -------------------------- Correlation -------------------------- #

def compute_correlation_intrinsic_extrinsic(task, metric1, metric2):
    dataset = load_computed_metrics(task, metric1, metric2)
    scores1 = []
    scores2 = []
    for dictionary in dataset:
        scores1.append(float(dictionary[metric1]))
        scores2.append(float(dictionary[metric2]))
    # print("pearson:", pearsonr(scores1, scores2))
    pearson_score = pearsonr(scores1, scores2)
    # print("spearman:", spearmanr(scores1, scores2))
    spearman_score = spearmanr(scores1, scores2)
    return pearson_score[0], spearman_score[0]


def correlation_minED_extrinsic(task, metric1, metric2, Random=False):
    # compute correlation between the minimum edit distance and the extrinsic metric
    improvements_intrinsic, improvements_extrinsic = load_only_improvements(task, metric1, metric2)

    if Random:
        for i in range(len(improvements_intrinsic)):
            # erase list with a list of random uniform numbers
            improvements_intrinsic[i] = random.uniform(0, 1)
            improvements_extrinsic[i] = random.uniform(0, 1)
    pearson_score = pearsonr(improvements_intrinsic, improvements_extrinsic)
    # print("pearson:", pearson)
    spearman_score = spearmanr(improvements_intrinsic, improvements_extrinsic)
    # print("spearman:", spearman)
    return pearson_score[0], spearman_score[0]


def load_list_improvements(task, metric1, metric2, soustraction=True):
    zero = 0
    improvements_intrinsic = []
    improvements_extrinsic = []
    dataset = load_corrected_hats(task, metric1, metric2)
    for dictionary in dataset:
        improvements_local_intrinsic = []
        improvements_local_extrinsic = []
        score1 = float(dictionary[metric1])
        score2 = float(dictionary[metric2])
        if score1 == 0 or score2 == 0:
            zero += 1
            if not soustraction:
                continue
        for correction in dictionary["corrections"]:
            score1_correction = float(correction[-2])
            score2_correction = float(correction[-1])
            if soustraction:
                improvement_intrinsic = score1 - score1_correction
                improvement_extrinsic = score2 - score2_correction
            else:
                improvement_intrinsic = score1_correction/score1*100 - 100
                improvement_extrinsic = score2_correction/score2*100 - 100
            improvements_local_intrinsic.append(improvement_intrinsic)
            improvements_local_extrinsic.append(improvement_extrinsic)
        improvements_intrinsic.append(improvements_local_intrinsic)
        improvements_extrinsic.append(improvements_local_extrinsic)
    # print("Correct generations despite transcription errors:", zero, "out of", len(dataset), "times.")
    return improvements_intrinsic, improvements_extrinsic


def correlation_minED_extrinsic_local(task, metric1, metric2, signif=0.05, Random=False):
    # compute correlation between the minimum edit distance and the extrinsic metric

    skipped = 0

    improvements_intrinsic, improvements_extrinsic = load_list_improvements(task, metric1, metric2)
    pearsons = []
    spearmans = []
    pvalue_pearsons = []
    pvalue_spearmans = []

    for i in range(len(improvements_intrinsic)):
        if len(improvements_intrinsic[i]) < 2:
            continue

        if Random:
            # erase list with a list of random uniform numbers
            improvements_intrinsic[i] = [random.uniform(0, 1) for _ in range(len(improvements_intrinsic[i]))]
            improvements_extrinsic[i] = [random.uniform(0, 1) for _ in range(len(improvements_extrinsic[i]))]

        pearson = pearsonr(improvements_intrinsic[i], improvements_extrinsic[i])
        spearman = spearmanr(improvements_intrinsic[i], improvements_extrinsic[i])

        # check if pearson and spearman are nan
        if pearson[0] != pearson[0] or spearman[0] != spearman[0] or spearman[1] != spearman[1]:
            skipped += 1
        else:
            if pearson[1] > signif or spearman[1] > signif: # significativity if pvalue > 0.05
                skipped += 1
            else:
                pearsons.append(pearson[0])
                spearmans.append(spearman[0])

                pvalue_pearsons.append(pearson[1])
                pvalue_spearmans.append(spearman[1])
    try:
        pearson_avg = sum(pearsons)/len(pearsons)
        spearman_avg = sum(spearmans)/len(spearmans)
    except ZeroDivisionError:
        pearson_avg = 0
        spearman_avg = 0
    return pearson_avg, spearman_avg


def correlation_best(task, metric1, metric2, Random=False):
    # compute the number of times the intrisic metric agree to determine the best correction

    improvements_intrinsic, improvements_extrinsic = load_list_improvements(task, metric1, metric2)

    best_agree = 0
    disagree = 0
    skipped = 0
    for i in range(len(improvements_intrinsic)):
        if len(improvements_intrinsic[i]) < 2:
            skipped += 1
            continue

        if Random:
            # erase list with a list of random uniform numbers
            improvements_intrinsic[i] = [random.uniform(0, 1) for _ in range(len(improvements_intrinsic[i]))]
            improvements_extrinsic[i] = [random.uniform(0, 1) for _ in range(len(improvements_extrinsic[i]))]
        
        # compute rank of intrinsic and extrinsic list
        intrinsic_rank = []
        extrinsic_rank = []
        for j in range(len(improvements_intrinsic[i])):
            intrinsic_rank.append((j, improvements_intrinsic[i][j]))
            extrinsic_rank.append((j, improvements_extrinsic[i][j]))
        intrinsic_rank.sort(key=lambda x: x[1], reverse=True)
        extrinsic_rank.sort(key=lambda x: x[1], reverse=True)        

        # check if the best correction is the same for intrinsic and extrinsic metrics
        if intrinsic_rank[0][0] == extrinsic_rank[0][0]:
            best_agree += 1
        else:
            disagree += 1
    return best_agree/(len(improvements_intrinsic)-skipped)*100



def correlation_ANR(task, metric1, metric2, Random=False):
    # compute the Average Normalized Rank
    anrs = []
    improvements_intrinsic, improvements_extrinsic = load_list_improvements(task, metric1, metric2)
    skipped = 0
    for i in range(len(improvements_intrinsic)):
        if len(improvements_intrinsic[i]) < 5:
            skipped += 1
            continue
        if Random:
            # erase list with a list of random uniform numbers
            improvements_intrinsic[i] = [random.uniform(0, 1) for _ in range(len(improvements_intrinsic[i]))]
            improvements_extrinsic[i] = [random.uniform(0, 1) for _ in range(len(improvements_extrinsic[i]))]

        # find index of the best intrinsic improvement
        index_best_intrinsic = improvements_intrinsic[i].index(max(improvements_intrinsic[i]))
        sorted_list = sorted(improvements_extrinsic[i], reverse=True)
        rank = sorted_list.index(improvements_extrinsic[i][index_best_intrinsic])

        a = rank+1
        b = len(improvements_extrinsic[i])
        
        ANR = 1-(a-1)/(b-1)
        anrs.append(ANR)
    return sum(anrs)/len(anrs)


def generate_all_data(task, metric1, metric2):
    print("Generating all data for", task, "and metrics", metric1, "and", metric2)
    print("Reading dataset...")
    dataset = read_hats()
    print("Generating intermediate data...")
    intermediate_data(dataset, task, verbose=True)
    print("Computing metrics...")
    compute_metrics(task, metric1, metric2, verbose=True)
    print("Correcting and saving...")
    correct_and_save(task, metric1, metric2, verbose=True)


def compute_all_correlations(task, metric1, metric2):
    print("task:", task)
    print("metric1:", metric1)
    print("metric2:", metric2)
    pearson, spearman = compute_correlation_intrinsic_extrinsic(task, metric1, metric2)
    print("pearson:", pearson)
    print("spearman:", spearman)
    print()

def read_results():
    # Task,Intrisinc Metric,Extrinsic Metric,Global Correlation Pearson,Global Correlation Spearman,Local Correlation Pearson pvalue 0.05,Local Correlation Spearman pvalue 0.05,Choice Agreement P@1,Choice Agreement ANR
    results = dict()

    # check if file does not exist
    if os.path.exists("results/results.txt"):
        with open("results/results.txt", "r", encoding="utf8") as file:
            next(file)
            for line in file:
                line = line[:-1].split(",")
                task = line[0]
                metric1 = line[1]
                metric2 = line[2]
                if task not in results:
                    results[task] = dict()
                if metric1 not in results[task]:
                    results[task][metric1] = dict()
                if metric2 not in results[task][metric1]:
                    results[task][metric1][metric2] = dict()
                results[task][metric1][metric2]["Global Correlation Pearson"] = float(line[3])
                results[task][metric1][metric2]["Global Correlation Spearman"] = float(line[4])
                results[task][metric1][metric2]["Local Correlation Pearson pvalue 0.05"] = float(line[5])
                results[task][metric1][metric2]["Local Correlation Spearman pvalue 0.05"] = float(line[6])
                results[task][metric1][metric2]["Local Correlation Pearson"] = float(line[7])
                results[task][metric1][metric2]["Local Correlation Spearman"] = float(line[8])
                results[task][metric1][metric2]["Choice Agreement P@1"] = float(line[9])
                results[task][metric1][metric2]["Choice Agreement ANR"] = float(line[10])
    return results

def write_results(results):
    with open("results/results.txt", "w", encoding="utf8") as file:
        file.write("Task,Intrisinc Metric,Extrinsic Metric,Global Correlation Pearson,Global Correlation Spearman,Local Correlation Pearson pvalue 0.05,Local Correlation Spearman pvalue 0.05,Local Correlation Pearson,Local Correlation Spearman,Choice Agreement P@1,Choice Agreement ANR\n")
        for task in results:
            for metric1 in results[task]:
                for metric2 in results[task][metric1]:
                    file.write(task + "," + metric1 + "," + metric2 + "," + str(results[task][metric1][metric2]["Global Correlation Pearson"]) + "," + str(results[task][metric1][metric2]["Global Correlation Spearman"]) + "," + str(results[task][metric1][metric2]["Local Correlation Pearson pvalue 0.05"]) + "," + str(results[task][metric1][metric2]["Local Correlation Spearman pvalue 0.05"]) + "," + str(results[task][metric1][metric2]["Local Correlation Pearson"]) + "," + str(results[task][metric1][metric2]["Local Correlation Spearman"]) + "," + str(results[task][metric1][metric2]["Choice Agreement P@1"]) + "," + str(results[task][metric1][metric2]["Choice Agreement ANR"]) + "\n")
    print("Results saved.")

def already_computed(task, metric1, metric2, correlation, results):
    if task in results:
        if metric1 in results[task]:
            if metric2 in results[task][metric1]:
                if correlation in results[task][metric1][metric2]:
                    return True
            else:
                results[task][metric1][metric2] = dict()
        else:
            results[task][metric1] = dict()
            results[task][metric1][metric2] = dict()
    else:
        results[task] = dict()
        results[task][metric1] = dict()
        results[task][metric1][metric2] = dict()
    return False

def massive_test(task, metric1, metric2):
    results = read_results()
    
    # check first if the results are already computed
    if not already_computed(task, metric1, metric2, "Global Correlation Pearson", results):
        pearson, spearman = correlation_minED_extrinsic(task, metric1, metric2, Random=False)
        results[task][metric1][metric2]["Global Correlation Pearson"] = pearson
        results[task][metric1][metric2]["Global Correlation Spearman"] = spearman
    if not already_computed(task, metric1, metric2, "Local Correlation Pearson pvalue 0.05", results):
        pearson, spearman = correlation_minED_extrinsic_local(task, metric1, metric2, signif=0.05, Random=False)
        results[task][metric1][metric2]["Local Correlation Pearson pvalue 0.05"] = pearson
        results[task][metric1][metric2]["Local Correlation Spearman pvalue 0.05"] = spearman
    if not already_computed(task, metric1, metric2, "Local Correlation Pearson", results):
        pearson, spearman = correlation_minED_extrinsic_local(task, metric1, metric2, signif=100, Random=False)
        results[task][metric1][metric2]["Local Correlation Pearson"] = pearson
        results[task][metric1][metric2]["Local Correlation Spearman"] = spearman
    if not already_computed(task, metric1, metric2, "Choice Agreement P@1", results):
        score = correlation_best(task, metric1, metric2, Random=False)
        results[task][metric1][metric2]["Choice Agreement P@1"] = score
    if not already_computed(task, metric1, metric2, "Choice Agreement ANR", results):
        anrs = correlation_ANR(task, metric1, metric2, Random=False)
        results[task][metric1][metric2]["Choice Agreement ANR"] = anrs

    write_results(results)
    

if __name__ == '__main__':
    # task = "traduction"
    # metric1 = "semdist"
    # metric2 = "bertscore"
    
    # task = "ner"
    # metric1 = "semdist"
    # metric2 = "ner_error_rate"

    # task = "tts"
    # metric1 = "semdist"
    # metric2 = "speech_difference"

    # task = "tts"
    # metric1 = "phoneme_error_rate"
    # metric2 = "speech_difference"

    # generate_all_data(task, metric1, metric2)
    # massive_test(task, metric1, metric2)

    intrinsic = ["semdist", "semdist", "cer", "cer"]
    extrinsic = ["bertscore", "bleu", "bertscore", "bleu"]
    tasks = ["traduction"]*len(intrinsic)

    for i in range(len(intrinsic)):
        task = tasks[i]
        metric1 = intrinsic[i]
        metric2 = extrinsic[i]
        # generate_all_data(task, metric1, metric2)
        # massive_test(task, metric1, metric2)
        compute_all_correlations(task, metric1, metric2)

    print("Ended without problem.")