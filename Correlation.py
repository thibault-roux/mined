from deep_translator import GoogleTranslator as Translator
import progressbar
import random

from utils.utils import corrector

from sklearn.metrics.pairwise import cosine_similarity


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

def translate_hats_and_save(dataset, verbose=True):
    translator = Translator(source='fr', target='en')
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
        tradref = translator.translate(ref)
        tradhypA = translator.translate(hypA)
        tradhypB = translator.translate(hypB)
        txt += ref + "\t" + hypA + "\t" + tradref + "\t" + tradhypA + "\n"
        txt += ref + "\t" + hypB + "\t" + tradref + "\t" + tradhypB + "\n"

        if verbose:
            bar.update(i)
            i += 1

    with open("datasets/hats_with_translations.txt", "w", encoding="utf8") as file:
        file.write(txt)

def load_translated_hats():
    dataset = []
    with open("datasets/hats_with_translations.txt", "r", encoding="utf8") as file:
        for line in file:
            line = line[:-1].split("\t")
            dictionary = dict()
            dictionary["ref"] = line[0]
            dictionary["hyp"] = line[1]
            dictionary["tradref"] = line[2]
            dictionary["tradhyp"] = line[3]
            dataset.append(dictionary)
    return dataset


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

def save_semdist_bertscore(verbose=True):
    # hats must have been translated

    # SemDist Sentence Camembert-large
    from sentence_transformers import SentenceTransformer
    semdist_model = SentenceTransformer('dangvantuan/sentence-camembert-large')

    # BERTScore
    from bert_score import BERTScorer
    bertscore_model = BERTScorer(lang="en")

    dataset = load_translated_hats()
    txt = ""

    if verbose:
        # progressbar
        bar = progressbar.ProgressBar(maxval=len(dataset))
        bar.start()
        i = 0
    for dictionary in dataset:
        ref = dictionary["ref"]
        hyp = dictionary["hyp"]
        tradref = dictionary["tradref"]
        tradhyp = dictionary["tradhyp"]
        semdist_score = semdist(ref, hyp, semdist_model)
        bertscore_score = bertscore(tradref, tradhyp, bertscore_model)
        txt += ref + "\t" + hyp + "\t" + tradref + "\t" + tradhyp + "\t" + str(semdist_score) + "\t" + str(bertscore_score) + "\n"

        if verbose:
            bar.update(i)
            i += 1
    
    # save data set
    with open("datasets/hats_with_semdist_bertscore.txt", "w", encoding="utf8") as file:
        file.write(txt)

def load_semdist_bertscore():
    dataset = []
    with open("datasets/hats_with_semdist_bertscore.txt", "r", encoding="utf8") as file:
        for line in file:
            line = line[:-1].split("\t")
            dictionary = dict()
            dictionary["reference"] = line[0]
            dictionary["hyp"] = line[1]
            dictionary["tradref"] = line[2]
            dictionary["tradhyp"] = line[3]
            dictionary["semdist"] = line[4]
            dictionary["bertscore"] = line[5]
            dataset.append(dictionary)
    return dataset

def compute_correlation_intrinsic_extrinsic():
    dataset = load_semdist_bertscore()
    semdist = []
    bertscore = []
    for dictionary in dataset:
        semdist.append(float(dictionary["semdist"]))
        bertscore.append(float(dictionary["bertscore"]))
    from scipy.stats import pearsonr
    print("pearson:", pearsonr(semdist, bertscore))
    from scipy.stats import spearmanr
    print("spearman:", spearmanr(semdist, bertscore))


def correct_and_save(verbose=False):
    # correct word error in the hypothesis and compute the improvements
    dataset = load_semdist_bertscore()

    # SemDist Sentence Camembert-large
    from sentence_transformers import SentenceTransformer
    semdist_model = SentenceTransformer('dangvantuan/sentence-camembert-large')

    # BERTScore
    from bert_score import BERTScorer
    bertscore_model = BERTScorer(lang="en")
    
    translator = Translator(source='fr', target='en')

    txt = ""
    if verbose:
        # progressbar
        bar = progressbar.ProgressBar(maxval=len(dataset))
        bar.start()
        i = 0
    for dictionary in dataset:
        ref = dictionary["reference"]
        hyp = dictionary["hyp"]
        tradref = dictionary["tradref"]
        tradhyp = dictionary["tradhyp"]
        semdist_score = float(dictionary["semdist"])
        bertscore_score = float(dictionary["bertscore"])
        corrections = corrector(ref, hyp) # list of possible word corrections
        txt += ref + "\t" + hyp + "\t" + tradref + "\t" + tradhyp + "\t" + str(semdist_score) + "\t" + str(bertscore_score)
        for correction in corrections:
            semdist_correction = semdist(ref, correction, semdist_model)
            tradcorrection = translator.translate(correction)
            bertscore_correction = bertscore(tradref, tradcorrection, bertscore_model)
            txt += "\t" + correction + "," + tradcorrection + "," + str(semdist_correction) + "," + str(bertscore_correction)
        txt += "\n"
        if verbose:
            bar.update(i)
            i += 1
    with open("datasets/hats_with_corrections.txt", "w", encoding="utf8") as file:
        file.write(txt)
    print("Function worked properly.")


def load_corrected_hats():
    dataset = []
    with open("datasets/hats_with_corrections.txt", "r", encoding="utf8") as file:
        for line in file:
            line = line[:-1].split("\t")
            dictionary = dict()
            dictionary["reference"] = line[0]
            dictionary["hyp"] = line[1]
            dictionary["tradref"] = line[2]
            dictionary["tradhyp"] = line[3]
            dictionary["semdist"] = line[4]
            dictionary["bertscore"] = line[5]
            dictionary["corrections"] = []
            for i in range(6, len(line)):
                dictionary["corrections"].append(line[i].split(","))
            dataset.append(dictionary)
    return dataset

def repair_corrected_dataset(): # does not work because it is not easy to find where the comma are really
    # load the dataset and repair it by selecting only float numbers
    with open("datasets/hats_with_corrections.txt", "r", encoding="utf8") as file:
        txt = ""
        nbrError = 0
        commaInref = 0
        commaInhyp = 0
        for line in file:
            linesplit = line[:-1].split("\t")
            ref = linesplit[0]
            hyp = linesplit[1]
            if "," in ref:
                commaInref += 1
            if "," in hyp:
                commaInhyp += 1
            corrections = linesplit[6:]
            for quadruple in corrections:
                if len(quadruple.split(",")) != 4:
                    print(line)
                    input()
                    print(quadruple)
                    nbrError += 1
        print("nbrError:", nbrError)
        print("commaInref:", commaInref)
        print("commaInhyp:", commaInhyp)
        # should check if there is a comma in references
        # I can keep the first as a real comma and the one preceding numbers?
        # also check if there are numbers in the references or hypotheses?

def load_only_improvements(soustraction=True):
    zero = 0
    improvements_intrinsic = []
    improvements_extrinsic = []
    dataset = load_corrected_hats()
    for dictionary in dataset:
        semdist_score = float(dictionary["semdist"])
        bertscore_score = float(dictionary["bertscore"])
        if semdist_score == 0 or bertscore_score == 0:
            zero += 1
            if not soustraction:
                continue
        for correction in dictionary["corrections"]:
            semdist_correction = float(correction[-2])
            bertscore_correction = float(correction[-1])
            if soustraction:
                improvement_intrinsic = semdist_score - semdist_correction
                improvement_extrinsic = bertscore_score - bertscore_correction
            else:
                improvement_intrinsic = semdist_correction/semdist_score*100 - 100
                improvement_extrinsic = bertscore_correction/bertscore_score*100 - 100
            improvements_intrinsic.append(improvement_intrinsic)
            improvements_extrinsic.append(improvement_extrinsic)
    print("Correct traductions despite trascription errors:", zero, "out of", len(dataset), "times.")
    return improvements_intrinsic, improvements_extrinsic


def correlation_minED_extrinsic(Random=False):
    # compute correlation between the minimum edit distance and the extrinsic metric
    improvements_intrinsic, improvements_extrinsic = load_only_improvements()

    if Random:
        for i in range(len(improvements_intrinsic)):
            # erase list with a list of random uniform numbers
            improvements_intrinsic[i] = random.uniform(0, 1)
            improvements_extrinsic[i] = random.uniform(0, 1)
    
    from scipy.stats import pearsonr
    pearson = pearsonr(improvements_intrinsic, improvements_extrinsic)
    print("pearson:", pearson)
    from scipy.stats import spearmanr
    spearman = spearmanr(improvements_intrinsic, improvements_extrinsic)
    print("spearman:", spearman)
    print(type(pearson[0]), type(spearman[0]))
    return pearson[0], spearman[0]


def load_list_improvements(soustraction=True):
    zero = 0
    improvements_intrinsic = []
    improvements_extrinsic = []
    dataset = load_corrected_hats()
    for dictionary in dataset:
        improvements_local_intrinsic = []
        improvements_local_extrinsic = []
        semdist_score = float(dictionary["semdist"])
        bertscore_score = float(dictionary["bertscore"])
        if semdist_score == 0 or bertscore_score == 0:
            zero += 1
            if not soustraction:
                continue
        for correction in dictionary["corrections"]:
            semdist_correction = float(correction[-2])
            bertscore_correction = float(correction[-1])
            if soustraction:
                improvement_intrinsic = semdist_score - semdist_correction
                improvement_extrinsic = bertscore_score - bertscore_correction
            else:
                improvement_intrinsic = semdist_correction/semdist_score*100 - 100
                improvement_extrinsic = bertscore_correction/bertscore_score*100 - 100
            improvements_local_intrinsic.append(improvement_intrinsic)
            improvements_local_extrinsic.append(improvement_extrinsic)
        improvements_intrinsic.append(improvements_local_intrinsic)
        improvements_extrinsic.append(improvements_local_extrinsic)
    print("Correct traductions despite trascription errors:", zero, "out of", len(dataset), "times.")
    return improvements_intrinsic, improvements_extrinsic


def correlation_minED_extrinsic_local(signif=0.05, Random=False):
    # compute correlation between the minimum edit distance and the extrinsic metric
    from scipy.stats import pearsonr
    from scipy.stats import spearmanr

    skipped = 0

    improvements_intrinsic, improvements_extrinsic = load_list_improvements()
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

    # print("skipped:", skipped)
    
    # print("pearson:", sum(pearsons)/len(pearsons), "pvalue:", sum(pvalue_pearsons)/len(pvalue_pearsons))
    # print("spearman:", sum(spearmans)/len(spearmans), "pvalue:", sum(pvalue_spearmans)/len(pvalue_spearmans))

    return sum(pearsons)/len(pearsons), sum(spearmans)/len(spearmans)


def correlation_best(Random=False):
    # compute the number of times the intrisic metric agree to determine the best correction

    improvements_intrinsic, improvements_extrinsic = load_list_improvements()

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

    print("skipped:", skipped, "out of", len(improvements_intrinsic), "times.")
    print("best_agree:", best_agree, "out of", len(improvements_intrinsic), "times aka", best_agree/(len(improvements_intrinsic)-skipped)*100, "%")
    print("disagree:", disagree, "out of", len(improvements_intrinsic), "times.", disagree/(len(improvements_intrinsic)-skipped)*100, "%")

    return best_agree/(len(improvements_intrinsic)-skipped)*100



def correlation_ANR(Random=False):
    # compute the Average Normalized Rank
    anrs = []
    improvements_intrinsic, improvements_extrinsic = load_list_improvements()
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
        
    print("skipped:", skipped, "out of", len(improvements_intrinsic), "times.")
    return sum(anrs)/len(anrs)



# a: rank of correct solution
# b: number of elements
def metric(a, b):
	return 1-(a-1)/(b-1)


def test():
	args = [(1,2), (2,2), (2,100), (98,100), (50,100), (2,4), (4,4), (2,3), (3,4), (4,5)]
	answ = [1, 0, 0.98, 0.02, 0.5, 0.5, 0, 0.3, 0.2, 0.15]
	# args = [(80,100), (20,21), (10,20), (22,33), (1,10), (2,10)]
	# answ = [0.2, 0.05, 0.5, 0.33, 1, 0.9]
	# args = [(50000000,100000000)]
	# answ = [0.5]
	
	for i in range(len(args)):
		a = args[i][0]
		b = args[i][1]
		answer = metric(a, b)
		print(str(a) + "/" + str(b) + " = " + str(answer) + " (real answer == " + str(answ[i]) + ")")



if __name__ == '__main__':
    
    # dataset = read_hats()
    # translate_hats_and_save(dataset)
    # save_semdist_bertscore(verbose=False)
    # compute_correlation_intrinsic_extrinsic()
    # correct_and_save()
    # dataset = load_corrected_hats()
    # load_only_improvements()
    # compute_correlation_minED_extrinsic()
    # correlation_minED_extrinsic_local()


    # test()
    # exit()

    anrs = []
    for i in range(100):
        print(i)
        anrs.append(correlation_ANR(Random=True))
    print(sum(anrs)/len(anrs))


    exit(-1)

    # random test
    random_pearsons = []
    random_spearmans = []
    for i in range(100):
        print(i)
        pearson, spearman = correlation_minED_extrinsic(Random=True)
        random_pearsons.append(pearson)
        random_spearmans.append(spearman)
    print(sum(random_pearsons)/len(random_pearsons))
    print(sum(random_spearmans)/len(random_spearmans))
    

    exit(-1)
   
    
    # random test
    random_scores = []
    for i in range(1):
        print(i)
        random_scores.append(correlation_best(Random=False))
    print(sum(random_scores)/len(random_scores))

    exit(-1)
    
    random_pearsons = []
    random_spearmans = []
    for i in range(20):
        print(i)
        pearson, spearman = correlation_minED_extrinsic_local(signif=0.10, Random=True)
        random_pearsons.append(pearson)
        random_spearmans.append(spearman)
    print(sum(random_pearsons)/len(random_pearsons))
    print(sum(random_spearmans)/len(random_spearmans))