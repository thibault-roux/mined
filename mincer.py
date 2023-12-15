import progressbar
import aligned_wer as awer
import numpy
import pickle

# problem: in a previous version, the computation of graph can be very expensive, without considering the metric cost
# with 40 errors (which happens), we have 10^12 nodes.

# instead, we would like to compute to compute only the next level

# for scores, I should store them in a file with 'ref \t hyp \t score \n', store them in a set and rewrite the file every time



# DONE - 1) compute the number of errors in the dataset
# 2) compute the partial graph when computations is too excessive
# 3) reformulate the problem with boolean vector



def get_next_level(prev_level):
    # Will compute all possibilities for the next level of the graph.

    # INPUT: 
    #   prev_level = {001, 010, 000}
    # OUTPUT: 
    #   level == {101, 011, 110}

    level = set()
    for errors in prev_level:
        errors = list(errors) # string to list for item assigment
        for i in range(len(errors)):
            error = errors[i]
            if error == '0':
                new_errors = errors.copy()
                new_errors[i] = 1
                level.add(''.join(str(x) for x in new_errors)) # add list (converted to string)
    return level

def correcter_mincer(ref, hyp, corrected, errors):
    # ref, hyp, corrected (100), errors (deesei)

    # ref = ref.split(" ")
    # hyp = hyp.split(" ")
    INDEX = 0

    new_hyp = ""
    ir = 0
    ih = 0
    for i in range(len(errors)):
        if errors[i] == "e": # already
            new_hyp += ref[ir]
            ih += 1
            ir += 1
            # print("e\t", new_hyp)
        elif errors[i] == "i": # insertion corrected
            if corrected[INDEX] == '0': # if we do not correct the error
                new_hyp += hyp[ih] # the extra word is not deleted
            ih += 1
            INDEX += 1
            # print("i\t", new_hyp)
        elif errors[i] == "d": # deletion
            if corrected[INDEX] == '1': # if we do correct the error
                new_hyp += ref[ir] # we add the missing word
            # else  # we do not restaure the missing word
            ir += 1
            INDEX += 1
            # print("d\t", new_hyp)
        elif errors[i] == "s": # substitution
            if corrected[INDEX] == '1':
                new_hyp += ref[ir]
            else:
                new_hyp += hyp[ih] # we do not correct the substitution 
            ih += 1
            ir += 1
            INDEX += 1
            # print("s\t", new_hyp)
        else: 
            print("Error: the newhyp inputs 'errors' and 'new_errors' are expected to be string of e,s,i,d. Received", errors[i])
            exit(-1)
        i += 1
    return new_hyp

def bertscore(ref, hyp, memory):
    scorer = memory
    P, R, F1 = scorer.score([hyp], [ref])
    return 1-F1

def semdist(ref, hyp, memory):
    model = memory
    ref_projection = model.encode(ref).reshape(1, -1)
    hyp_projection = model.encode(hyp).reshape(1, -1)
    score = cosine_similarity(ref_projection, hyp_projection)[0][0]
    return (1-score) # lower is better

def wer(ref, hyp, memory):
    return jiwer.wer(ref, hyp)

def MinCER(ref, hyp, metric, threshold, save, memory):
    __MAX__ = 15 # maximum distance to avoid too high computational cost
    errors, distance = awer.wer(ref, hyp)
    base_errors = ''.join(errors)
    level = {''.join(str(x) for x in [0]*distance)}
    # base_errors = ['esieed']
    # distance = 3
    # level = {000}
    if distance <= __MAX__: # to limit the size of graph
        mincer = 0
        while mincer < distance:
            for node in level:
                corrected_hyp = correcter_mincer(ref, hyp, node, base_errors)
                # optimization to avoid recomputation
                try:
                    score = save[ref][corrected_hyp]
                except KeyError:
                    score = metric(ref, corrected_hyp, memory)
                    if ref not in save:
                        save[ref] = dict()
                    save[ref][corrected_hyp] = score
                if score < threshold: # lower-is-better
                    return mincer
            level = get_next_level(level)
            mincer += 1
        return distance
    else:
        return distance
        



def read_dataset(dataname):
    # dataset = [{"reference": ref, "hypA": hypA, "nbrA": nbrA, "hypB": hypB, "nbrB": nbrB}, ...]
    dataset = []
    with open("datasets/" + dataname, "r", encoding="utf8") as file:
        next(file)
        for line in file:
            line = line[:-1].split("\t")
            dictionary = dict()
            dictionary["reference"] = line[0]
            dictionary["hypA"] = line[1]
            dictionary["nbrA"] = int(line[2])
            dictionary["hypB"] = line[3]
            dictionary["nbrB"] = int(line[4])
            dataset.append(dictionary)
    return dataset

def evaluator(metric, dataset, threshold, memory, picklename_metric, certitude=0.7, verbose=True):
    ignored = 0
    accepted = 0
    correct = 0
    incorrect = 0
    egal = 0



    """
    from termcolor import colored
    class bcolors:
        BLUE    = '\033[94m'
        RED     = '\033[91m'
        YELLOW  = '\033[93m'
        ENDC    = '\033[0m'
    from jiwer import cer
    """



    # recover scores save
    try:
        with open(picklename_metric, "rb") as handle:
            save = pickle.load(handle)
    except FileNotFoundError:
        save = dict()

    if verbose:
        bar = progressbar.ProgressBar(max_value=len(dataset))
    for i in range(len(dataset)):
        if verbose:
            bar.update(i)
        nbrA = dataset[i]["nbrA"]
        nbrB = dataset[i]["nbrB"]
        
        if nbrA+nbrB < 5:
            ignored += 1
            continue
        maximum = max(nbrA, nbrB)
        c = maximum/(nbrA+nbrB)
        if c >= certitude: # if humans are certain about choice
            accepted += 1
            scoreA = MinCER(dataset[i]["reference"], dataset[i]["hypA"], metric, threshold, save, memory)
            scoreB = MinCER(dataset[i]["reference"], dataset[i]["hypB"], metric, threshold, save, memory)

            """
            cerA = int(cer(dataset[i]["reference"], dataset[i]["hypA"])*len(dataset[i]["reference"]))
            cerB = int(cer(dataset[i]["reference"], dataset[i]["hypB"])*len(dataset[i]["reference"]))

            # check if stateCER > stateMinCER
            stateCER = 0 # 0 = incorrect, 1 = egal, 2 = correct
            stateMinCER = 0 # 0 = incorrect, 1 = egal, 2 = correct
            
            if (scoreA < scoreB and nbrA > nbrB) or (scoreB < scoreA and nbrB > nbrA):
                stateMinCER = 2
            elif scoreA == scoreB:
                stateMinCER = 1
            else:
                stateMinCER = 0

            if (cerA < cerB and nbrA > nbrB) or (cerB < cerA and nbrB > nbrA):
                stateCER = 2
            elif cerA == cerB:
                stateCER = 1
            else:
                stateCER = 0

            if stateMinCER > stateCER:
                # good for minCER
                print(bcolors.BLUE)
            elif stateMinCER == stateCER:
                # no progress
                print(bcolors.YELLOW)
            elif stateMinCER < stateCER:
                # no good
                print(bcolors.RED)
            else:
                # unexpected error
                print(1/0)
                exit(-1)
            
            print(dataset[i]["reference"])
            print(nbrA, scoreA, cerA, dataset[i]["hypA"])
            print(nbrB, scoreB, cerB, dataset[i]["hypB"])
            print(bcolors.ENDC)
            input()
            """


            if (scoreA < scoreB and nbrA > nbrB) or (scoreB < scoreA and nbrB > nbrA):
                correct += 1
            elif scoreA == scoreB:
                egal += 1
            else:
                incorrect += 1
            continue
        else:
            ignored += 1
    # storing scores save
    with open(picklename_metric, "wb") as handle:
        pickle.dump(save, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print()
    print("correct:", correct)
    print("egal:", egal)
    print("incorrect:", incorrect)
    return correct, egal, incorrect

def write(namefile, threshold, x, y):
    with open("results/MINCER/" + namefile + ".txt", "a", encoding="utf8") as file:
        file.write(namefile + "," + str(threshold) + "," + str(x) + "," + str(y) + "\n")

if __name__ == '__main__':
    dataset = read_dataset("hats.txt")

    
    # choice = "wer"
    # choice = "bertscore"
    # choice = "bertscore_rescale"
    # choice = "SD_sent_camembase"
    choice = "SD_sent_camemlarge"
    

    if choice == "wer":
        import jiwer
        memory = 0
        metric = wer
        picklename_metric = "pickle/wer.pickle"
    elif choice == "bertscore":
        from bert_score import BERTScorer
        memory = BERTScorer(lang="fr")
        metric = bertscore
        picklename_metric = "pickle/bertscore.pickle"
    elif choice == "bertscore_rescale":
        from bert_score import BERTScorer
        memory = BERTScorer(lang="fr", rescale_with_baseline=True)
        metric = bertscore
        picklename_metric = "pickle/bertscore_rescale.pickle"
    elif choice == "SD_sent_camembase":
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        model = SentenceTransformer('dangvantuan/sentence-camembert-base')
        memory = model
        metric = semdist
        picklename_metric = "pickle/SD_sent_camembase.pickle"
    elif choice == "SD_sent_camemlarge":
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        model = SentenceTransformer('dangvantuan/sentence-camembert-large')
        memory = model
        metric = semdist
        picklename_metric = "pickle/SD_sent_camemlarge.pickle"
    else:
        raise Exception("Unknown choice: ", choice)
    
    print()    
    
    #for threshold in [0.005, 0.01, 0.015, 0.025, 0.03]:
    #for threshold in numpy.arange(0.006, 0.015, 0.002):
    for threshold in numpy.arange(0.02, 0.25, 0.008):
        threshold = int(threshold*100000)/100000
        if threshold != 0.01:
            x = evaluator(metric, dataset, threshold, memory, picklename_metric, certitude=1)
            y = evaluator(metric, dataset, threshold, memory, picklename_metric, certitude=0.7)
            write(choice, threshold, x, y)
        
    # to delete
    for threshold in numpy.arange(0.026, 0.07, 0.002):
        threshold = int(threshold*100000)/100000
        if threshold != 0.01:
            x = evaluator(metric, dataset, threshold, memory, picklename_metric, certitude=1)
            y = evaluator(metric, dataset, threshold, memory, picklename_metric, certitude=0.7)
            write(choice, threshold, x, y)
    