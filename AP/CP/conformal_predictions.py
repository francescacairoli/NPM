import numpy as np
from numpy.random import rand

def compute_NN_nonconformity_scores(labels, pred_probs, sorting = True, dictionary = False):
    # return the calibration scores sorted in descending order
    q = len(labels)
    alphas = np.abs(labels-pred_probs.reshape((q,)))
    
    if sorting:
        alphas = np.sort(alphas)[::-1] # descending order

    alphas_dict = {}
    if dictionary:
        for a in alphas:
            alphas_dict[a] = alphas_dict.get(a, 0) + 1
        
    return alphas, alphas_dict #unsorted

def compute_SVC_nonconformity_scores(x, y, svc,  sorting = True, dictionary = False):

    # return the calibration scores sorted in descending order
    q = len(y)

    A = 1

    w = svc.dual_coef_ 
    w_norm = np.linalg.norm(w[0])

    # 'ovo' = decision function values are proportional to the distance of samples X to the separating hyperplane
    svc_output = svc.decision_function(x) 

    #margin_distance = (np.abs(svc_output-y))/w_norm ---> WRONG
    
    margin_distance = np.zeros(q)
    for i in range(q):
        if y[i] == 1:
            margin_distance[i] = svc_output[i]-y[i]
        else:
            margin_distance[i] = -svc_output[i]+y[i]
    margin_distance = margin_distance/w_norm
    alphas = np.exp(-A*margin_distance)

    if sorting:
        alphas = np.sort(alphas)[::-1] # descending order

    alphas_dict = {}
    if dictionary:  
        for a in alphas:
            alphas_dict[a] = alphas_dict.get(a, 0) + 1
        
    return alphas, alphas_dict


def compute_p_values(alphas, pred_probs = None, x= None, cal_labels = None, mondrian = False, svc = None, classifier = "NN", class_dict = {"class_pos": 1, "class_neg": 0}):
    '''
    alphas: non conformity measures sorted in descending order
    pred_probs: class 1 prediction probabilities on the points considered
    classifier: "NN" for neural networks with single unit output
                "SVC" for support vector classifier
    class_dict: specify the label used to distinguish the two classes (pos: positive and neg: negative)
    return: 2-dim array containing values of confidence and credibility (which are not exactly the p-values)
            shape = (n_points,2)
    
    TODO: reimplement this method using dictionaries and check if the computational time descreases
    '''
    q = len(alphas)
    if mondrian:
        alphas_pos = alphas[(cal_labels==class_dict["class_pos"])]
        alphas_neg = alphas[(cal_labels==class_dict["class_neg"])]
        q_pos = alphas_pos.shape[0]
        q_neg = alphas_neg.shape[0]

    else:
        alphas_pos = alphas
        alphas_neg = alphas
        q_pos = q
        q_neg = q
    
    if classifier=="NN":
        n_points = len(pred_probs)
        A_pos, _ = compute_NN_nonconformity_scores(class_dict["class_pos"]*np.ones(n_points), pred_probs, sorting = False, dictionary = False)
        A_neg, _ = compute_NN_nonconformity_scores(class_dict["class_neg"]*np.ones(n_points), pred_probs, sorting = False, dictionary = False)
    elif classifier=="SVC":
        n_points = x.shape[0]
        A_pos, _ = compute_SVC_nonconformity_scores(x, class_dict["class_pos"]*np.ones(n_points), svc,  sorting = False, dictionary = False)
        A_neg, _ = compute_SVC_nonconformity_scores(x, class_dict["class_neg"]*np.ones(n_points), svc,  sorting = False, dictionary = False)

    else:
        print("WARNING: Unsupported classifier!")
    
    p_pos = np.zeros(n_points)
    p_neg = np.zeros(n_points)
    for k in range(n_points):
        c_pos_a = 0
        c_pos_b = 0
        c_neg_a = 0
        c_neg_b = 0
        for count_pos in range(q_pos):
            if alphas_pos[count_pos] > A_pos[k]:
                c_pos_a += 1
            elif alphas_pos[count_pos] == A_pos[k]:
                c_pos_b += 1
            else:
                break
        for count_neg in range(q_neg):
            if alphas_neg[count_neg] > A_neg[k]:
                c_neg_a += 1
            elif alphas_neg[count_neg] == A_neg[k]:
                c_neg_b += 1
            else:
                break
        p_pos[k] = ( c_pos_a + rand() * (c_pos_b + 1) ) / (q_pos + 1)
        p_neg[k] = ( c_neg_a + rand() * (c_neg_b + 1) ) / (q_neg + 1)

        
    confidence_credibility = np.zeros((n_points,2))
    for i in range(n_points):
        if p_pos[i] > p_neg[i]:
            confidence_credibility[i,0] = 1-p_neg[i]
            confidence_credibility[i,1] = p_pos[i]
        else:
            confidence_credibility[i,0] = 1-p_pos[i]
            confidence_credibility[i,1] = p_neg[i]
    if mondrian:
        pvalues = np.zeros((2,n_points))
        pvalues[0] = p_neg
        pvalues[1] = p_pos
        return pvalues 
    else:
        return confidence_credibility

def compute_p_values_dict(alphas_dict, pred_probs = None, x= None, svc = None, classifier = "NN", class_dict = {"class_pos": 1, "class_neg": 0}):
    q = len(alphas_dict)
    n_test = len(pred_probs)

    if classifier=="NN":
        A_pos, _ = compute_NN_nonconformity_scores(class_dict["class_pos"]*np.ones(n_test), pred_probs, sorting = False, dictionary = False)
        A_neg, _ = compute_NN_nonconformity_scores(class_dict["class_neg"]*np.ones(n_test), pred_probs, sorting = False, dictionary = False)
    elif classifier=="SVC":
        A_pos, _ = compute_SVC_nonconformity_scores(x, class_dict["class_pos"]*np.ones(n_test), svc,  sorting = False, dictionary = False)
        A_neg, _ = compute_SVC_nonconformity_scores(x, class_dict["class_neg"]*np.ones(n_test), svc,  sorting = False, dictionary = False)
    else:
        print("WARNING: Unsupported classifier!")


    gamma = np.zeros(n_test)
    credibility = np.zeros(n_test)
    start = time.time()
    for i in range(n_test):
        c_pos_1 = np.sum([v for k, v in alpha_dict.items() if k > A_pos[i]])
        c_pos_2 = np.sum([v for k, v in alpha_dict.items() if k == A_pos[i]])

        p_pos = (c_pos_1 + rand() * (c_pos_2 + 1) )/ (q + 1)

        c_neg_1 = np.sum([v for k, v in alpha_dict.items() if k > A_neg[i]])
        c_neg_2 = np.sum([v for k, v in alpha_dict.items() if k == A_neg[i]])
        
        p_neg = (c_neg_1 + rand() * (c_neg_2 + 1) )/ (q + 1)

        if p_pos > p_neg:
            credibility[i] = p_pos
            gamma[i] = p_neg
        else:
            credibility[i] = p_neg
            gamma[i] = p_pos

    end = (time.time()-start)
    confidence_credibility = np.zeros((n_test,2))
    confidence_credibility[:,0] = 1-gamma
    confidence_credibility[:,1] = credibility
    return confidence_credibility
    
def compute_calibration_p_values_dict(alphas, alphas_dict, cal_pred_probs, classifier = "NN", class_dict = {"class_pos": 1, "class_neg": 0}):
    q = len(alphas)
    
    if classifier=="NN":
        A_pos, _ = compute_NN_nonconformity_scores(class_dict["class_pos"]*np.ones(q), cal_pred_probs, sorting = False, dictionary = False)
        A_neg, _ = compute_NN_nonconformity_scores(class_dict["class_neg"]*np.ones(q), cal_pred_probs, sorting = False, dictionary = False)
    elif classifier=="SVC":
        A_pos, _ = compute_SVC_nonconformity_scores(x_cal, class_dict["class_pos"]*np.ones(q), svc,  sorting = False, dictionary = False)
        A_neg, _ = compute_SVC_nonconformity_scores(x_cal, class_dict["class_neg"]*np.ones(q), svc,  sorting = False, dictionary = False)
    else:
        print("WARNING: Unsupported classifier!")

    gamma = np.zeros(q)
    credibility = np.zeros(q)
    start = time.time()
    for i in range(q):
        tmp_dict = alphas_dict
        tmp_dict[alphas[i]] -= 1
        c_pos_1 = np.sum([v for k, v in alpha_dict.items() if k > A_pos[i]])
        c_pos_2 = np.sum([v for k, v in alpha_dict.items() if k == A_pos[i]])
        
        p_pos = (c_pos_1 + rand() * (c_pos_2 + 1) )/ (q + 1)

        c_neg_1 = np.sum([v for k, v in alpha_dict.items() if k > A_neg[i]])
        c_neg_2 = np.sum([v for k, v in alpha_dict.items() if k == A_neg[i]])
        
        p_neg = (c_neg_1 + rand() * (c_neg_2 + 1) )/ (q + 1)

        if p_pos > p_neg:
            credibility[i] = p_pos
            gamma[i] = p_neg
        else:
            credibility[i] = p_neg
            gamma[i] = p_pos

    end = (time.time()-start)
    confidence_credibility = np.zeros((n_test,2))
    confidence_credibility[:,0] = 1-gamma
    confidence_credibility[:,1] = credibility
    return confidence_credibility
    
def  compute_calibration_p_values(alphas, cal_pred_probs = None, x_cal= None, cal_labels = None, mondrian = False, svc = None, classifier = "NN", class_dict = {"class_pos": 1, "class_neg": 0}):
    '''
	alphas: non conformity measures sorted in descending order
	cal_pred_probs: class 1 prediction probabilities on points of the calibration set 

    return: 2-dim array containing values of confidence and credibility (which are not exactly the p-values)
    		shape = (n_points,2) -- CROSS VALIDATION STRATEGY
    TODO: reimplement this method using dictionaries and check if the computational time descreases
	'''
    q = len(alphas) # number of calibration scores

    if classifier=="NN":
        A_pos, _ = compute_NN_nonconformity_scores(class_dict["class_pos"]*np.ones(q), cal_pred_probs, sorting = False, dictionary = False)
        A_neg, _ = compute_NN_nonconformity_scores(class_dict["class_neg"]*np.ones(q), cal_pred_probs, sorting = False, dictionary = False)
    elif classifier=="SVC":
        A_pos, _ = compute_SVC_nonconformity_scores(x_cal, class_dict["class_pos"]*np.ones(q), svc,  sorting = False, dictionary = False)
        A_neg, _ = compute_SVC_nonconformity_scores(x_cal, class_dict["class_neg"]*np.ones(q), svc,  sorting = False, dictionary = False)
    else:
        print("WARNING: Unsupported classifier!")

    if mondrian:
        alphas_pos = alphas[(cal_labels==class_dict["class_pos"])]
        alphas_neg = alphas[(cal_labels==class_dict["class_neg"])]
        q_pos = alphas_pos.shape[0]
        q_neg = alphas_neg.shape[0]
    else:
        alphas_pos = alphas
        alphas_neg = alphas
        q_pos = q
        q_neg = q


    p_pos = np.zeros(q) # p_values for class 1
    p_neg = np.zeros(q) # p_values for class 0
    pos_counter = 0
    neg_counter = 0
    if mondrian:
        for k in range(q):
            if cal_labels[k] == class_dict["class_pos"]:
                alphas_pos_mask = alphas_pos
                alphas_pos_mask = np.delete(alphas_pos_mask,pos_counter)
                pos_counter += 1
                alphas_neg_mask = alphas_neg
            else:
                alphas_neg_mask = alphas_neg
                alphas_neg_mask = np.delete(alphas_neg_mask,neg_counter)
                neg_counter += 1
                alphas_pos_mask = alphas_pos
            c_pos_1 = 0
            c_pos_2 = 0


            for count_pos in range(len(alphas_pos_mask)):
                if alphas_pos_mask[count_pos] > A_pos[k]:
                    c_pos_1 += 1
                elif alphas_pos_mask[count_pos] == A_pos[k]:
                    c_pos_2 += 1
                else:
                    break
            p_pos[k] = (c_pos_1 + rand()*(c_pos_2+1) )/(len(alphas_pos_mask)+1)


            c_neg_1 = 0
            c_neg_2 = 0
            for count_neg in range(len(alphas_neg_mask)):
                if alphas_neg_mask[count_neg] > A_neg[k]:
                    c_neg_1 += 1
                elif alphas_neg_mask[count_neg] == A_neg[k]:
                    c_neg_2 += 1
                else:
                    break
            p_neg[k] = (c_neg_1 + rand()*(c_neg_2+1) )/(len(alphas_neg_mask)+1)
    else:        
        for k in range(q):
            alphas_mask = alphas
            alphas_mask = np.delete(alphas_mask,k)
            c_pos_1 = 0
            c_pos_2 = 0
            c_neg_1 = 0
            c_neg_2 = 0
            for count_pos in range(q_pos-1):
                if alphas_mask[count_pos] > A_pos[k]:
                    c_pos_1 += 1
                elif alphas_mask[count_pos] == A_pos[k]:
                    c_pos_2 += 1
                else:
                    break

            for count_neg in range(q_neg-1):
                if alphas_mask[count_neg] > A_neg[k]:
                    c_neg_1 += 1
                elif alphas_mask[count_neg] == A_neg[k]:
                    c_neg_2 += 1
                else:
                    break
            
            p_pos[k] = (c_pos_1 + rand()*(c_pos_2+1) )/q
            p_neg[k] = (c_neg_1 + rand()*(c_neg_2+1) )/q

    confidence_credibility = np.zeros((q,2))
    for i in range(q):
    	if p_pos[i] > p_neg[i]:
    		confidence_credibility[i,0] = 1-p_neg[i]
    		confidence_credibility[i,1] = p_pos[i]
    	else:
    		confidence_credibility[i,0] = 1-p_pos[i]
    		confidence_credibility[i,1] = p_neg[i]
    if mondrian:
        return confidence_credibility, p_pos, p_neg
    else:
        return confidence_credibility
