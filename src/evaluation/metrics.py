#this folder is used to compare the answers of the model and the actual correct answers.

from sklearn.metrics import precision_score, recall_score, f1_score

def compute_metrics(y_true,y_pred): #y_true is the ground truth labels and y_pred is the predicted labels. 
                                    #NOTE that 1 - duplicate, 0 - not duplicate.

    #out of all the duplicate images, how many were actually duplicated is handled by precision.
    #precision is the true positive/true +ve + false +ve so the proportion of 
    #correct positive predictions out of all the positive predictions made by the model
    precision = precision_score(y_true,y_pred)

    #out of all the duplicate images, how much did the system correctly detect is handled by recall
    #recall is the true +ve / true +ve + false -ve, ratio of true positives to the total actual positives.
    recall = recall_score(y_true,y_pred)

    #harmonic mean of precision and recall, balances false positives and negatives.
    #why HM? to make sure that the imabalances in the precision and recall can be handled properly.
    #it also makes sure by telling if the precision and recall is good(high) then only the f1 score is good making the 
    #f1 score more accurate
    f1 = f1_score(y_true,y_pred)
    return precision,recall,f1