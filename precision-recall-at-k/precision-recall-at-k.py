def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    # Write code here
    precision = len([i for i in recommended[:k] if i in relevant])/k
    recall = len([i for i in recommended[:k] if i in relevant])/len(relevant)
    return [precision, recall]