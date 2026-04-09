def edit_distance(s1, s2):
    """
    Compute the minimum edit distance between two strings.
    """
    # Write code here
    if len(s1) == 0: return len(s2)
    if len(s2) == 0: return len(s1)

    if s1[-1] == s2[-1]:
        return edit_distance(s1[:-1], s2[:-1])
    
    insert  = edit_distance(s1, s2[:-1])
    delete  = edit_distance(s1[:-1], s2)
    replace = edit_distance(s1[:-1], s2[:-1])

    return 1 + min(insert, delete, replace)