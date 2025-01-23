def get_train_val_masks(X, train_end, val_start, val_end):
    """
    Generate boolean masks for training and validation based on date cutoffs.
    
    Arguments:
        X          : Features DataFrame containing 'year', 'month', 'day'
        train_end  : Tuple (year, month, day) inclusive cutoff for training
        val_start  : Tuple (year, month, day) inclusive start for validation
        val_end    : Tuple (year, month, day) inclusive end for validation
        
    Returns:
        train_mask : Boolean Series (True = row belongs in training)
        val_mask   : Boolean Series (True = row belongs in validation)
    """
    # Unpack the cutoff tuples
    y_te, m_te, d_te = train_end   # Train end date
    y_vs, m_vs, d_vs = val_start   # Validation start date
    y_ve, m_ve, d_ve = val_end     # Validation end date
    
    # Training mask: up to train_end (inclusive)
    train_mask = (
        (X['year'] < y_te)
        | (
            (X['year'] == y_te)
            & (
                (X['month'] < m_te)
                | (
                    (X['month'] == m_te) 
                    & (X['day'] <= d_te)
                )
            )
        )
    )
    
    # Validation mask: between val_start and val_end (inclusive)
    val_mask = (
        (X['year'] == y_vs)
        & (X['month'] == m_vs)
        & (X['day'] >= d_vs)
        & (X['day'] <= d_ve)   # same month & year assumption in this example
    )
    
    return train_mask, val_mask