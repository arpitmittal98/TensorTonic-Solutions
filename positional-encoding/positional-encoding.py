import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(0, (d_model + 1) // 2)
    div_term = 1 / (base ** (2 * i / d_model))

    # 3. Create the matrix of angles (T, ceil(d_model/2))
    angles = pos * div_term
    
    # 4. Initialize the PE matrix
    pe = np.zeros((seq_len, d_model), dtype=np.float64)
    
    # 5. Fill alternating columns
    pe[:, 0::2] = np.sin(angles)
    
    # Only fill cosine if there's an odd slot available to match it
    # This naturally handles the "last column is sin" requirement for odd d_model
    pe[:, 1::2] = np.cos(angles[:, :d_model // 2])
    
    return pe