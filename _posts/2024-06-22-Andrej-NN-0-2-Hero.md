# Learnings from Andrej's NN 0 to Hero series

## Self Attention

Suppose this is your list of tokens

```[13, 42, 65, 21, 0, 78]```

Learning from history meaning,

$$ L_i = Avg(L_{0,i}) $$

Learned attention at $L_i$ is average of previous all context $Avg(L_{0,i})$.

For token 65 in above list,

learned attention weights of

```65 = Avg[13, 42, 65]```

**Note:** Mathematically to make these calculations faster parallely, we can utilize traiangular matrix multiplication followed by average.
