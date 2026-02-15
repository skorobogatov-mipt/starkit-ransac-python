# Model parameters
## x
This parameter represents the x coordinate of the point.
## y
This parameter represents the y coordinate of the point.
## z
This parameter represents the z coordinate of the point.

# Model fitting
Let $X$ be the matrix that is passed to `fit_model`.
This matrix' dimensions are $1 \times 3$.

To fit a point to $X$, we just equalize all the model parameters to this
point's coordinates:
$$
\begin{cases}
x = X_{11}\\
y = X_{12}\\
z = X_{13}\\
\end{cases}
$$

# Distance evaluation
To evaluate distance from the fitted model, Euclidean distance is used.
Let $\overline{p} = (x, y, z)^T$ be the representation of our model. Let $p$
be a point that we want to evaluate distance to. Then, distance from $p$ to
our model is calculated like this:
$$
d = \|\overline{p} - p\|_2
$$

