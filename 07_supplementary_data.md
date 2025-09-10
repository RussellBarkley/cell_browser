---
Chapter 8
---

# Supplementary Data

---

# Equations

ReLU
$$
\quad 
f(x)=\max(0,x)
$$

Sigmoid
$$
\quad 
\sigma(x)=\frac{1}{1+e^{-x}}
$$

Linear
$$
\quad 
f(x)=x
$$

Mean Squared Error
$$
\quad
\operatorname{MSE}(\{ \mathbf{y}_i\}, \{\hat{\mathbf{y}}_i\}) 
= \frac{1}{n}\sum_{i=1}^{n}\lVert \mathbf{y}_i-\hat{\mathbf{y}}_i\rVert_2^{2}
$$

Data
$$
\quad
X_\ell \;=\;\{\,x_i \in [0,1]^d\,\}_{i=1}^{N_\ell},\quad d=28\times 28=784
$$

Arithmetic mean
$$
\label{equation_A}
\quad
\mu_\ell \;=\;\frac{1}{N_\ell}\sum_{i=1}^{N_\ell} x_i
\;\;\in\; \mathbb{R}^d
$$

Median
$$
\label{equation_B}
\quad
m_\ell[j] \;\in\; \underset{z\in\mathbb{R}}{\arg\min}\;
\sum_{i=1}^{N_\ell} \lvert z - x_i[j]\rvert,\qquad j=1,\dots,d,
\quad\text{and}\quad
m_\ell=\big(m_\ell[1],\dots,m_\ell[d]\big)\in\mathbb{R}^d
$$

Geometric median
$$
\label{equation_C}
\quad
g_\ell \;\in\; \underset{x\in\mathbb{R}^d}{\arg\min}\;\sum_{i=1}^{N_\ell}\big\lVert x - x_i\big\rVert_2
$$