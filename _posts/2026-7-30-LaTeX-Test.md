---
layout: post
title: LaTeX Test — Math Rendering Playground
tags: [math]
math: true
---

This post verifies the MathJax setup and doubles as a template for future math posts.

## Inline math

The famous identity $e^{i\pi} + 1 = 0$ connects five fundamental constants.
Given a matrix $A \in \mathbb{R}^{m \times n}$, its spectral norm is $\|A\|_2 = \sigma_{\max}(A)$.

## Display math

$$
\int_{-\infty}^{\infty} e^{-x^2} \, dx = \sqrt{\pi}
$$

$$
\nabla \times \mathbf{B} = \mu_0 \mathbf{J} + \mu_0 \varepsilon_0 \frac{\partial \mathbf{E}}{\partial t}
$$

## Aligned equations

$$
\begin{aligned}
f(x) &= (x+1)^2 \\
     &= x^2 + 2x + 1
\end{aligned}
$$

## Matrix

$$
\begin{pmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{pmatrix}
\begin{pmatrix} x \\ y \end{pmatrix}
=
\begin{pmatrix} x\cos\theta - y\sin\theta \\ x\sin\theta + y\cos\theta \end{pmatrix}
$$

## Long formula (horizontal scroll on mobile)

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V, \quad Q \in \mathbb{R}^{n \times d_k},\; K \in \mathbb{R}^{m \times d_k},\; V \in \mathbb{R}^{m \times d_v}
$$

To enable math on a post, add `math: true` to its front matter.
