---
Chapter 3
---

You can then embed interactive widget cell outputs inside .ipynb files under `notebooks` using the following notation:

This is a sentence.

This is a citation [@doi:10.1091/mbc.E24-01-0013]

RI

Here is a wiki article on [representative images](<wiki:Representative_image>)

Here is an equation 
$$
\label{equation_A}
A=b+c
$$

Reference to [Eq1](#equation_A)

````
:::{figure} #app:example_widget
:name: fig_example_widget
:placeholder: ./figures/example_widget_placeholder.png
Example widget.
:::
````

:::{figure} #app:example_widget
:name: fig_example_widget
:placeholder: ./figures/example_widget_placeholder.png
Example widget.
:::

