# Head First Taichi: A Beginner's Guide to High Performance Computing in Python

Ever since the Python programming language was born, its core philosophy has always been to maximize the readability and simplicity of code. In fact, the reach for readability and simplicity is so deep within Python's root, that if you type `import this` in a Python console, it will recite a little poem:

```
Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
...
```

Simple is better than complex. Readability counts. No doubt, Python has indeed been quite successful at these achieving these goals: it is by far the most friendly language to learn, and an average Python program is often [5-10 times shorter](https://www.python.org/doc/essays/comparisons/) than equivalent C++ code. Unfortunately, there is a catch: Python's simplicity comes at the cost of reduced performance. It is never surprising for a Python program to be [10-100 times slower](https://benchmarksgame-team.pages.debian.net/benchmarksgame/fastest/gpp-python3.html) than its C++ counterpart. It thus appears that there is a perpetual trade-off between speed and simplicity, and no programming language shall ever possess both.

But don't you worry, all hope is not lost.

## Taichi: best of both worlds

The [Taichi Programming Language](https://github.com/taichi-dev/taichi) is an attempt to extend the Python programming language with constructs that enables general purpose, high performance computing. It is seamlessly embedded in Python, yet it can summon every ounce of computing power in a machine -- the multi-core CPU, and more importantly, the GPU.

The following gif shows an example program written using taichi. The program uses the GPU to run a real-time physical simulation of a piece of cloth falling onto a sphere, and simultaneously renders the result. 

<p align="center">
  <img width="400" height="400" src="https://github.com/AmesingFlank/taichi/raw/blog_0/blog_0/cloth.gif">
</p>


Writing a real-time GPU physics simulator is rarely an easy task, but the Taichi source code behind this program is surprisingly simple. The remainder of this article will walk you through the entire implementation, so you can get a taste of the functionalities that taichi provides, and just how powerful and friendly they are.

Before we begin, take a guess of how many lines of code does this program consist of. You will find the answer at the end of the article.

### Getting Started


Although Taichi is a programming language in its own right, it exists in the form of a Python package, and can be installed by simply running `pip install taichi`. 

