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

### Algorithmic Overview
Before we get to coding, let's take a small detour to review the algorithmic side of our cloth simulator. Our program will model the piece of cloth as a mass-spring system. More specifically, we will represent the pice of cloth as a $N$ by $N$ grid of point-masses, where adjacent points are linked by a spring. The following image illustrates this structure:

<p align="center">
  <img width="400" height="300" src="https://graphics.stanford.edu/~mdfisher/TutorialData/ClothSag.png">
</p>

The motion of this mass-spring system is affected by 4 factors:

* Gravity
* Collision with the red ball in the middle
* Internal forces of the springs
* Damping

Our program begins at time $t=0$. Then, at each step of the simulation, it advances the time by a small constant $dt$. The program estimates what happens to the system in this small period of time by evaluating the effect of each of the 4 factors above, and updates the the position and velocity of each mass point at the end of the timestep. The updated positions of mass points are then used to update the image rendered on screen.


### Getting Started
Although Taichi is a programming language in its own right, it exists in the form of a Python package, and can be installed by simply running
```
pip install taichi
```
To start using Taichi in a python program, import it under the alias `ti`:
```python
import taichi as ti
```
The performance of a Taichi program is maximized if your machine has a CUDA-enabled nVidia GPU. If this is case, add the following line of code after the import:
```python
ti.init(ti.cuda)
```
If you don't have a CUDA GPU, Taichi can still interact with your GPU via other graphics APIs, such as `ti.metal`, `ti.vulkan`, and `ti.opengl`. However, Taichi's support for these APIs are not as complete as its CUDA support, so for now, use the CPU backend: 
```python
ti.init(ti.cpu)
```
And don't worry, taichi is blazing fast even if it only runs on the CPU.

Having initialized taichi, we can start declaring the data structures used to describe the mass-spring cloth. We add the following line of code:

```python
N = 128
x = ti.Vector.field(3, float, (N, N))
v = ti.Vector.field(3, float, (N, N))
```


### WIP





Gravity is the simplest of these factors: each point mass in the system is constantly subjected to a downwards force, whose magnitude is fixed. For simplicity, we treat the mass of each point to be 1 unit, and we take gravity to be 0.5 units. This means that gravity causes the $y$ component of the velocity of each point to decrease at the rate of 0.5 units.

As the cloth moves down, it will eventually come into contact with the red ball in the middle. We will use a simple model to represent this collision: for each mass point, as soon as it hits the ball, it "sticks" there and stops moving.

Each spring $s$ in the system is initialized with a rest length, $l_0(s)$, and at any moment $t$, if the current length $l_t(s)$ of $s$ exceeds $l_0(s)$, the spring will exert a force on its endpoints that pulls them together, where the magnitude of the force is proportional to $l_t(s)-l_0(s)$. Conversely, if $l_t(s)$ is smaller than $l_0(s)$, then the spring will push the endpoints away from each other, with a force proportional to $l_0(s)-l_t(s)$. If you remember high school physics, this is called Hooke's law.

Finally, to prevent this system from falling into perpetual and chaotic motion, we will damp the velocity of each mass point. This simply means that we slightly reduce the magnitude of its velocity at each timestep.

