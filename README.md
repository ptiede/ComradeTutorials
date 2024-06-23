# ComradeTutorials


## Getting Started

To run these tutorials you will need to install Julia onto your system and add the `Pluto.jl` package. 
To install Julia we recommend using [juliaup](https://github.com/JuliaLang/juliaup) and following the installation 
instructions there. Please use the most recent Julia version which of this writing is 1.10.4.

To install [Pluto](https://plutojl.org/), first enter the Julia REPL by typing `julia` into your terminal. To install Pluto, please enter
Pkg mode by pressing `]` and then typing `add Pluto`. This should look similar to

```julia
julia> ]
pkg> add Pluto
```

To launch the notebook then exit `Pkg` model by pressing backspace and then type

```julia
julia> using Pluto
julia> Pluto.run()
```

to launch the Pluto server in your default browser. 
