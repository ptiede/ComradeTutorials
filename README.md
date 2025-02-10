# ComradeTutorials


## Getting Started

To run these tutorials you will need to install Julia onto your system and add the `Pluto.jl` package.


### Installing Julia 
Currently we recommend installing [juliaup](https://github.com/JuliaLang/juliaup).
As of Feb 10, 2025, you should use the LTS 1.10 Julia series version, and not the current release 1.11.
juliaup will by default install the current release. To install the LTS version do 
```bash 
juliaup install lts 
juliaup default lts 
```
Which will make LTS the default version when you type `julia` in your terminal. 


### Editing environment

The easiest editing environment is probably with vscode or one of its free versions like codium, 
with the Julia Language server installed. You can use other editors as well. Instructions can generally
be found [here](https://github.com/JuliaEditorSupport). If you use neovim a nice tutorial can be found [here](https://allanchain.github.io/blog/post/julia-nvim/).

For this set of tutorials we will be using notebooks. For Julia there are two versions of notebooks,
 1. `jupyter` (ju standard for Julia) which can be used if you install the [`IJulia`](https://github.com/JuliaLang/IJulia.jl) package in your global environment
 2. [Pluto](https://github.com/fonsp/Pluto.jl) a reactive notebook written in Julia and javascript, which tends to be easier to use immediately. 


For this tutorial we will use Pluto.jl. Although note that the Julia REPL is really very good and I hardly use notebooks in 
my day to day work.


### Installing Necessary Packages 

Since we are using Pluto the easiest thing to do is to install Pluto into your global Julia environment by opening
a Julia REPL and typing
```julia
julia> ]
pkg> add Pluto
```
#### Warning:
  **In General I do not recommend installing packages in your global environment. Julia's package manager is much 
    better than Python's and its TOML system (Project.toml and Manifest.toml) make it easy to create reproducible
    environments. In general I would recommend making local environments (this is what Pluto does automatically)
    and installing packages there.**

### Launching a Notebook

Assuming you do not have a Julia session open (or if you are still in Pkg mode press backspace to move into REPL mode)
you can start Pluto by typing 
```julia
julia> using Pluto
julia> Pluto.run()
```

to launch the Pluto server in your default browser.


### Tutorials 

Currently we recommend people try the tutorials in this order

 1. `hw1_loadingdata.jl`: How to load and plot data with Comrade 
 2. `hw2_geommodeling.jl`: How to fit simple geometric models with Comrade 
 3. `hw3_imaging.jl`: How to image total intensity visibilities with Comrade including Bayesian self-calibration.
 4. `hw4_polarized_imaging.jl`: How to do polarized imaging of `Comrade` including gains, gain ratios, and leakage calibration.

To run the notebooks first launch Pluto then open the notebook there. Note that the it will not run immediately since the 
default behavior is that the notebook will be in safe mode. To run it you need to hit run notebook on the top banner.
Additionally, note that hw 1 and 2 will note immediately run since the user will need to specify the data to read in. 
