### A Pluto.jl notebook ###
# v0.19.47

using Markdown
using InteractiveUtils

# ╔═╡ 3873154a-e7b2-11ef-092b-3ff178b7ca29
import Pkg #hide
__DIR = @__DIR__ #hide
pkg_io = open(joinpath(__DIR, "pkg.log"), "w") #hide
Pkg.activate(__DIR; io=pkg_io) #hide
Pkg.develop(; path=joinpath(__DIR, "..", "..", ".."), io=pkg_io) #hide
Pkg.instantiate(; io=pkg_io) #hide
Pkg.precompile(; io=pkg_io) #hide
close(pkg_io) #hide

# ╔═╡ 387315ae-e7b2-11ef-2cb6-3b74b3e25304
md"# Polarized Image and Instrumental Modeling"

# ╔═╡ 387315c4-e7b2-11ef-1640-11b47a037068
md"""
In this tutorial, we will analyze a simulated simple polarized dataset to demonstrate
Comrade's polarized imaging capabilities.
"""

# ╔═╡ 387315e0-e7b2-11ef-1801-4355a35cbfed
md"""
## Introduction to Polarized Imaging
The EHT is a polarized interferometer. However, like all VLBI interferometers, it does not
directly measure the Stokes parameters (I, Q, U, V). Instead, it measures components
related to the electric field at the telescope along two *directions* using feeds.
There are two types of feeds at telescopes: circular, which measure $R/L$ components of the
electric field, and linear feeds, which measure $X/Y$ components of the electric field.
Most sites in the EHT use circular feeds, meaning they measure the right (R) and left
electric field (L) at each telescope. Although note that ALMA actually uses linear feeds.
Currently Comrade has the ability to fit natively mixed polarization data however, the
publically released EHT data has been converted to circular polarization.
For a VLBI array whose feeds are purely circluar the **coherency matrices** are given by,

$$
 C_{ij} = \begin{pmatrix}
       RR^* &  RL^*\\
       LR^* &  LL^*
     \end{pmatrix}.
$$

These coherency matrices are the fundamental object in interferometry and what
the telescope observes. For a perfect interferometer, these circular coherency matrices
are related to the usual Fourier transform of the stokes parameters by

$$
  \begin{pmatrix}
      \tilde{I}\\ \tilde{Q} \\ \tilde{U} \\ \tilde{V}
  \end{pmatrix}
  =\frac{1}{2}
  \begin{pmatrix}
     RR^* + LL^* \\
     RL^* + LR^* \\
     i(LR^* - RL^*)\\
     RR^* - LL^*
  \end{pmatrix}.
$$
"""

# ╔═╡ 387315ea-e7b2-11ef-1b28-9d0df55706d5
md"""
> **Note**
>
> In this tutorial, we stick to circular feeds but Comrade has the capabilities
> to model linear (XX,XY, ...) and mixed basis coherencies (e.g., RX, RY, ...).
"""

# ╔═╡ 387315fe-e7b2-11ef-22d4-d191df34b146
md"""
In reality, the measure coherencies are corrupted by both the atmosphere and the
telescope itself. In `Comrade` we use the RIME formalism [^1] to represent these corruptions,
namely our measured coherency matrices $V_{ij}$ are given by
$$
   V_{ij} = J_iC_{ij}J_j^\dagger
$$
where $J$ is known as a *Jones matrix* and $ij$ denotes the baseline $ij$ with sites $i$ and $j$.
"""

# ╔═╡ 3873161c-e7b2-11ef-2a9c-59f863b2648e
md"""
`Comrade` is highly flexible with how the Jones matrices are formed and provides several
convenience functions that parameterize standard Jones matrices. These matrices include:
  - `JonesG` which builds the set of complex gain Jones matrices
$$
  G = \begin{pmatrix}
          g_a   &0\\
          0     &g_b\\
      \end{pmatrix}
$$
  - `JonesD` which builds the set of complex d-terms Jones matrices
$$
  D = \begin{pmatrix}
          1   & d_a\\
          d_b     &1\\
      \end{pmatrix}
$$
  - `JonesR` is the basis transform matrix $T$. This transformation is special and
     combines two things using the decomposition $T=FB$. The first, $B$, is the transformation from
     some reference basis to the observed coherency basis (this allows for mixed basis measurements).
     The second is the feed rotation, $F$, that transforms from some reference axis to the axis of the
     telescope as the source moves in the sky. The feed rotation matrix `F` for circular feeds
     in terms of the per station feed rotation angle $\varphi$ is
$$
  F = \begin{pmatrix}
          e^{-i\varphi}   & 0\\
          0     & e^{i\varphi}\\
      \end{pmatrix}
$$
"""

# ╔═╡ 38731624-e7b2-11ef-118f-6bd02b8b3646
md"""
 In the rest of the tutorial, we are going to solve for all of these instrument model terms in
 while re-creating the polarized image from the first [`EHT results on M87`](https://iopscience.iop.org/article/10.3847/2041-8213/abe71d).
"""

# ╔═╡ 38731630-e7b2-11ef-316e-4341c7999469
md"""
## Load the Data
To get started we will load Comrade
"""

# ╔═╡ 3873163a-e7b2-11ef-2525-cfb4c9dd1f83
using Comrade

# ╔═╡ 3873164e-e7b2-11ef-33d1-a1ccc4e0524c
md"## Load the Data"

# ╔═╡ 38731656-e7b2-11ef-23f3-171b79a08160
using Pyehtim

# ╔═╡ 38731662-e7b2-11ef-2b94-bd67ec4cbb02
md"For reproducibility we use a stable random number genreator"

# ╔═╡ 3873166c-e7b2-11ef-25c0-e529a9fb9965
using StableRNGs
rng = StableRNG(42)

# ╔═╡ 38731676-e7b2-11ef-0eb4-79857d24568d
md"Now we will load some synthetic polarized data."

# ╔═╡ 3873168a-e7b2-11ef-102e-298050e28028
fname = Base.download("https://de.cyverse.org/anon-files/iplant/home/shared/commons_repo/curated/EHTC_M87pol2017_Nov2023/hops_data/April11/SR2_M87_2017_101_lo_hops_ALMArot.uvfits",
                      joinpath(__DIR, "m87polarized.uvfits")
                    )
obs = Pyehtim.load_uvfits_and_array(
        fname,
        joinpath(__DIR, "..", "..", "Data", "array.txt"), polrep="circ")

# ╔═╡ 38731694-e7b2-11ef-015e-393111d75441
md"""
Notice that, unlike other non-polarized tutorials, we need to include a second argument.
This is the **array file** of the observation and is required to determine the feed rotation
of the array.
"""

# ╔═╡ 3873169e-e7b2-11ef-01c2-45256b27c713
md"Now we scan average the data since the data to boost the SNR and reduce the total data volume."

# ╔═╡ 387316a8-e7b2-11ef-2f64-4f5c3d0d356d
obs = scan_average(obs).add_fractional_noise(0.01).flag_uvdist(uv_min=0.1e9)

# ╔═╡ 387316b2-e7b2-11ef-227c-e981fce7bbd9
md"Now we extract our observed/corrupted coherency matrices."

# ╔═╡ 387316bc-e7b2-11ef-034c-310581c04c07
dvis = extract_table(obs, Coherencies())

# ╔═╡ 387316c8-e7b2-11ef-0a94-137e0a5eeeb2
md"##Building the Model/Posterior"

# ╔═╡ 387316e4-e7b2-11ef-08b6-453d7e5db983
md"""
To build the model, we first break it down into two parts:
   1. **The image or sky model**. In Comrade, all polarized image models are written in terms of the Stokes parameters.
      In this tutorial, we will use a polarized image model based on Pesce (2021)[^2], and
      parameterizes each pixel in terms of the [`Poincare sphere`](https://en.wikipedia.org/wiki/Unpolarized_light#Poincar%C3%A9_sphere).
      This parameterization ensures that we have all the physical properties of stokes parameters.
      Note that we also have a parameterization in terms of hyperbolic trig functions `VLBISkyModels.PolExp2Map`
   2. **The instrument model**. The instrument model specifies the model that describes the impact of instrumental and atmospheric effects.
      We will be using the $J = GDR$ decomposition we described above. However, to parameterize the
      R/L complex gains, we will be using a gain product and ratio decomposition. The reason for this decomposition
      is that in realistic measurements, the gain ratios and products have different temporal characteristics.
      Namely, many of the EHT observations tend to demonstrate constant R/L gain ratios across an
      nights observations, compared to the gain products, which vary every scan. Additionally,
      the gain ratios tend to be smaller (i.e., closer to unity) than the gain products.
      Using this apriori knowledge, we can build this into our model and reduce
      the total number of parameters we need to model.
"""

# ╔═╡ 387316ee-e7b2-11ef-1d15-334aca9e89f7
md"""
First we specify our sky model. As always `Comrade` requires this to be a two argument
function where the first argument is typically a NamedTuple of parameters we will fit
and the second are additional metadata required to build the model.
"""

# ╔═╡ 38731702-e7b2-11ef-38f9-597abc365e40
using StatsFuns: logistic
function sky(θ, metadata)
    (;c, σ, p, p0, pσ, angparams) = θ
    (;mimg, ftot) = metadata
    # Build the stokes I model
    rast = apply_fluctuations(CenteredLR(), mimg, σ.*c.params)
    brast = baseimage(rast)
    brast .= ftot.*brast
    # The total polarization fraction is modeled in logit space so we transform it back
    pim = logistic.(p0 .+ pσ.*p.params)
    # Build our IntensityMap
    pmap = PoincareSphere2Map(rast, pim, angparams)
    # Construct the actual image model which uses a third order B-spline pulse
    m = ContinuousImage(pmap, BSplinePulse{3}())
    # Finally find the image centroid and shift it to be at the center
    x0, y0 = centroid(pmap)
    ms = shifted(m, -x0, -y0)
    return ms
end

# ╔═╡ 38731716-e7b2-11ef-2776-b3c9562abfbd
md"""
Now, we define the model metadata required to build the model.
We specify our image grid and cache model needed to define the polarimetric
image model. Our image will be a 10x10 raster with a 60μas FOV.
"""

# ╔═╡ 3873172c-e7b2-11ef-25c4-7ff0bb49d94e
using Distributions
using VLBIImagePriors
fovx = μas2rad(200.0)
fovy = μas2rad(200.0)
nx = ny = 32
grid = imagepixels(fovx, fovy, nx, ny)

fwhmfac = 2*sqrt(2*log(2))
mpr  = modify(Gaussian(), Stretch(μas2rad(50.0)./fwhmfac))
mimg = intensitymap(mpr, grid)

# ╔═╡ 38731734-e7b2-11ef-1a0c-773b850628ec
md"""
For the image metadata we specify the grid and the total flux of the image, which is 1.0.
Note that we specify the total flux out front since it is degenerate with an overall shift
in the gain amplitudes.
"""

# ╔═╡ 3873173e-e7b2-11ef-35c7-a5985834901b
skymeta = (; mimg=mimg./flux(mimg), ftot=0.6);

# ╔═╡ 38731752-e7b2-11ef-09e0-5370f7cd7827
md"""
We use again use a GMRF prior similar to the Imaging a Black Hole using only Closure Quantities tutorial
for the log-ratio transformed image. We use the same correlated image prior for the inverse-logit transformed
total polarization. The mean total polarization fraction `p0` is centered at -2.0 with a standard deviation of 2.0
which logit transformed puts most of the prior mass < 0.8 fractional polarization. The standard deviation of the
total polarization fraction `pσ` again uses a Half-normal process. The angular parameters of the polarizaton are
given by a uniform prior on the sphere.
"""

# ╔═╡ 3873175a-e7b2-11ef-25ae-c59a91065f99
cprior = corr_image_prior(grid, dvis)
skyprior = (
    c = cprior,
    σ  = Exponential(0.1),
    p  = cprior,
    p0 = Normal(-2.0, 2.0),
    pσ =  truncated(Normal(0.0, 1.0); lower=0.01),
    angparams = ImageSphericalUniform(nx, ny),
    )

skym = SkyModel(sky, skyprior, grid; metadata=skymeta)

# ╔═╡ 38731770-e7b2-11ef-13a8-f30fea9656a7
md"""
Now we build the instrument model. Due to the complexity of VLBI the instrument model is critical
to the success of imaging and getting reliable results. For this example we will use the standard
instrument model used in polarized EHT analyses expressed in the RIME formalism. Our Jones
decomposition will be given by `GDR`, where `G` are the complex gains, `D` are the d-terms, and `R`
is what we call the *ideal instrument response*, which is how an ideal interferometer using the
feed basis we observe relative to some reference basis.

Given the possible flexibility in different parameterizations of the individual Jones matrices
each Jones matrix requires the user to specify a function that converts from parameters
to specific parameterization f the jones matrices.
"""

# ╔═╡ 3873177a-e7b2-11ef-03c0-d704c3eda4ce
md"""
For the complex gain matrix, we used the `JonesG` jones matrix. The first argument is now
a function that converts from the parameters to the complex gain matrix. In this case, we
will use a amplitude and phase decomposition of the complex gain matrix. Note that since
the gain matrix is a diagonal 2x2 matrix the function must return a 2-element tuple.
The first element of the tuple is the gain for the first polarization feed (R) and the
second is the gain for the second polarization feed (L).
"""

# ╔═╡ 3873178c-e7b2-11ef-36df-13c0ada008f7
function fgain(x)
    gR = exp(x.lgR + 1im*x.gpR)
    gL = gR*exp(x.lgrat + 1im*x.gprat)
    return gR, gL
end
G = JonesG(fgain)

# ╔═╡ 38731798-e7b2-11ef-0c2a-4789350d40ee
md"""
Similarly we provide a `JonesD` function for the leakage terms. Since we assume that we
are in the small leakage limit, we will use the decomposition
1 d1
d2 1
Therefore, there are 2 free parameters for the JonesD our parameterization function
must return a 2-element tuple. For d-terms we will use a re-im parameterization.
"""

# ╔═╡ 387317ac-e7b2-11ef-2bd7-f1b749dfe530
function fdterms(x)
    dR = complex(x.dRx, x.dRy)
    dL = complex(x.dLx, x.dLy)
    return dR, dL
end
D = JonesD(fdterms)

# ╔═╡ 387317b6-e7b2-11ef-2c89-e90970cab2f0
md"""
Finally we define our response Jones matrix. This matrix is a basis transform matrix
plus the feed rotation angle for each station. These are typically set by the telescope
so there are no free parameters, so no parameterization is necessary.
"""

# ╔═╡ 387317be-e7b2-11ef-0b92-fd0c8aaf9e71
R = JonesR(;add_fr=true)

# ╔═╡ 387317ca-e7b2-11ef-0404-c5c45b7bc57f
md"""
Finally, we build our total Jones matrix by using the `JonesSandwich` function. The
first argument is a function that specifies how to combine each Jones matrix. In this case
we will use the standard decomposition J = adjoint(R)*G*D*R, where we need to apply the adjoint
of the feed rotaion matrix `R` because the data has feed rotation calibration.
"""

# ╔═╡ 387317d4-e7b2-11ef-1b74-a1d44ba38b17
js(g,d,r) = adjoint(r)*g*d*r
J = JonesSandwich(js, G, D, R)

# ╔═╡ 387317e8-e7b2-11ef-044a-adf7db2ee3f3
md"""
> **Note**
>
> This is a general note that for arrays with non-zero leakage, feed rotation calibration
> does not remove the impact of feed rotations on the instrument model. That is,
> when modeling feed rotation must be taken into account. This is because
> the R and D matrices are not commutative. Therefore, to recover the correct instrumental
> terms we must include the feed rotation calibration in the instrument model. This is not
> ideal when doing polarized modeling, especially for interferometers using a mixture of linear
> and circular feeds. For linear feeds R does not commute with G or D and applying feed rotation
> calibration before solving for gains can mix gains and leakage with feed rotation calibration terms
> breaking many of the typical assumptions about the stabilty of different instrument effects.
"""

# ╔═╡ 387317fc-e7b2-11ef-256b-7d8d7994d009
md"""
For the instrument prior, we will use a simple IID prior for the complex gains and d-terms.
The `IIDSitePrior` function specifies that each site has the same prior and each value is independent
on some time segment. The current time segments are
 - `ScanSeg()` which specifies each scan has an independent value
 - `TrackSeg()` which says that the value is constant over the track.
 - `IntegSeg()` which says that the value changes each integration time
For the released EHT data, the calibration procedure makes gains stable over each scan
so we use `ScanSeg` for those quantities. The d-terms are typically stable over the track
so we use `TrackSeg` for those.
"""

# ╔═╡ 38731806-e7b2-11ef-3e6f-3f746986ddd6
intprior = (
    lgR  = ArrayPrior(IIDSitePrior(ScanSeg(), Normal(0.0, 0.2)); LM = IIDSitePrior(ScanSeg(), Normal(0.0, 1.0))),
    lgrat= ArrayPrior(IIDSitePrior(ScanSeg(), Normal(0.0, 0.1))),
    gpR  = ArrayPrior(IIDSitePrior(ScanSeg(), DiagonalVonMises(0.0, inv(π^2))); refant=SEFDReference(0.0), phase=true),
    gprat= ArrayPrior(IIDSitePrior(ScanSeg(), DiagonalVonMises(0.0, inv(0.1^2))); refant = SingleReference(:AA, 0.0), phase=false),
    dRx  = ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.2))),
    dRy  = ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.2))),
    dLx  = ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.2))),
    dLy  = ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.2))),
)

# ╔═╡ 3873181a-e7b2-11ef-11f4-01661ec4ac52
md"""
Finally, we can build our instrument model which takes a model for the Jones matrix `J`
and priors for each term in the Jones matrix.
"""

# ╔═╡ 3873181a-e7b2-11ef-0367-c5e964d730cb
intmodel = InstrumentModel(J, intprior)

# ╔═╡ 38731830-e7b2-11ef-3a52-df57bf79f989
md"""
intmodel = InstrumentModel(JonesR(;add_fr=true))
Putting it all together, we form our likelihood and posterior objects for optimization and
sampling, and specifying to use Enzyme.Reverse with runtime activity for AD.
"""

# ╔═╡ 38731838-e7b2-11ef-22ef-f1a4b7cdf790
using Enzyme
post = VLBIPosterior(skym, intmodel, dvis; admode=set_runtime_activity(Enzyme.Reverse))

# ╔═╡ 38731842-e7b2-11ef-1f77-a947a7b20bfc
md"## Reconstructing the Image and Instrument Effects"

# ╔═╡ 3873184c-e7b2-11ef-1586-c1b25b225eea
md"""
To sample from this posterior, it is convenient to move from our constrained parameter space
to an unconstrained one (i.e., the support of the transformed posterior is (-∞, ∞)). This transformation is
done using the `asflat` function.
"""

# ╔═╡ 38731862-e7b2-11ef-14a4-33bf0e381b0c
tpost = asflat(post)

# ╔═╡ 3873186a-e7b2-11ef-0c7d-6ff80d0deb4b
md"""
We can also query the dimension of our posterior or the number of parameters we will sample.
> **Warning**
>
> This can often be different from what you would expect. This difference is especially true when using
> angular variables, where we often artificially increase the dimension
> of the parameter space to make sampling easier.
"""

# ╔═╡ 38731874-e7b2-11ef-00ef-f995e0ec405c
md"""
Now we optimize. Unlike other imaging examples, we move straight to gradient optimizers
due to the higher dimension of the space. In addition the only AD package that can currently
work with the polarized Comrade posterior is Enzyme.
"""

# ╔═╡ 3873187e-e7b2-11ef-08ea-59e241b9bc08
using Optimization
using OptimizationOptimisers
xopt, sol = comrade_opt(post, Optimisers.Adam();
                        initial_params=prior_sample(rng, post), maxiters=25_000)

# ╔═╡ 38731888-e7b2-11ef-3ab3-2599dac5d225
md"Now let's evaluate our fits by plotting the residuals"

# ╔═╡ 3873189c-e7b2-11ef-0bf7-9fc07b34f598
using CairoMakie
using DisplayAs #hide
res = residuals(post, xopt)
fig = Figure(;size=(800, 600))
baselineplot(fig[1,1], res[1], :uvdist, x->Comrade.measurement(x)[1,1]/noise(x)[1,1], axis=(ylabel="RR Residual", xlabel="uv distance (λ)"))
baselineplot(fig[2,1], res[1], :uvdist, x->Comrade.measurement(x)[2,1]/noise(x)[1,1], axis=(ylabel="LR Residual", xlabel="uv distance (λ)"))
baselineplot(fig[1,2], res[1], :uvdist, x->Comrade.measurement(x)[1,2]/noise(x)[1,1], axis=(ylabel="RL Residual", xlabel="uv distance (λ)"))
baselineplot(fig[2,2], res[1], :uvdist, x->Comrade.measurement(x)[2,2]/noise(x)[1,1], axis=(ylabel="LL Residual", xlabel="uv distance (λ)"))
fig |> DisplayAs.PNG |> DisplayAs.Text

# ╔═╡ 387318a6-e7b2-11ef-1015-b1df16411c36
md"""
These look reasonable, although there may be some minor overfitting.
Let's compare our results to the ground truth values we know in this example.
First, we will load the polarized truth
"""

# ╔═╡ 387318b0-e7b2-11ef-2d24-df8c85f4be4f
imgtrue = load_fits(joinpath(__DIR, "..", "..", "Data", "polarized_gaussian.fits"), IntensityMap{StokesParams});

# ╔═╡ 387318ba-e7b2-11ef-247d-8ffc65c8d78a
md"Select a reasonable zoom in of the image."

# ╔═╡ 387318c2-e7b2-11ef-09b8-29684c29342a
imgtruesub = regrid(imgtrue, imagepixels(fovx, fovy, nx*4, ny*4))
img = intensitymap(Comrade.skymodel(post, xopt), axisdims(imgtruesub))

#Plotting the results gives
fig = imageviz(img, adjust_length=true, colormap=:bone, pcolormap=:RdBu)
fig |> DisplayAs.PNG |> DisplayAs.Text

# ╔═╡ 387318ce-e7b2-11ef-3a18-39f0dc312400
md"""
> **Note**
>
> The image looks a little noisy. This is an artifact of the MAP image. To get a publication quality image
> we recommend sampling from the posterior and averaging the samples. The results will be essentially
> identical to the results from [EHTC VII](https://iopscience.iop.org/article/10.3847/2041-8213/abe71d).
"""

# ╔═╡ 387318e2-e7b2-11ef-28ea-cb34298e131d
md"""
We can also analyze the instrument model. For example, we can look at the gain ratios and products.
To grab the ratios and products we can use the `caltable` function which will return analyze the gprat array
and convert it to a uniform table. We can then plot the gain phases and amplitudes.
"""

# ╔═╡ 387318ec-e7b2-11ef-16a2-ebb299998162
gphase_ratio = caltable(xopt.instrument.gprat)
gamp_ratio   = caltable(exp.(xopt.instrument.lgrat))

# ╔═╡ 387318f4-e7b2-11ef-38e5-898be1fe4e37
md"""
Plotting the phases first, we see large trends in the righ circular polarization phase. This is expected
due to a lack of image centroid and the absense of absolute phase information in VLBI. However, the gain
phase difference between the left and right circular polarization is stable and close to zero. This is
expected since gain ratios are typically stable over the course of an observation and the constant
offset was removed in the EHT calibration process.
"""

# ╔═╡ 38731900-e7b2-11ef-0531-c593a24b4d57
gphaseR = caltable(xopt.instrument.gpR)
fig = plotcaltable(gphaseR, gphase_ratio, labels=["R Phase", "L/R Phase"]);
fig |> DisplayAs.PNG |> DisplayAs.Text

# ╔═╡ 38731914-e7b2-11ef-07ce-83b328c37e2b
md"""
Moving to the amplitudes we see largely stable gain amplitudes on the right circular polarization except for LMT which is
known and due to pointing issues during the 2017 observation. Again the gain ratios are stable and close to unity. Typically
we expect that apriori calibration should make the gain ratios close to unity.
"""

# ╔═╡ 3873191e-e7b2-11ef-302a-83dfcd58f326
gampr = caltable(exp.(xopt.instrument.lgR))
fig = plotcaltable(gampr, gamp_ratio, labels=["R Amp", "L/R Amp"], axis_kwargs=(;limits=(nothing, (0.6, 1.3))));
fig |> DisplayAs.PNG |> DisplayAs.Text

# ╔═╡ 38731928-e7b2-11ef-1c25-7d2b23a745ee
md"""
To sample from the posterior, you can then just use the `sample` function from AdvancedHMC like in the
other imaging examples. For example
```julia
using AdvancedHMC
chain = sample(rng, post, NUTS(0.8), 10_000, n_adapts=5000, progress=true, initial_params=xopt)
```
"""

# ╔═╡ 3873193c-e7b2-11ef-37ae-df2db7c5a9dd
md"""
[^1]: Hamaker J.P, Bregman J.D., Sault R.J. (1996) [https://articles.adsabs.harvard.edu/pdf/1996A%26AS..117..137H]
[^2]: Pesce D. (2021) [https://ui.adsabs.harvard.edu/abs/2021AJ....161..178P/abstract]
"""

# ╔═╡ 38731946-e7b2-11ef-32a4-e78dc923ecd4
md"""
---

*This notebook was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
"""

# ╔═╡ Cell order:
# ╠═3873154a-e7b2-11ef-092b-3ff178b7ca29
# ╟─387315ae-e7b2-11ef-2cb6-3b74b3e25304
# ╟─387315c4-e7b2-11ef-1640-11b47a037068
# ╟─387315e0-e7b2-11ef-1801-4355a35cbfed
# ╟─387315ea-e7b2-11ef-1b28-9d0df55706d5
# ╟─387315fe-e7b2-11ef-22d4-d191df34b146
# ╟─3873161c-e7b2-11ef-2a9c-59f863b2648e
# ╟─38731624-e7b2-11ef-118f-6bd02b8b3646
# ╟─38731630-e7b2-11ef-316e-4341c7999469
# ╠═3873163a-e7b2-11ef-2525-cfb4c9dd1f83
# ╟─3873164e-e7b2-11ef-33d1-a1ccc4e0524c
# ╠═38731656-e7b2-11ef-23f3-171b79a08160
# ╟─38731662-e7b2-11ef-2b94-bd67ec4cbb02
# ╠═3873166c-e7b2-11ef-25c0-e529a9fb9965
# ╟─38731676-e7b2-11ef-0eb4-79857d24568d
# ╠═3873168a-e7b2-11ef-102e-298050e28028
# ╟─38731694-e7b2-11ef-015e-393111d75441
# ╟─3873169e-e7b2-11ef-01c2-45256b27c713
# ╠═387316a8-e7b2-11ef-2f64-4f5c3d0d356d
# ╟─387316b2-e7b2-11ef-227c-e981fce7bbd9
# ╠═387316bc-e7b2-11ef-034c-310581c04c07
# ╟─387316c8-e7b2-11ef-0a94-137e0a5eeeb2
# ╟─387316e4-e7b2-11ef-08b6-453d7e5db983
# ╟─387316ee-e7b2-11ef-1d15-334aca9e89f7
# ╠═38731702-e7b2-11ef-38f9-597abc365e40
# ╟─38731716-e7b2-11ef-2776-b3c9562abfbd
# ╠═3873172c-e7b2-11ef-25c4-7ff0bb49d94e
# ╟─38731734-e7b2-11ef-1a0c-773b850628ec
# ╠═3873173e-e7b2-11ef-35c7-a5985834901b
# ╟─38731752-e7b2-11ef-09e0-5370f7cd7827
# ╠═3873175a-e7b2-11ef-25ae-c59a91065f99
# ╟─38731770-e7b2-11ef-13a8-f30fea9656a7
# ╟─3873177a-e7b2-11ef-03c0-d704c3eda4ce
# ╠═3873178c-e7b2-11ef-36df-13c0ada008f7
# ╟─38731798-e7b2-11ef-0c2a-4789350d40ee
# ╠═387317ac-e7b2-11ef-2bd7-f1b749dfe530
# ╟─387317b6-e7b2-11ef-2c89-e90970cab2f0
# ╠═387317be-e7b2-11ef-0b92-fd0c8aaf9e71
# ╟─387317ca-e7b2-11ef-0404-c5c45b7bc57f
# ╠═387317d4-e7b2-11ef-1b74-a1d44ba38b17
# ╟─387317e8-e7b2-11ef-044a-adf7db2ee3f3
# ╟─387317fc-e7b2-11ef-256b-7d8d7994d009
# ╠═38731806-e7b2-11ef-3e6f-3f746986ddd6
# ╟─3873181a-e7b2-11ef-11f4-01661ec4ac52
# ╠═3873181a-e7b2-11ef-0367-c5e964d730cb
# ╟─38731830-e7b2-11ef-3a52-df57bf79f989
# ╠═38731838-e7b2-11ef-22ef-f1a4b7cdf790
# ╟─38731842-e7b2-11ef-1f77-a947a7b20bfc
# ╟─3873184c-e7b2-11ef-1586-c1b25b225eea
# ╠═38731862-e7b2-11ef-14a4-33bf0e381b0c
# ╟─3873186a-e7b2-11ef-0c7d-6ff80d0deb4b
# ╟─38731874-e7b2-11ef-00ef-f995e0ec405c
# ╠═3873187e-e7b2-11ef-08ea-59e241b9bc08
# ╟─38731888-e7b2-11ef-3ab3-2599dac5d225
# ╠═3873189c-e7b2-11ef-0bf7-9fc07b34f598
# ╟─387318a6-e7b2-11ef-1015-b1df16411c36
# ╠═387318b0-e7b2-11ef-2d24-df8c85f4be4f
# ╟─387318ba-e7b2-11ef-247d-8ffc65c8d78a
# ╠═387318c2-e7b2-11ef-09b8-29684c29342a
# ╟─387318ce-e7b2-11ef-3a18-39f0dc312400
# ╟─387318e2-e7b2-11ef-28ea-cb34298e131d
# ╠═387318ec-e7b2-11ef-16a2-ebb299998162
# ╟─387318f4-e7b2-11ef-38e5-898be1fe4e37
# ╠═38731900-e7b2-11ef-0531-c593a24b4d57
# ╟─38731914-e7b2-11ef-07ce-83b328c37e2b
# ╠═3873191e-e7b2-11ef-302a-83dfcd58f326
# ╟─38731928-e7b2-11ef-1c25-7d2b23a745ee
# ╟─3873193c-e7b2-11ef-37ae-df2db7c5a9dd
# ╟─38731946-e7b2-11ef-32a4-e78dc923ecd4
