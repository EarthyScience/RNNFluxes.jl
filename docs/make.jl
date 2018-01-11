using Documenter, RNNFluxes

makedocs(
    modules = [RNNFluxes],
    clean   = false,
    format   = :html,
    sitename = "RNNFluxes.jl",
    authors = "Fabian Gans and contributors",
    pages    = Any[ # Compat: `Any` for 0.4 compat
      "Home" => "index.md",
      "Manual" => Any[
        "quickstart.md",
        "models.md",
        "training.md",
        "example_lossfunctions.md"
      ]
    ]
)

deploydocs(
    #deps   = Deps.pip("mkdocs", "python-markdown-math"),
    repo   = "github.com/bgi-jena/RNNFluxes.jl.git",
    julia  = "0.6",
    deps   = nothing,
    make   = nothing,
    target = "build"
)
