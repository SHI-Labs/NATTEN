![NATTENLogo](../assets/natten_dark.png#gh-dark-mode-only) ![NATTENLogo](../assets/natten_light.png#gh-light-mode-only)

Welcome to NATTEN docs!

* [What is NATTEN and why does it exist?](history.md)
* [How is Neigborhood Attention implemented?](methodology/README.md)
* [How do I install NATTEN?](install.md)
* [Front-end](frontend.md)
  * Describes how to check NATTEN's compatibility with your system, and its available features.
  * Describes how to use torch modules `NeighborhoodAttention1D`, `NeighborhoodAttention2D`, and `NeighborhoodAttention3D`.
  * Describes when and how to use ops directly instead of modules.
* [Fused Neighborhood Attention](fna/) (new)
  * [Quick start guide](fna/fna-quickstart.md)
  * [Fused vs unfused](fna/fused-vs-unfused.md): differences in supported features, layouts, etc.
  * [KV parallelism](fna/kv-parallelism.md) (training only)
  * [Auto-tuner](fna/autotuner.md) (experimental)
* [API](api.md)
  * Describes the current C++ API structure.
* [Backend](backend.md)
  * Overview of different implementations, and available framework extensions.
  * Implementation details and limitations.
  * Dispatch mechanism and auto-generated instantiations.
* [Build system](build.md)
* [Tests](tests.md)
* [Changelog](../CHANGELOG.md)
