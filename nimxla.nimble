# Package
version       = "0.1.0"
author        = "John Banks"
description   = "Bindings for the XLA accelerated linear algebra library and associate packages for machine learning"
license       = "MIT"

srcDir = "src"

# Dependencies
requires "nim >= 1.6.12"


task makedocs, "build the docs":
  exec "rm -fr htdocs"
  exec "nim doc --outdir:htdocs --path:./src --project src/nimxla.nim"

task makereadme, "convert readme to HTML":
  # uses https://github.com/KrauseFx/markdown-to-html-github-style
  exec "rm -f README.html"
  exec "node ~/src/markdown-to-html-github-style/convert.js NimXLA"
