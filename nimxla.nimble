# Package
version       = "0.1.0"
author        = "John Banks"
description   = "Bindings for the XLA accelerated linear algebra library and associate packages for machine learning"
license       = "MIT"

srcDir = "src"

# Dependencies
requires "nim >= 1.6.12"
requires "zippy >= 0.10.10"

task makedocs, "build the docs":
  exec "rm -fr htdocs"
  # do this manually to exclude private stuff
  for file in @["src/nimxla.nim", "src/nimxla/tensor.nim", "src/nimxla/literal.nim", "src/nimxla/shape.nim", "src/nimxla/graph.nim", "src/nimxla/nn.nim", "src/nimxla/data.nim"]:
    exec "nim doc --outdir:htdocs --path:./src --index:on --git.url:https://github.com/jnb666/nimxla --git.commit:main " & file 
  exec "nim buildIndex -o:htdocs/theindex.html htdocs"

task makereadme, "convert readme to HTML":
  # uses https://github.com/KrauseFx/markdown-to-html-github-style
  exec "rm -f README.html"
  exec "node ~/src/markdown-to-html-github-style/convert.js NimXLA"
