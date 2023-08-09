# Package
version       = "0.1.0"
author        = "John Banks"
description   = "Bindings for the XLA accelerated linear algebra library and associate packages for machine learning"
license       = "MIT"

srcDir = "src"
installExt = @["nim"]
bin = @["nimxla/nimxla_plot"]

# Dependencies
requires "nim >= 2.0.0"
requires "zippy >= 0.10.10"
requires "ws >= 0.5.0"
requires "zip >= 0.3.1"

task makedocs, "build the docs":
  exec "rm -fr htdocs"
  exec "find src -maxdepth 2 -name '*.nim' -exec nim doc --outdir:htdocs --path:./src --index:on --git.url:https://github.com/jnb666/nimxla --git.commit:main {} \\;"
  exec "nim buildIndex -o:htdocs/theindex.html htdocs"

task makereadme, "convert readme to HTML":
  # uses https://github.com/KrauseFx/markdown-to-html-github-style
  exec "rm -f README.html"
  exec "node ~/src/markdown-to-html-github-style/convert.js NimXLA"
