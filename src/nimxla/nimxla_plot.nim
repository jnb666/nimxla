# nimxxla_plot starts a webserver listening on localhost for weebsocket
# reequests and launches the default webserver to connect to that page
# see apps in the examples directory where messages are sent from the client
# to update the plot data

import std/[asyncdispatch, logging]
import plots

const logLevel = lvlInfo

proc ctrlc() {.noconv.} =
  echo ":quit"
  quit()

var logger = newConsoleLogger(levelThreshold=logLevel)
addHandler(logger)

setControlCHook(ctrlc)

waitFor servePlots()
