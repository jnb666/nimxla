# nimxxla_plot starts a webserver listening on localhost for weebsocket
# reequests and launches the default webserver to connect to that page
# see apps in the examples directory where messages are sent from the client
# to update the plot data

import std/[asyncdispatch, logging, parseopt, os]
import plots

proc ctrlc() {.noconv.} =
  echo ":quit"
  quit()

proc main() =
  var p = initOptParser(shortNoVal = {'d'}, longNoVal = @["debug"])
  var logLevel = lvlInfo
  for kind, key, val in p.getopt():
    if (kind == cmdLongOption and key == "debug") or (kind == cmdShortOption and key == "d"):
      logLevel = lvlDebug
  let logFile = joinPath(getCurrentDir(), "nimxla_plot.log")
  echo "writing log to ", logFile
  var logger = newFileLogger(logFile, levelThreshold=logLevel, fmtStr="[$datetime] $levelname: ")
  addHandler(logger)
  setControlCHook(ctrlc)
  waitFor servePlots()

main()