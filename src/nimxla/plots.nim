## The plots module contains some utilities for plotting graphs and images.
{.warning[BareExcept]:off.}
import std/[asynchttpserver, asyncdispatch, json, strformat, browsers, logging, os, osproc, nativesockets]
import ws
import tensor

type Pixel = array[3, uint8]

const 
  httpPort  = 8080
  indexHTML = staticRead("resources/index.html")
  plotlyJS  = staticRead("resources/plotly-2.24.1.min.js")

# default dark layout
let defaultLayout = %*{
  "paper_bgcolor": "#222",
  "plot_bgcolor" : "#111",
  "colorway"     : ["#d62728", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", 
                    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#1f77b4"],
  "font"         : {"color": "#f2f5fa"},
  "margin"       : {"t": 30, "l": 60, "r": 8, "b": 30},
};

let defaultAxisLayout = %*{
  "gridcolor": "#283442", "linecolor": "#506784", "zerolinecolor": "#283442"
}

proc mergeJson(src, default: JsonNode): JsonNode =
  ## Merge keys either from src or default.
  if src == nil:
    return default
  if src.kind != JObject or default.kind != JObject:
    raise newException(ValueError, "expecting args to be JObject kind")
  result = default.copy
  for key, value in src:
    result[key] = value

proc xaxis(i: int): string =
  if i == 1: "xaxis" else: "xaxis" & $i

proc yaxis(i: int): string =
  if i == 1: "yaxis" else: "yaxis" & $i

proc gridLayout*(rows = 1, cols = 1, ytitle: openarray[string] = [], xtitle: openarray[string] = []): JsonNode =
  ## Create plotly layout for grid of plots with optional labels for each axis.
  result = %*{
    "grid": {"rows": rows, "columns": cols},
  }
  for i in 1..rows:
    result[yaxis(i)] = defaultAxisLayout.copy
    if ytitle.len >= i:
      result[yaxis(i)]["title"] = %ytitle[i-1]
  for i in 1..cols:
    result[xaxis(i)] = defaultAxisLayout.copy
    if xtitle.len >= i:
      result[xaxis(i)]["title"] = %xtitle[i-1]

proc annotation(row, col: int, title: string, err: bool): JsonNode =
  let bgcol = if err: "#611" else: "#222"
  %*{"text": title, "showarrow": false, "xref": "x" & $col, "yref": "y" & $row,
     "x": 0, "y": 0, "xanchor": "left", "yanchor": "top", "bgcolor": bgcol, "opacity": 0.8}

proc plotImage*(t: Tensor[uint8], row, col: int): JsonNode =
  ## Convert data from a uint8 tensor in \[H, W, C\] layout to a plotly image.
  ## number of channels should be either 1 for greyscale or 3 or RGB image data.
  let (h, w, c) = (t.dims[0], t.dims[1], t.dims[2])
  assert c == 1 or c == 3
  var pixels: seq[seq[Pixel]]
  var pixrow = newSeq[Pixel](w)
  for y in 0 ..< h:
    for x in 0 ..< w:
      pixrow[x] = if c == 1:
        let v = t[y, x, 0]
        [v, v, v]
      else:
        [t[y, x, 0], t[y, x, 1], t[y, x, 2]]
    pixels.add pixrow
  %*{
    "z": pixels,
    "type": "image",
    "xaxis": "x" & $col,
    "yaxis": "y" & $row,
    "colormodel": "rgb",
  }

proc plotImageGrid*(title: string, rows, cols: int, getData: proc(): (Tensor[uint8], seq[string], seq[bool])): (JsonNode, JsonNode) =
  ## Plot grid of images and returns data and loyout Json objects for input to plotly.
  ## getData callBack function recieves should return a \[N,H,W,C\] 4d tensor with the images for this page -
  ## i.e. `t.at(row*cols + col)` returns a \[H,W,C\] grayscale (C=0) or RGB (C=0,1,2) image.
  ## and an optional sequence of labels for each image indexed in the same way.
  var layout = %*{
    "title": {"text": title},
    "margin": {"t": 30, "l": 30, "r": 0, "b": 0},
    "grid": {"rows": rows, "columns": cols},
    "annotations": [],
  }
  var images = %[]
  let (t, labels, err) = getData()
  if t.dims.len != 4:
    raise newException(ValueError, &"plotImageGrid: expecting 4d tensor with images from getData() - got {t.dims}")
  let (width, height) = (t.dims[2], t.dims[1])
  let blank = zeros[uint8](t.dims[1 .. ^1])
  var ix = 0
  for row in 1 .. rows:
    for col in 1 .. cols:
      let img = if ix < t.dims[0]: t.at(ix) else: blank
      images.add plotImage(img, row, col)
      if labels.len > ix:
        layout["annotations"].add annotation(row, col, labels[ix], err[ix])
      ix += 1
  for i in 1 .. rows:
    layout[yaxis(i)] = %*{"visible": false, "range": [height, 0]}
  for i in 1 .. cols:
    layout[xaxis(i)] = %*{"visible": false, "range": [0, width]}
  return (images, layout)

proc getWsUrl(): string =
  &"ws://{getHostname()}:{httpPort}/ws"

proc openWebSocket*(): WebSocket =
  ## Blocking open to get client websocket.
  ## Will start nimxla_plot server in the background if not already running.
  let hostname = getHostname()
  let wsUrl = getWsUrl()
  let ps = execProcess("""ps x | grep " nimxla_plot" | grep -v grep""")
  if ps == "":
    echo &"nimxla_plot not running - starting it on http://{hostname}:{httpPort}/"
    let loggers = getHandlers()
    let debugFlag = if loggers.len > 0 and loggers[0].levelThreshold <= lvlDebug: "--debug" else: ""
    let origDir = getCurrentDir()
    try:
      setCurrentDir(getTempDir())
      discard execCmd(&"nohup nimxla_plot {debugFlag} &")
      sleep 1000
      waitFor newWebSocket(wsUrl)
    except OSError:
      quit "ERROR: cannot launch nimxla_plot server - check it is installed in your $PATH - nimble install should do this"
    finally:
      setCurrentDir(origDir)
  else:
    echo &"nimxla_plot already listening on http://{hostname}:{httpPort}/"
    try:
      waitFor newWebSocket(wsUrl)
    except OSError:
      quit "ERROR: cannot connect to {wsUrl} - check nimxla_plot is running and you can connect that address"

proc updatePlot*(ws: WebSocket, data: JsonNode, layout: JsonNode = nil) =
  ## Send message to websocket server to update the plot.
  let msg = %*{
    "data": data,
    "layout": mergeJson(layout, defaultLayout),
  }
  waitfor ws.send($msg)

proc servePlots*() {.async.} =
  ## Start async http server, serve plot using plotly on localhost and open the default browser on this page.
  let jsonLayout = $defaultLayout
  var connections = newSeq[WebSocket]()

  proc cb(req: Request) {.async.} =
    debug req.reqMethod, " ", req.url.path
    let wsUrl = getWsUrl()
    var contentType = "text/plain"
    var content = ""
    if req.url.path == "/ws":
      try:
        var ws = await newWebSocket(req)
        let id = connections.len
        connections.add ws
        debug &"registered websocket client {id}"
        while ws.readyState == Open:
          let packet = await ws.receiveStrPacket()
          for i, other in connections:
            if ws.key != other.key and other.readyState == Open:
              debug &"send msg from {id} => {i}"
              asyncCheck other.send(packet)
      except WebSocketClosedError:
        debug "websocket closed"
        var toDel: seq[int]
        for i, conn in connections:
          if conn.readyState == Closed: toDel.add i
        for i in toDel:
          debug &"purge connection {i}"
          del(connections, i)
      except WebSocketError:
        error "web socket error:", getCurrentExceptionMsg()
    elif req.url.path == "/plotly.js":
      content = plotlyJS
      contentType = "text/javascript; charset=utf-8"
    elif req.url.path != "/favicon.ico":
      content = fmt indexHtml
      contentType = "text/html; charset=utf-8"
    let headers = {"Content-type": contentType}
    await req.respond(Http200, content, headers.newHttpHeaders())

  var server = newAsyncHttpServer()
  let url = &"http://localhost:{httpPort}/"
  info "listening on ", url
  openDefaultBrowser(url)
  waitFor server.serve(Port(httpPort), cb)

