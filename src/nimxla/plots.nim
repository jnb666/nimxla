## The plots module contains some utilities for plotting graphs and images.
{.warning[BareExcept]:off.}
import std/[asynchttpserver, asyncdispatch, json, strformat, browsers, logging]
import ws
import tensor

type Pixel = array[3, uint8]

const 
  httpPort  = 8080
  wsUrl     = &"ws://localhost:{httpPort}/ws"  
  indexHTML = staticRead("resources/index.html")
  plotlyJS  = staticRead("resources/plotly-2.24.1.min.js")

# default dark layout
let defaultLayout = %*{
  "paper_bgcolor": "#222",
  "plot_bgcolor" : "#111",
  "colorway"     : ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", 
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

proc gridLayout*(rows = 1, cols = 1, ytitle, xtitle: openarray[string] = []): JsonNode =
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

proc annotation(row, col: int, title: string): JsonNode =
  %*{"text": title, "showarrow": false, "xref": "x" & $col, "yref": "y" & $row,
     "x": 0, "y": 0, "xanchor": "left", "yanchor": "top", "bgcolor": "#222", "opacity": 0.8}

proc plotImage*(t: Tensor[uint8], row, col: int): JsonNode =
  ## Convert data from a uint8 tensor in [H, W, C] layout to a plotly image.
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

proc plotImageGrid*(title: string, rows, cols: int, getData: proc(i: int): (Tensor[uint8], string)): (JsonNode, JsonNode) =
  ## Plot grid of images and returns data and loyout Json objects for input to plotly.
  ## getData callBack function recieves the index of the plot on the page and returns the
  ## image data and an optional text label.
  var layout = %*{
    "title": {"text": title},
    "margin": {"t": 30, "l": 30, "r": 0, "b": 0},
    "grid": {"rows": rows, "columns": cols},
    "annotations": [],
  }
  var width, height: int
  var images = %[]
  var i = 0
  for row in 1 .. rows:
    for col in 1 .. cols:
      let (t, label) = getData(i)
      (width, height) = (t.dims[1], t.dims[0])
      images.add t.plotImage(row, col)
      if label != "":
        layout["annotations"].add annotation(row, col, label)
      i += 1

  for i in 1 .. rows:
    layout[yaxis(i)] = %*{"visible": false, "range": [height, 0]}
  for i in 1 .. cols:
    layout[xaxis(i)] = %*{"visible": false, "range": [0, width]}

  return (images, layout)

proc openWebSocket*(): WebSocket =
  ## Blocking open to get client websocket
  waitFor newWebSocket(wsUrl)

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
    var contentType = "text/plain"
    var content = ""
    if req.url.path == "/ws":
      try:
        var ws = await newWebSocket(req)
        let id = connections.len
        connections.add ws
        info &"registered websocket client {id}"
        while ws.readyState == Open:
          let packet = await ws.receiveStrPacket()
          for i, other in connections:
            if ws.key != other.key and other.readyState == Open:
              debug &"send msg from {id} => {i}"
              asyncCheck other.send(packet)
      except WebSocketClosedError:
        info "websocket closed"
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
  echo "listening on ", url
  openDefaultBrowser(url)
  waitFor server.serve(Port(httpPort), cb)

