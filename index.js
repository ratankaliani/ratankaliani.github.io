var http = require("http");
var url = require("url");

http.createServer(function(req, res) {
  var pathname = url.parse(req.url).pathname;
  res.writeHead(301,{Location: 'http://ratankaliani.com/'});
  res.end();
}).listen(8888);