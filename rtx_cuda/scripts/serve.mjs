// Copyright 2026 The ChromiumRTXCuda Authors. BSD-3-Clause; see ../LICENSE.
import {createServer} from 'node:http';
import {readFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import {resolve,relative,extname} from 'node:path';

const root = fileURLToPath(new URL('..',import.meta.url));
const allowed = new Set(['.html','.js','.cu','.css','.png','.ico']);
export function serve(port=8087) {
  const server = createServer(async (request,response) => {
    try {
      if (request.method !== 'GET') { response.writeHead(405).end(); return; }
      const url = new URL(request.url,'http://127.0.0.1');
      const path = resolve(root,'.'+decodeURIComponent(url.pathname === '/' ? '/demo/index.html' : url.pathname));
      const name = relative(root,path).replaceAll('\\','/');
      if (!(name.startsWith('demo/') || name.startsWith('js/') || name.startsWith('assets/')) || !allowed.has(extname(name)) || name.includes('..')) {
        response.writeHead(404).end(); return;
      }
      const mime = {'.html':'text/html','.js':'text/javascript','.cu':'text/plain','.css':'text/css',
        '.png':'image/png','.ico':'image/vnd.microsoft.icon'}[extname(name)];
      const body = await readFile(path);
      response.writeHead(200,{'Content-Type':`${mime}; charset=utf-8`,'Cache-Control':'no-store',
        'Permissions-Policy':url.searchParams.has('deny') ? 'native-gpu=()' : 'native-gpu=(self)',
        'Content-Security-Policy':"default-src 'self'; script-src 'self'; style-src 'self'; frame-src 'self'; object-src 'none'"});
      response.end(body);
    } catch { if (!response.headersSent) response.writeHead(404); response.end(); }
  });
  return new Promise((resolve,reject) => {
    server.once('error',reject);
    server.listen(port,'127.0.0.1',() => resolve(server));
  });
}
if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const server = await serve(Number(process.env.PORT || 8087));
  console.log(`ChromiumRTXCuda demo: http://127.0.0.1:${server.address().port}/`);
}
